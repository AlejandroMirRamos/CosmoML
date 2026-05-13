"""Marginalized posterior corner plots from an XGBoost chi2 emulator.

Sampling: parallel adaptive MCMC (n_chains independent chains, one batched
predict call per step).
Render:    getdist (KDE-smoothed, Planck/DESI style).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Predict helper
# ---------------------------------------------------------------------------

def _make_predict_fn(
    model,
    features: list[str],
    n_trees: int | None = None,
):
    """Build a (numpy_array) -> chi2_array using a CPU copy of the booster.

    CPU copy is faster than GPU for batch sizes ≤512 (typical n_chains) because
    kernel-launch latency dominates on WSL2+CUDA at those batch sizes.
    XGBoost parallelises over trees, so it uses all available cores by default.

    n_trees: if not None, use only the first n_trees trees (iteration_range).
    """
    from .train import LogChi2Model
    import xgboost as xgb

    it_range = (0, n_trees) if n_trees is not None else (0, 0)  # (0,0) = all trees

    if isinstance(model, LogChi2Model):
        cpu_booster = model.raw_model.get_booster().copy()
        cpu_booster.set_param({"device": "cpu"})
        y_min = model.y_min

        def predict_fn(arr: np.ndarray) -> np.ndarray:
            log_y = cpu_booster.inplace_predict(
                arr.astype(np.float32), iteration_range=it_range
            )
            return (10.0 ** log_y) - 1.0 + y_min

    else:
        cpu_booster = model.get_booster().copy()
        cpu_booster.set_param({"device": "cpu"})

        def predict_fn(arr: np.ndarray) -> np.ndarray:
            return cpu_booster.inplace_predict(
                arr.astype(np.float32), iteration_range=it_range
            )

    return predict_fn


# ---------------------------------------------------------------------------
# Parallel adaptive MCMC
# ---------------------------------------------------------------------------

def _sokal_tau(post_chain: np.ndarray, n_chains_sample: int = 128) -> float:
    """Max integrated autocorrelation time over dims (Sokal windowing)."""
    n_post, n_chains, ndim = post_chain.shape
    tau_max = 1.0
    for d in range(ndim):
        for c in range(min(n_chains, n_chains_sample)):
            x = post_chain[:, c, d].astype(float)
            x -= x.mean()
            var = np.dot(x, x) / n_post
            if var < 1e-30:
                continue
            acf = np.correlate(x, x, mode='full')[n_post - 1:] / (var * n_post)
            tau = 1.0
            for k in range(1, n_post // 2):
                tau += 2.0 * acf[k]
                if k > 5 * tau:
                    break
            tau_max = max(tau_max, tau)
    return tau_max


def _parallel_mcmc(
    predict_fn,
    lows: np.ndarray,
    highs: np.ndarray,
    center: np.ndarray,
    ndim: int,
    n_chains: int,
    n_steps: int,
    burn_in: int,
    seed: int,
    ess_target: int | None = 5_000,
    progress_every: int = 200,
) -> np.ndarray:
    """n_chains independent RW-MH chains, one batched predict per step.

    Phase 1 (0..burn_in):  diagonal proposal, adaptive step size.
    Phase 2 (burn_in..):   multivariate Gaussian from empirical covariance,
                           handles correlated posteriors (e.g. w0-wa).

    ess_target: stop early once ESS ≥ ess_target (checked every progress_every
    steps). Set to None to always run the full n_steps.
    """
    rng = np.random.default_rng(seed)

    sigma_init = 0.02 * (highs - lows)
    pos = np.clip(
        center + rng.normal(0, 1, (n_chains, ndim)) * sigma_init,
        lows + 1e-10, highs - 1e-10,
    )

    chi2 = predict_fn(pos)
    log_p = -0.5 * chi2

    chain = np.empty((n_steps, n_chains, ndim), dtype=np.float32)
    chain[0] = pos

    step_scale = 0.05 * (highs - lows)
    L = None  # Cholesky of proposal cov — set at burn_in
    n_accepted = 0
    t0 = time.time()
    stopped_at = n_steps

    for i in range(1, n_steps):
        # Proposal
        if L is None:
            proposal = pos + rng.normal(0, step_scale, (n_chains, ndim))
        else:
            z = rng.standard_normal((n_chains, ndim))
            proposal = pos + (z @ L.T) * (2.38 / np.sqrt(ndim))

        # One batched predict call
        in_box = np.all((proposal >= lows) & (proposal <= highs), axis=1)
        log_p_new = np.full(n_chains, -np.inf)
        if in_box.any():
            chi2_new = np.asarray(predict_fn(proposal[in_box]), dtype=float)
            log_p_new[in_box] = -0.5 * chi2_new

        # Accept / reject (vectorised)
        accept = np.log(rng.uniform(size=n_chains)) < (log_p_new - log_p)
        pos = np.where(accept[:, None], proposal, pos)
        log_p = np.where(accept, log_p_new, log_p)
        n_accepted += int(accept.sum())
        chain[i] = pos

        # Tune step size (diagonal phase only)
        if i < 200 and i % 50 == 0 and L is None:
            acc = n_accepted / (i * n_chains)
            step_scale *= 0.7 if acc < 0.15 else (1.4 if acc > 0.40 else 1.0)

        # Switch to empirical covariance at burn_in
        if i == burn_in:
            past = chain[:burn_in].reshape(-1, ndim).astype(float)
            cov = np.cov(past.T) + 1e-8 * np.eye(ndim)
            try:
                L = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                pass

        # Progress + optional ESS early stop
        if progress_every and i % progress_every == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            acc_so_far = n_accepted / (i * n_chains)
            phase = "diagonal" if L is None else "multivar"
            if i > burn_in and ess_target is not None:
                tau = _sokal_tau(chain[burn_in:i + 1])
                n_post = i - burn_in
                ess = n_post * n_chains / tau
                eta = max(0.0, (ess_target - ess) * tau / (n_chains * rate))
                print(f"  step {i:>5}/{n_steps}  |  {rate:.0f} it/s  |  "
                      f"ETA ≤{eta:.0f}s  |  acc {acc_so_far:.2f}  |  "
                      f"τ={tau:.1f}  ESS={ess:.0f}/{ess_target}  |  {phase}")
                if ess >= ess_target:
                    stopped_at = i + 1
                    break
            else:
                eta = (n_steps - i) / rate
                print(f"  step {i:>5}/{n_steps}  |  {rate:.0f} it/s  |  "
                      f"ETA {eta:.0f}s  |  acc {acc_so_far:.2f}  |  {phase}")

    acc_rate = n_accepted / (stopped_at * n_chains)
    elapsed = time.time() - t0
    print(f"  done: {elapsed:.1f}s  |  {stopped_at/elapsed:.0f} it/s  |  "
          f"acceptance {acc_rate:.3f}  |  "
          f"{'diagonal' if L is None else 'multivariate'} proposal")

    post_chain = chain[burn_in:stopped_at]
    samples = post_chain.reshape(-1, ndim).astype(float)
    n_post = stopped_at - burn_in
    tau_final = _sokal_tau(post_chain)
    ess_final = n_post * n_chains / tau_final
    print(f"  flat chain: {len(samples):,} samples "
          f"({n_chains} chains × {n_post} steps)  |  "
          f"τ_max={tau_final:.1f}  |  ESS={ess_final:.0f}")
    return samples


# ---------------------------------------------------------------------------
# getdist render
# ---------------------------------------------------------------------------

def _render_getdist(
    samples: np.ndarray,
    features: list[str],
    str_labels: list[str],
    markers: dict[str, float] | None,
    title: str,
    smooth_scale: float,
) -> matplotlib.figure.Figure:
    try:
        import getdist
        import getdist.plots
    except ImportError as e:
        raise ImportError("getdist required: pip install getdist") from e

    mc = getdist.MCSamples(
        samples=samples,
        names=features,
        labels=str_labels,
        settings={"smooth_scale_2D": smooth_scale, "smooth_scale_1D": smooth_scale},
    )
    g = getdist.plots.get_subplot_plotter()
    g.triangle_plot(
        mc,
        filled=True,
        contour_colors=["#0044cc"],
        markers=markers,
        marker_args={"ls": "--", "color": "gray", "lw": 1.5, "alpha": 0.8}
        if markers else None,
    )
    if title:
        g.fig.suptitle(title, fontsize=12, y=1.01)
    return g.fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_corner_marginal(
    model,
    features: list[str],
    ranges: dict[str, tuple[float, float]],
    *,
    labels: dict[str, str] | None = None,
    ref: dict[str, float] | None = None,
    markers: dict[str, float] | None = None,
    title: str = "",
    n_chains: int = 512,
    n_steps: int = 5_000,
    burn_in: int = 400,
    ess_target: int | None = 5_000,
    n_trees: int | None = None,
    seed: int = 42,
    smooth_scale: float = 0.5,
    save_path: str | Path | None = None,
    show: bool = True,
) -> np.ndarray:
    """Marginalized corner plot (getdist) from an XGBoost chi2 emulator.

    Uses parallel adaptive MCMC: ``n_chains`` independent chains evaluated
    with a single batched XGBoost predict call per step (CPU copy of the
    booster — faster than GPU for these batch sizes due to kernel-launch
    overhead).

    Parameters
    ----------
    model : LogChi2Model or XGBRegressor
        Trained emulator. ``predict(X)`` must return LINEAR chi2.
    features : list[str]
        Parameter names matching model input columns.
    ranges : dict param -> (lo, hi)
        Bounding box (uniform prior) and axis limits.
    labels : dict param -> LaTeX str, optional
    ref : dict param -> float, optional
        Best-fit from Minuit (used to initialise walkers; not drawn).
    markers : dict param -> float, optional
        Reference lines drawn as dashed grey lines on all panels.
        Example: ``{'w0': -1.0, 'wa': 0.0}`` for ΛCDM.
    n_chains : int
        Parallel chains = batch size per predict call. CPU booster copy is
        faster at ≤512 rows due to kernel-launch overhead (especially on WSL2).
    n_steps : int
        Hard cap on steps per chain. With ess_target set, the chain usually
        stops well before this limit.
    ess_target : int | None
        Stop as soon as the effective sample size (ESS = n_post*n_chains/τ_max)
        reaches this value. 5_000 is plenty for getdist in 3D.
        Set to None to always run the full n_steps.
    n_trees : int | None
        If set, use only the first n_trees trees for each predict call.
        Speeds up MCMC significantly (e.g. 500 out of 3000 trees ≈ 6×).
        Run an A/B visual check before using this in production.
    smooth_scale : float
        getdist KDE bandwidth multiplier. Raise to suppress tree artefacts.

    Returns
    -------
    samples : ndarray, shape ((n_steps - burn_in) * n_chains, ndim)
    """
    ndim = len(features)
    lows = np.array([ranges[f][0] for f in features])
    highs = np.array([ranges[f][1] for f in features])
    center = (np.array([ref[f] for f in features]) if ref is not None
              else (lows + highs) / 2.0)

    str_labels = [
        (labels[f] if labels and f in labels else f).replace("$", "")
        for f in features
    ]

    trees_str = f", n_trees={n_trees}" if n_trees is not None else ""
    print(f"--- corner marginal: {n_chains} chains × {n_steps} steps "
          f"(burn-in {burn_in}, ndim={ndim}{trees_str}) ---")
    print("  loading CPU booster copy...")
    predict_fn = _make_predict_fn(model, features, n_trees=n_trees)

    samples = _parallel_mcmc(
        predict_fn, lows, highs, center, ndim,
        n_chains, n_steps, burn_in, seed,
        ess_target=ess_target,
    )

    fig = _render_getdist(samples, features, str_labels, markers, title, smooth_scale)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return samples
