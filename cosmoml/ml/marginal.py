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

from .style import texify, style_getdist, reassert_usetex


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
    proposal_cov: np.ndarray | None = None,
) -> np.ndarray:
    """n_chains independent RW-MH chains, one batched predict per step.

    Phase 1 (0..burn_in):  diagonal proposal, adaptive step size.
                           Skipped if proposal_cov is given — uses it directly.
    Phase 2 (burn_in..):   multivariate Gaussian from empirical covariance,
                           handles correlated posteriors (e.g. w0-wa).

    proposal_cov: optional Hessian covariance (e.g. from Minuit). When provided
    the diagonal phase is skipped and the chain starts with a well-oriented
    proposal, which is critical for degenerate posteriors (e.g. BAO-only w0-wa).

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
    L = None  # Cholesky of proposal cov — set at burn_in (or immediately if proposal_cov given)
    if proposal_cov is not None:
        try:
            L = np.linalg.cholesky(np.array(proposal_cov, dtype=float))
            print(f"  proposal_cov: Hessian covariance accepted — skipping diagonal phase")
        except np.linalg.LinAlgError:
            print("  proposal_cov: Cholesky failed — falling back to diagonal phase")
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
    ranges: dict[str, tuple[float, float]] | None = None,
    axis_limits: dict[str, tuple[float, float]] | None = None,
    legend_label: str | None = None,
) -> matplotlib.figure.Figure:
    """Render a getdist triangle plot.

    ``ranges`` are the prior bounds passed to getdist (they drive the KDE
    boundary correction and hence the contour shape).  ``axis_limits``, when
    given, only controls the displayed ``set_xlim``/``set_ylim`` window, so the
    posterior is computed identically and the figure is merely zoomed.
    ``legend_label`` adds a legend even for a single-dataset plot.
    """
    try:
        import getdist
        import getdist.plots
    except ImportError as e:
        raise ImportError("getdist required: pip install getdist") from e

    mc_ranges = {f: list(r) for f, r in ranges.items()} if ranges else None
    mc = getdist.MCSamples(
        samples=samples,
        names=features,
        labels=str_labels,
        label=legend_label,
        ranges=mc_ranges,
        settings={"smooth_scale_2D": smooth_scale, "smooth_scale_1D": smooth_scale},
    )
    g = getdist.plots.get_subplot_plotter()
    style_getdist(g)
    reassert_usetex()  # getdist's style resets text.usetex; restore our choice
    g.triangle_plot(
        mc,
        filled=True,
        contour_colors=["#0044cc"],
        markers=markers,
        marker_args={"ls": "--", "color": "gray", "lw": 1.5, "alpha": 0.8}
        if markers else None,
    )
    # Display window: axis_limits zooms the view without touching the priors above.
    lims = axis_limits if axis_limits is not None else ranges
    if lims:
        ndim = len(features)
        for i in range(ndim):
            for j in range(i + 1):
                ax = g.subplots[i][j]
                if ax is None:
                    continue
                ax.set_xlim(*lims.get(features[j], ranges[features[j]]))
                if i != j:
                    ax.set_ylim(*lims.get(features[i], ranges[features[i]]))
    if legend_label:
        # Place a compact legend inside the empty upper-right cell of the
        # triangle (right-most column x top row) so it never overlaps the
        # marginal panels; long "A & B" / "A + B" labels are wrapped onto two
        # lines so they stay inside the cell and don't spill over the panels.
        from matplotlib.patches import Patch
        disp_label = (legend_label.replace(" & ", "\n& ")
                                  .replace(" + ", "\n+ "))
        disp_label = texify(disp_label)
        handle = Patch(facecolor="#0044cc", edgecolor="none")
        ndim = len(features)
        try:
            pos_top = g.subplots[0][0].get_position()
            pos_right = g.subplots[ndim - 1][ndim - 1].get_position()
            legax = g.fig.add_axes([pos_right.x0, pos_top.y0,
                                    pos_right.width, pos_top.height])
            legax.axis("off")
            legax.legend([handle], [disp_label], loc="center", frameon=True,
                         fontsize=12, handlelength=1.2, handleheight=1.1,
                         borderpad=0.4)
        except Exception:
            g.fig.legend([handle], [disp_label], loc="upper right",
                         bbox_to_anchor=(0.99, 0.99), frameon=True, fontsize=12)
    if title:
        g.fig.suptitle(texify(title), fontsize=19, y=1.02)
    return g.fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _make_gpu_predict_fn(model, n_trees: int = 0):
    """GPU booster predict function from a trained model."""
    from .train import LogChi2Model

    irange = (0, n_trees) if n_trees > 0 else (0, 0)
    if isinstance(model, LogChi2Model):
        booster = model.raw_model.get_booster().copy()
        booster.set_param({"device": "cuda"})
        y_min = model.y_min
        def predict_fn(arr: np.ndarray) -> np.ndarray:
            log_y = booster.inplace_predict(arr.astype(np.float32), iteration_range=irange)
            return (10.0 ** log_y) - 1.0 + y_min
    else:
        booster = model.get_booster().copy()
        booster.set_param({"device": "cuda"})
        def predict_fn(arr: np.ndarray) -> np.ndarray:
            return booster.inplace_predict(arr.astype(np.float32), iteration_range=irange)
    return predict_fn


def run_mcmc_and_getdist(
    model,
    features: list[str],
    ranges: dict,
    ref: dict,
    section: str,
    labels: dict[str, str],
    *,
    markers: dict[str, float] | None = None,
    title: str = "",
    proposal_cov: np.ndarray | None = None,
    n_trees_mcmc: int = 0,
    ess_target: int = 10_000,
    figures_dir=None,
    axis_limits: dict | None = None,
    legend_label: str | None = None,
    show: bool = True,
) -> np.ndarray:
    """Run GPU-boosted MCMC and render a getdist corner plot.

    Parameters
    ----------
    model : LogChi2Model or XGBRegressor
    features, ranges, ref : parameter names, bounds, best-fit
    section : prefix used for the saved figure filename
    labels : dict param -> LaTeX label
    figures_dir : Path or None — if set, saves PNG there
    """
    import pathlib

    lows   = np.array([ranges[f][0] for f in features])
    highs  = np.array([ranges[f][1] for f in features])
    center = np.array([ref[f] for f in features])
    str_labels = [(labels[f] if labels and f in labels else f).replace("$", "")
                  for f in features]

    predict_fn = _make_gpu_predict_fn(model, n_trees=n_trees_mcmc)
    samples = _parallel_mcmc(
        predict_fn, lows, highs, center, len(features),
        n_chains=1024, n_steps=10_000, burn_in=500, seed=42,
        ess_target=ess_target, proposal_cov=proposal_cov,
    )

    fig = _render_getdist(samples, features, str_labels, markers, title,
                          smooth_scale=0.5, ranges=ranges,
                          axis_limits=axis_limits, legend_label=legend_label)
    if figures_dir is not None:
        save_path = pathlib.Path(figures_dir) / f"{section}_getdist.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return samples


def plot_getdist_comparison(
    samples_list: list[np.ndarray],
    dataset_labels: list[str],
    features: list[str],
    labels: dict[str, str],
    *,
    markers: dict[str, float] | None = None,
    title: str = "",
    save_path=None,
    filled: list[bool] | bool | None = None,
    ranges: dict | None = None,
    axis_limits: dict | None = None,
    show: bool = True,
) -> matplotlib.figure.Figure:
    """Overlay multiple MCMC chains in one getdist triangle plot.

    ``ranges`` are the getdist priors (contour shape); ``axis_limits`` only
    zooms the displayed window without changing the computed posteriors.
    """
    try:
        import getdist
        import getdist.plots
    except ImportError as e:
        raise ImportError("getdist required: pip install getdist") from e

    import pathlib

    COLORS = ["#0044cc", "#cc0000", "#009933", "#cc6600",
              "#9900cc", "#008080", "#cc0099", "#806000"]
    n = len(samples_list)
    if filled is None:
        filled = [True] + [False] * (n - 1)
    elif isinstance(filled, bool):
        filled = [filled] * n

    str_labels = [(labels[f] if labels and f in labels else f).replace("$", "")
                  for f in features]
    mc_ranges = {f: list(r) for f, r in ranges.items()} if ranges else None
    mc_list = [
        getdist.MCSamples(
            samples=s, names=features, labels=str_labels, label=dl,
            ranges=mc_ranges,
            settings={"smooth_scale_2D": 0.5, "smooth_scale_1D": 0.5},
        )
        for s, dl in zip(samples_list, dataset_labels)
    ]
    g = getdist.plots.get_subplot_plotter()
    style_getdist(g)
    reassert_usetex()  # getdist's style resets text.usetex; restore our choice
    g.triangle_plot(
        mc_list, filled=filled, contour_colors=COLORS[:n],
        contour_lws=[2.0] * n, markers=markers,
        marker_args={"ls": "--", "color": "gray", "lw": 1.5, "alpha": 0.8}
        if markers else None,
        legend_labels=[texify(dl) for dl in dataset_labels], legend_loc="upper right",
    )
    lims = axis_limits if axis_limits is not None else ranges
    if lims:
        for i, fi in enumerate(features):
            for j, fj in enumerate(features[:i + 1]):
                ax = g.subplots[i][j]
                if ax is None:
                    continue
                ax.set_xlim(*lims.get(fj, ranges[fj]))
                if i != j:
                    ax.set_ylim(*lims.get(fi, ranges[fi]))
    if title:
        g.fig.suptitle(texify(title), fontsize=19, y=1.02)
    if save_path is not None:
        sp = pathlib.Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        g.fig.savefig(sp, dpi=200, bbox_inches="tight")
        print(f"  Saved: {sp}")
    if show:
        plt.show()
    else:
        plt.close(g.fig)
    return g.fig


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
    proposal_cov: np.ndarray | None = None,
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
        proposal_cov=proposal_cov,
    )

    fig = _render_getdist(samples, features, str_labels, markers, title, smooth_scale,
                          ranges=ranges)

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
