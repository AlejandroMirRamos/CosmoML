"""2D Delta-chi2 contours (ML surrogate vs theory)."""
from __future__ import annotations
from collections.abc import Callable
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import concurrent.futures
import multiprocessing

from ..config import CONF_LEVELS_2D


# Module-level so ProcessPoolExecutor can pickle it.
def _eval_point(task):
    func, i_idx, j_idx, kw = task
    return i_idx, j_idx, float(func(**kw))


def predict_grid(model, features: list[str], grid_df: pd.DataFrame,
                 res: int, sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Predict chi2 on a grid and return (smoothed Z, Delta Z).

    Smoothing happens BEFORE subtracting the minimum so that ``argmin(Z_smooth)``
    matches the location actually drawn by the contour.
    """
    Z_raw = model.predict(grid_df[features]).reshape(res, res)
    Z_smooth = gaussian_filter(Z_raw, sigma=sigma) if sigma > 0 else Z_raw
    delta = Z_smooth - Z_smooth.min()
    return Z_smooth, delta


def _build_grid(
    features: list[str],
    x_param: str, y_param: str,
    x_range: tuple[float, float], y_range: tuple[float, float],
    fixed: dict[str, float],
    res: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    xr = np.linspace(*x_range, res)
    yr = np.linspace(*y_range, res)
    XX, YY = np.meshgrid(xr, yr)
    data = {x_param: XX.ravel(), y_param: YY.ravel()}
    for f in features:
        if f not in (x_param, y_param):
            data[f] = np.full(res * res, fixed[f])
    grid = pd.DataFrame(data)[features]
    return xr, yr, XX, YY, grid


def plot_contour_2d(
    model,
    features: list[str],
    *,
    x_param: str, y_param: str,
    x_range: tuple[float, float], y_range: tuple[float, float],
    fixed: dict[str, float],
    theory_fn: Callable[..., float] | None = None,
    global_min_chi2: float | None = None,
    res: int = 200,
    sigma: float = 1.0,
    theory_threshold: float = 50.0,
    theory_step: int = 1,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    save_path: str | Path | None = None,
    show: bool = False,
    figsize: tuple[float, float] = (9, 7),
):
    """Plot Delta-chi2 contours of the ML surrogate, optionally overlaid with theory.

    Parameters
    ----------
    theory_fn : callable | None
        Receives kwargs matching ``features`` and returns chi2. Set to None to
        skip the theory overlay.
    global_min_chi2 : float | None
        If given, use it as the Delta-chi2=0 reference for BOTH ML and theory.
        Pass ``res_opt.fun`` from a prior Nelder-Mead to keep all 2D contours of
        the same model on a consistent baseline. When None, each plot uses the
        local minimum of its own grid as zero.
    theory_threshold : float
        Only evaluate ``theory_fn`` where the ML Delta-chi2 is below this
        threshold (huge speedup; the heavy bit is the theory call).
    theory_step : int
        Subsampling step for the theory grid (1 = every pixel).
    """
    print(f"--- {y_label} vs {x_label}  ({fixed}) ---")
    xr, yr, XX, YY, grid = _build_grid(
        features, x_param, y_param, x_range, y_range, fixed, res
    )

    t0 = time.time()
    Z_ml, _ = predict_grid(model, features, grid, res, sigma)
    t_ml = time.time() - t0

    # Always use the smoothed ML minimum so that Gaussian smoothing does not
    # push d_ml.min() above the confidence levels (which would erase contours).
    # global_min_chi2 is only used for the theory reference below.
    ml_base = float(Z_ml.min())
    d_ml = Z_ml - ml_base

    Z_th = None
    t_th = 0.0
    if theory_fn is not None:
        print("  computing theory (parallel)...")
        t0 = time.time()
        Z_th = np.full((res, res), np.nan)

        tasks = []
        for i in range(0, res, theory_step):
            for j in range(0, res, theory_step):
                if d_ml[i, j] < theory_threshold:
                    kwargs = {x_param: float(xr[j]), y_param: float(yr[i])}
                    for f in features:
                        if f not in (x_param, y_param):
                            kwargs[f] = fixed[f]
                    tasks.append((theory_fn, i, j, kwargs))

        n_cores = max(1, multiprocessing.cpu_count() - 1)
        if tasks:
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
                for i_idx, j_idx, val in executor.map(_eval_point, tasks):
                    Z_th[i_idx:i_idx + theory_step, j_idx:j_idx + theory_step] = val

        t_th = time.time() - t0

        valid = Z_th[~np.isnan(Z_th)]
        if len(valid):
            th_base = global_min_chi2 if global_min_chi2 is not None else valid.min()
            d_th = Z_th - th_base
        else:
            d_th = None
    else:
        d_th = None

    fig, ax = plt.subplots(figsize=figsize)
    vmax = max(15.0, float(np.nanmax(d_ml)) + 1.0)
    ax.contourf(XX, YY, d_ml,
                levels=[0, CONF_LEVELS_2D[0], CONF_LEVELS_2D[1], vmax],
                colors=["#d1eefc", "#e3f4fd", "#f7fbff"], alpha=1)
    ax.contour(XX, YY, d_ml, levels=list(CONF_LEVELS_2D),
               colors="#0044cc", linewidths=2.5)

    if d_th is not None:
        ax.contour(XX, YY, d_th, levels=list(CONF_LEVELS_2D),
                   colors="#cc0000", linewidths=2, linestyles="--")

    i_ml = np.unravel_index(np.argmin(Z_ml), Z_ml.shape)
    ax.scatter(xr[i_ml[1]], yr[i_ml[0]], s=300, c="#0044cc", marker="*",
               label="Min ML", zorder=10)

    if Z_th is not None and np.isfinite(Z_th).any():
        i_th = np.unravel_index(np.nanargmin(Z_th), Z_th.shape)
        ax.scatter(xr[i_th[1]], yr[i_th[0]], s=180, c="#cc0000", marker="x",
                   linewidth=3, label="Min theory", zorder=10)

    ax.set_xlim(x_range); ax.set_ylim(y_range)
    ax.set_xlabel(x_label or x_param, fontsize=13)
    ax.set_ylabel(y_label or y_param, fontsize=13)

    full_title = title
    if Z_th is not None:
        full_title += f"\nML: {t_ml:.2f}s | theory: {t_th:.1f}s"
    ax.set_title(full_title, fontsize=13)
    ax.legend(loc="upper right", fontsize=11)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"  saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return dict(Z_ml=Z_ml, Z_th=Z_th, time_ml=t_ml, time_th=t_th)
