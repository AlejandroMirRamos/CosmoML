"""Contornos 2D Δχ² (ML vs teoría)."""
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

# --- FUNCIÓN AUXILIAR PARA MULTIPROCESSING ---
def _eval_point(task):
    """Desempaqueta la función teórica y las coordenadas y evalúa."""
    func, i_idx, j_idx, kw = task
    return i_idx, j_idx, float(func(**kw))

def predict_grid(model, features: list[str], grid_df: pd.DataFrame,
                 res: int, sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Predice χ² en la malla y devuelve (Z_ML_suavizado, ΔZ_ML)."""
    # 1. Predecir en la malla
    Z_raw = model.predict(grid_df[features]).reshape(res, res)
    
    # 2. Suavizar la superficie PRIMERO
    if sigma > 0:
        Z_smooth = gaussian_filter(Z_raw, sigma=sigma)
    else:
        Z_smooth = Z_raw
        
    # 3. Calcular el Δχ² DESPUÉS de suavizar para garantizar que el mínimo sea 0.0
    delta = Z_smooth - Z_smooth.min()
    
    # Devolvemos Z_smooth para que el np.argmin del plot_contour_2d coincida con los contornos
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
    global_min_chi2: float | None = None,  # <--- NUEVO
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
    """Dibuja contornos Δχ²(ML) + (opcional) contornos Δχ²(teoría) y best-fits.

    theory_fn : callable
        Función que recibe **kwargs con todos los `features` y devuelve χ².
        Si None se omite la capa teórica.
    theory_threshold : float
        Sólo se calcula la teoría donde Δχ²_ML < threshold (acelera mucho).
    theory_step : int
        Subsampling de la malla teórica (1 = todos los puntos).
    """
    print(f"--- {y_label} vs {x_label}  ({fixed}) ---")
    xr, yr, XX, YY, grid = _build_grid(
        features, x_param, y_param, x_range, y_range, fixed, res
    )

    t0 = time.time()
    # Tu predict_grid ya no debe restar el mínimo internamente. 
    # Asegúrate de que predict_grid devuelva (Z_smooth, Z_smooth), o hazlo aquí:
    Z_ml, _ = predict_grid(model, features, grid, res, sigma)
    t_ml = time.time() - t0

    # Determinar el cero absoluto (ML)
    ml_base = global_min_chi2 if global_min_chi2 is not None else Z_ml.min()
    d_ml = Z_ml - ml_base

    # Capa teórica
    Z_th = None
    t_th = 0.0
    if theory_fn is not None:
        print("  calculando teoría (en paralelo)...")
        t0 = time.time()
        
        Z_th = np.full((res, res), np.nan)
        
        # 1. Recopilar tareas (AÑADIMOS theory_fn AL PAQUETE)
        tasks = []
        for i in range(0, res, theory_step):
            for j in range(0, res, theory_step):
                if d_ml[i, j] < theory_threshold:
                    kwargs = {x_param: float(xr[j]), y_param: float(yr[i])}
                    for f in features:
                        if f not in (x_param, y_param):
                            kwargs[f] = fixed[f]
                    # Metemos func, i, j, y kwargs en la tupla
                    tasks.append((theory_fn, i, j, kwargs)) 
        
        # 2. Ejecutar los cálculos en paralelo
        n_cores = max(1, multiprocessing.cpu_count() - 1)
        if len(tasks) > 0:
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
                # Usamos la función _eval_point que está arriba del todo
                for i_idx, j_idx, val in executor.map(_eval_point, tasks):
                    Z_th[i_idx:i_idx + theory_step, j_idx:j_idx + theory_step] = val

        t_th = time.time() - t0
        
        # 3. Restar el mínimo
        valid = Z_th[~np.isnan(Z_th)]
        if len(valid):
            th_base = global_min_chi2 if global_min_chi2 is not None else valid.min()
            d_th = Z_th - th_base
        else:
            d_th = None
    else:
        d_th = None

    # Plot
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

    # Best-fit ML
    i_ml = np.unravel_index(np.argmin(Z_ml), Z_ml.shape)
    ax.scatter(xr[i_ml[1]], yr[i_ml[0]], s=300, c="#0044cc", marker="*",
               label="Min ML", zorder=10)

    # Best-fit teoría (np.nanargmin ignora celdas no calculadas)
    if Z_th is not None and np.isfinite(Z_th).any():
        i_th = np.unravel_index(np.nanargmin(Z_th), Z_th.shape)
        ax.scatter(xr[i_th[1]], yr[i_th[0]], s=180, c="#cc0000", marker="x",
                   linewidth=3, label="Min Teoría", zorder=10)

    ax.set_xlim(x_range); ax.set_ylim(y_range)
    ax.set_xlabel(x_label or x_param, fontsize=13)
    ax.set_ylabel(y_label or y_param, fontsize=13)

    full_title = title
    if Z_th is not None:
        full_title += f"\nML: {t_ml:.2f}s | Teoría: {t_th:.1f}s"
    ax.set_title(full_title, fontsize=13)
    ax.legend(loc="upper right", fontsize=11)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"  guardado: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return dict(Z_ml=Z_ml, Z_th=Z_th, time_ml=t_ml, time_th=t_th)
