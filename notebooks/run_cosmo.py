"""CosmoML 06 — pipeline completo para GPU server.

Equivalente al notebook 06_Paper.ipynb, adaptado para correr headless.
Outputs en CosmoML/outputs/paper/{datasets,models,figures,chains}/

Uso:
    cd /home/aleja/PhysicsML/CosmoML/notebooks
    /path/to/CosmoML/.venv/bin/python run_cosmo.py

GPU forzada en todo: XGBoost (device=cuda en hp_overrides) y MCMC (booster GPU directo).

Flags al inicio del archivo:
    FORCE_RETRAIN  — forzar regeneración de datasets y reentrenamiento
"""

# ── Backend no-interactivo ANTES de cualquier import de matplotlib ──
import matplotlib
matplotlib.use("Agg")

import sys
import os
import json
import time
import concurrent.futures
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import getdist
import getdist.plots
from iminuit import Minuit

# ── Importar cosmoml desde CosmoML/ ───────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
COSMO_ROOT  = SCRIPT_DIR.parent.parent / "CosmoML"
if str(COSMO_ROOT) not in sys.path:
    sys.path.insert(0, str(COSMO_ROOT))

from cosmoml.data import load_pantheon_plus, load_des_2024, load_des_2025, load_desi_bao
from cosmoml.theory import chi2_sne, chi2_sne_des, chi2_bao, chi2_joint
from cosmoml.priors import planck_prior_chi2
from cosmoml.sampling import build_chi2_dataset, load_or_build, _chi2_worker
from cosmoml.ml import (
    train_xgb, plot_learning_curve,
    shap_summary, shap_waterfall, shap_dependence_all,
    use_paper_style,
)
from cosmoml.ml.marginal import _parallel_mcmc, _render_getdist

try:
    from cosmoml.theory.jax_theory import make_chi2_gpu_fn as _make_chi2_gpu_fn
    _JAX_THEORY_OK = True
except ImportError:
    _JAX_THEORY_OK = False

use_paper_style()

# ── GPU predict wrapper (bypasa la copia CPU que usa plot_corner_marginal) ─────
def _make_gpu_predict_fn(model):
    from cosmoml.ml.train import LogChi2Model
    if isinstance(model, LogChi2Model):
        booster = model.raw_model.get_booster().copy()
        booster.set_param({"device": "cuda"})
        y_min = model.y_min
        def predict_fn(arr: np.ndarray) -> np.ndarray:
            log_y = booster.inplace_predict(arr.astype(np.float32), iteration_range=(0, 0))
            return (10.0 ** log_y) - 1.0 + y_min
    else:
        booster = model.get_booster().copy()
        booster.set_param({"device": "cuda"})
        def predict_fn(arr: np.ndarray) -> np.ndarray:
            return booster.inplace_predict(arr.astype(np.float32), iteration_range=(0, 0))
    return predict_fn

# ── Paths (todos en CosmoML/outputs/paper/) ────────────────────────────────────
PAPER_DIR    = SCRIPT_DIR.parent / "outputs" / "paper"
DATASETS_DIR = PAPER_DIR / "datasets"
MODELS_DIR   = PAPER_DIR / "models"
FIGURES_DIR  = PAPER_DIR / "figures"
CHAINS_DIR   = PAPER_DIR / "chains"
for _d in (DATASETS_DIR, MODELS_DIR, FIGURES_DIR, CHAINS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Flags ──────────────────────────────────────────────────────────────────────
FORCE_RETRAIN = False   # True → regenera datasets + reentrena + re-corre MCMC

# ── Parámetros globales ────────────────────────────────────────────────────────
GLOBAL_RANGES = {
    "Om": (0.1,   0.9),
    "H0": (20.0, 100.0),
    "w0": (-3.0,  0.2),
    "wa": (-3.0,  2.0),
}

LABELS = {
    "Om": r"$\Omega_m$",
    "H0": r"$H_0$",
    "w0": r"$w_0$",
    "wa": r"$w_a$",
}

MARKERS_DE = {"w0": -1.0, "wa": 0.0}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def locate_bestfit(chi2_fn, features, ranges):
    init = {f: (ranges[f][0] + ranges[f][1]) / 2.0 for f in features}
    if "w0" in features:
        init["w0"] = -1.0
    if "wa" in features:
        init["wa"] = 0.0
    if "Om" in features:
        init["Om"] = 0.3
    if "H0" in features:
        init["H0"] = 68.0
    m = Minuit(chi2_fn, **init)
    m.limits = [ranges[f] for f in features]
    m.migrad()
    REF = {f: float(m.values[f]) for f in features}
    print(f"  Best-fit : {', '.join(f'{f}={v:.4f}' for f, v in REF.items())}")
    print(f"  chi2_min : {m.fval:.2f}")
    return REF, m.fval


def build_dataset(chi2_fn, section, features, ranges):
    csv_path = DATASETS_DIR / f"{section}_dataset.csv"
    REF, chi2_min = locate_bestfit(chi2_fn, features, ranges)
    ndim = len(features)

    def builder():
        if ndim == 2:
            slices = [
                {features[0]: ranges[features[0]], features[1]: REF[features[1]], "_n": 15_000},
                {features[1]: ranges[features[1]], features[0]: REF[features[0]], "_n": 15_000},
            ]
            n_random = 50_000
        else:
            slices = []
            for _i in range(ndim):
                for _j in range(_i + 1, ndim):
                    fi, fj = features[_i], features[_j]
                    fixed  = {f: REF[f] for f in features if f not in (fi, fj)}
                    slices.append({fi: ranges[fi], fj: ranges[fj], **fixed, "_n": 20_000})
            n_random = 80_000
        return build_chi2_dataset(
            chi2_fn=chi2_fn,
            param_names=features,
            slices=slices,
            random_box={f: ranges[f] for f in features},
            n_random=n_random,
            save_to=csv_path,
            seed=42,
        )

    df = load_or_build(csv_path, builder, force=FORCE_RETRAIN)
    print(f"  Dataset  : {len(df):,} rows | chi2 [{df['chi2'].min():.2f}, {df['chi2'].max():.2f}]")
    return df, REF


def train_and_shap(df, features, section, title=""):
    model, info = train_xgb(
        df, features=features,
        log_target=True,
        hp_overrides=dict(n_estimators=5000, learning_rate=0.03, max_depth=10, device="cuda"),
        cache_path=MODELS_DIR / f"{section}_model.ubj",
        force_retrain=FORCE_RETRAIN,
    )
    plot_learning_curve(
        info,
        title=f"{title} — Learning Curve (R²={info['r2']:.5f})",
        save_path=FIGURES_DIR / f"{section}_learning_curve.png",
        show=False,
    )
    shap_v, X_s = shap_summary(
        model, info["X_val"],
        title=f"{title} — SHAP",
        save_dir=FIGURES_DIR,
        prefix=section,
        show=False,
    )
    shap_waterfall(
        shap_v, idx=0,
        title=f"{title} — SHAP waterfall",
        save_path=FIGURES_DIR / f"{section}_shap_waterfall.png",
        show=False,
    )
    shap_dependence_all(shap_v, X_s, save_dir=FIGURES_DIR, prefix=section, show=False)
    return model, info, shap_v, X_s


def run_mcmc_and_getdist(model, features, ranges, ref, section,
                         labels, markers=None, title=""):
    """Retorna (samples, meta_dict) donde meta tiene timing de la cadena ML."""
    chain_path  = CHAINS_DIR  / f"{section}_samples.npy"
    figure_path = FIGURES_DIR / f"{section}_getdist.png"

    if chain_path.exists() and not FORCE_RETRAIN:
        samples = np.load(chain_path)
        print(f"  [ML]  Cargado chain : {chain_path}  ({len(samples):,} muestras)")
        meta = {"cached": True, "wall_s": None, "n_steps_actual": None,
                "ess_final": None, "tau_max": None, "n_samples": len(samples)}
    else:
        print("  [ML]  Cargando booster GPU...")
        predict_fn = _make_gpu_predict_fn(model)
        lows   = np.array([ranges[f][0] for f in features])
        highs  = np.array([ranges[f][1] for f in features])
        center = np.array([ref[f] for f in features]) if ref is not None else (lows + highs) / 2.0
        t0 = time.perf_counter()
        samples = _parallel_mcmc(
            predict_fn, lows, highs, center, len(features),
            n_chains=1024, n_steps=10_000, burn_in=500, seed=42, ess_target=10_000,
        )
        wall = time.perf_counter() - t0
        np.save(chain_path, samples)
        print(f"  [ML]  Chain guardado: {chain_path}  ({wall:.1f}s)")
        meta = {"cached": False, "wall_s": round(wall, 2), "n_steps_actual": None,
                "ess_final": None, "tau_max": None, "n_samples": len(samples)}

    # Render getdist (siempre, aunque venga de caché)
    fig = _render_getdist(
        samples, features,
        [labels.get(f, f).replace("$", "") for f in features],
        markers, title, smooth_scale=0.5, ranges=ranges,
    )
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ML]  Guardado: {figure_path}")
    return samples, meta


def plot_getdist_comparison(samples_list, dataset_labels, features, labels,
                             markers=None, title="", save_path=None,
                             filled=None, ranges=None):
    """Overlay triangle plots for all chains in samples_list.

    filled: bool or list of bool per chain. Default: True for first, False for
    the rest so overlapping contours remain visible (outline-only for secondary).
    ranges: dict feature -> (lo, hi) — sets axis limits and prior boundaries.
    """
    COLORS = ["#0044cc", "#cc0000", "#009933", "#cc6600"]
    n = len(samples_list)
    if filled is None:
        filled = [True] + [False] * (n - 1)
    elif isinstance(filled, bool):
        filled = [filled] * n
    str_labels = [labels.get(f, f).replace("$", "") for f in features]
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
    g.triangle_plot(
        mc_list, filled=filled,
        contour_colors=COLORS[:n],
        contour_lws=[2.0] * n,
        markers=markers,
        marker_args={"ls": "--", "color": "gray", "lw": 1.5, "alpha": 0.8}
        if markers else None,
        legend_labels=dataset_labels,
        legend_loc="upper right",
    )
    if ranges:
        ndim = len(features)
        for i in range(ndim):
            for j in range(i + 1):
                ax = g.subplots[i][j]
                if ax is None:
                    continue
                ax.set_xlim(*ranges[features[j]])
                if i != j:
                    ax.set_ylim(*ranges[features[i]])
    if title:
        g.fig.suptitle(title, fontsize=13, y=1.01)
    if save_path:
        g.fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Guardado: {save_path}")
    plt.close(g.fig)
    return g.fig


def _make_theory_predict_fn(chi2_fn, features: list[str],
                             executor: concurrent.futures.ProcessPoolExecutor,
                             n_cores: int):
    """Devuelve predict_fn(arr) -> chi2_array usando el executor abierto.

    El patrón de tasks es idéntico al de cosmoml/sampling.py:_chi2_worker.
    _parallel_mcmc espera chi2 lineal (hace log_p = -0.5*chi2 internamente).
    """
    chunksize = max(1, 1024 // (n_cores * 4))

    def predict_fn(arr: np.ndarray) -> np.ndarray:
        n = len(arr)
        tasks = [
            (chi2_fn, {f: float(arr[i, j]) for j, f in enumerate(features)})
            for i in range(n)
        ]
        return np.array(list(executor.map(_chi2_worker, tasks,
                                          chunksize=chunksize)), dtype=float)
    return predict_fn


def run_theory_mcmc(chi2_fn, features, ranges, ref, section,
                    labels, markers=None, title=""):
    """MCMC teórico paralelo CPU emparejado con run_mcmc_and_getdist.

    Retorna (flat_samples, meta_dict).
    """
    chain_path  = CHAINS_DIR  / f"{section}_samples_theory.npy"
    figure_path = FIGURES_DIR / f"{section}_getdist_theory.png"

    n_cores  = max(1, multiprocessing.cpu_count() - 1)
    lows     = np.array([ranges[f][0] for f in features])
    highs    = np.array([ranges[f][1] for f in features])
    center   = np.array([ref[f] for f in features]) if ref else (lows + highs) / 2.0

    th_cached = chain_path.exists() and not FORCE_RETRAIN
    if th_cached:
        print(f"  [TH]  Usando caché: {chain_path}")
        samples = np.load(chain_path)
        meta = {"cached": True, "wall_s": None, "n_steps_actual": None,
                "ess_final": None, "tau_max": None, "n_samples": len(samples)}
    else:
        print(f"  [TH]  MCMC teórico ({n_cores} cores CPU)...")
        t0 = time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
            predict_fn = _make_theory_predict_fn(chi2_fn, features, executor, n_cores)
            samples = _parallel_mcmc(
                predict_fn, lows, highs, center, len(features),
                n_chains=1024, n_steps=10_000, burn_in=500, seed=42, ess_target=10_000,
            )
        wall = time.perf_counter() - t0
        np.save(chain_path, samples)
        print(f"  [TH]  Chain guardado: {chain_path}  ({wall:.1f}s)")

        from cosmoml.ml.marginal import _sokal_tau
        tau = _sokal_tau(samples.reshape(len(samples) // 1024, 1024, len(features))
                         if samples.ndim == 2 else samples)
        ess = len(samples) / tau
        meta = {"cached": False, "wall_s": round(wall, 2),
                "n_steps_actual": None, "ess_final": round(ess, 1),
                "tau_max": round(tau, 2), "n_samples": len(samples)}

    fig = _render_getdist(
        samples, features,
        [labels.get(f, f).replace("$", "") for f in features],
        markers, title + " [Theory]", smooth_scale=0.5, ranges=ranges,
    )
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [TH]  Guardado: {figure_path}")
    return samples, meta


def run_theory_mcmc_gpu(gpu_predict_fn, features, ranges, ref, section,
                        labels, markers=None, title=""):
    """MCMC teórico GPU via JAX — paralelo a run_theory_mcmc (CPU).

    Usa el mismo _parallel_mcmc con 1024 cadenas pero el predict_fn
    es el surrogate JAX/GPU (chi2 teórico exacto vectorizado en GPU).
    Retorna (flat_samples, meta_dict).
    """
    chain_path  = CHAINS_DIR  / f"{section}_samples_theory_gpu.npy"
    figure_path = FIGURES_DIR / f"{section}_getdist_theory_gpu.png"

    lows   = np.array([ranges[f][0] for f in features])
    highs  = np.array([ranges[f][1] for f in features])
    center = np.array([ref[f] for f in features]) if ref else (lows + highs) / 2.0

    if chain_path.exists() and not FORCE_RETRAIN:
        print(f"  [TH-GPU]  Cache: {chain_path}")
        samples = np.load(chain_path)
        meta = {"cached": True, "wall_s": None, "n_steps_actual": None,
                "ess_final": None, "tau_max": None, "n_samples": len(samples)}
    else:
        print(f"  [TH-GPU]  MCMC teórico JAX/GPU...")
        t0 = time.perf_counter()
        samples = _parallel_mcmc(
            gpu_predict_fn, lows, highs, center, len(features),
            n_chains=1024, n_steps=10_000, burn_in=500, seed=42, ess_target=10_000,
        )
        wall = time.perf_counter() - t0
        np.save(chain_path, samples)
        print(f"  [TH-GPU]  Chain guardado: {chain_path}  ({wall:.1f}s)")

        from cosmoml.ml.marginal import _sokal_tau
        tau = _sokal_tau(samples.reshape(len(samples) // 1024, 1024, len(features))
                         if samples.ndim == 2 else samples)
        ess = len(samples) / tau
        meta = {"cached": False, "wall_s": round(wall, 2),
                "n_steps_actual": None, "ess_final": round(ess, 1),
                "tau_max": round(tau, 2), "n_samples": len(samples)}

    fig = _render_getdist(
        samples, features,
        [labels.get(f, f).replace("$", "") for f in features],
        markers, title + " [Theory GPU]", smooth_scale=0.5, ranges=ranges,
    )
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [TH-GPU]  Guardado: {figure_path}")
    return samples, meta


# ──────────────────────────────────────────────────────────────────────────────
# Chi2 functions — deben estar a nivel de módulo (picklables por ProcessPoolExecutor)
# ──────────────────────────────────────────────────────────────────────────────

# Cargamos los datos aquí arriba (nivel módulo) para que las closures
# capturen objetos ya cargados; ProcessPoolExecutor serializa el tuple entero.
# Se inicializan en main() y se asignan a variables globales para las funciones.
panth = des2024 = des2025 = bao = None


def chi2_lcdm_panth_bao(Om, H0):
    return (
        chi2_sne(panth, "FlatLambdaCDM", Om=Om, H0=H0,
                 M="marginalize", use_cepheid_calibrators=False)
        + chi2_bao(bao, Om=Om, H0=H0)
    )


def chi2_panth_bao(Om, H0, w0, wa):
    return chi2_joint(panth, bao, Om=Om, H0=H0, w0=w0, wa=wa,
                      sne_kwargs={"use_cepheid_calibrators": False})


def chi2_panth_bao_cmb(Om, H0, w0, wa):
    return chi2_panth_bao(Om, H0, w0, wa) + planck_prior_chi2(Om=Om, H0=H0)


def chi2_des2024_bao(Om, H0, w0, wa):
    return (
        chi2_sne_des(des2024, "Flatw0waCDM", Om=Om, H0=H0, w0=w0, wa=wa)
        + chi2_bao(bao, Om=Om, H0=H0, w0=w0, wa=wa)
    )


def chi2_des2024_bao_cmb(Om, H0, w0, wa):
    return chi2_des2024_bao(Om, H0, w0, wa) + planck_prior_chi2(Om=Om, H0=H0)


def chi2_des2025_bao(Om, H0, w0, wa):
    return (
        chi2_sne_des(des2025, "Flatw0waCDM", Om=Om, H0=H0, w0=w0, wa=wa)
        + chi2_bao(bao, Om=Om, H0=H0, w0=w0, wa=wa)
    )


def chi2_des2025_bao_cmb(Om, H0, w0, wa):
    return chi2_des2025_bao(Om, H0, w0, wa) + planck_prior_chi2(Om=Om, H0=H0)


def chi2_bao_only(Om, H0, w0, wa):
    return chi2_bao(bao, Om=Om, H0=H0, w0=w0, wa=wa)


# ──────────────────────────────────────────────────────────────────────────────
def main():
    global panth, des2024, des2025, bao

    print("=" * 60)
    print("CosmoML 06 — GPU server run")
    print(f"  PAPER_DIR     : {PAPER_DIR}")
    print(f"  COSMO_ROOT    : {COSMO_ROOT}")
    print(f"  FORCE_RETRAIN : {FORCE_RETRAIN}")
    print("=" * 60)

    # Cargar datos observacionales
    print("\nCargando datos...")
    panth   = load_pantheon_plus(apply_mask=True)
    des2024 = load_des_2024()
    des2025 = load_des_2025()
    bao     = load_desi_bao()
    print(f"  Pantheon+ : {len(panth)} SNe")
    print(f"  DES 2024  : {len(des2024)} SNe")
    print(f"  DES 2025  : {len(des2025)} SNe")
    print(f"  DESI BAO  : {len(bao)} medidas")

    FEATURES_2 = ["Om", "H0"]
    RANGES_2   = {k: GLOBAL_RANGES[k] for k in FEATURES_2}
    FEATURES_4 = ["Om", "H0", "w0", "wa"]
    RANGES_4   = GLOBAL_RANGES.copy()

    # ── Section 1: ΛCDM · Pantheon+ + BAO ────────────────────────────────────
    SECTION_1 = "6_1"
    TITLE_1   = "LCDM · Pantheon+ + DESI BAO"
    print(f"\n{'='*60}\n=== Section 1: {TITLE_1} ===")

    print(f"  Step 1: Dataset")
    df_1, REF_1 = build_dataset(chi2_lcdm_panth_bao, SECTION_1, FEATURES_2, RANGES_2)

    print(f"  Steps 2-3: Train + SHAP")
    model_1, info_1, shap_v_1, X_s_1 = train_and_shap(
        df_1, FEATURES_2, SECTION_1, title=TITLE_1,
    )

    print(f"  Steps 4-5: MCMC + GetDist")
    samples_1, _ = run_mcmc_and_getdist(
        model_1, FEATURES_2, RANGES_2, REF_1, SECTION_1,
        labels=LABELS, markers=None, title=TITLE_1,
    )

    # ── Section 2: w0waCDM — pares comparativos ───────────────────────────────
    print(f"\n{'='*60}\n=== Section 2: w0waCDM — Pares comparativos ===")

    # Pair 1a: panth + BAO
    SECTION_2_1A = "6_2_1a"
    TITLE_2_1A   = "w0waCDM · Pantheon+ + DESI BAO"
    print(f"\n  --- Pair 1a: {TITLE_2_1A}")
    df_2_1a, REF_2_1a = build_dataset(chi2_panth_bao, SECTION_2_1A, FEATURES_4, RANGES_4)
    model_2_1a, info_2_1a, shap_v_2_1a, X_s_2_1a = train_and_shap(
        df_2_1a, FEATURES_4, SECTION_2_1A, title=TITLE_2_1A,
    )
    samples_2_1a, _ = run_mcmc_and_getdist(
        model_2_1a, FEATURES_4, RANGES_4, REF_2_1a, SECTION_2_1A,
        labels=LABELS, markers=MARKERS_DE, title=TITLE_2_1A,
    )

    # Pair 1b: panth + BAO + CMB
    SECTION_2_1B = "6_2_1b"
    TITLE_2_1B   = "w0waCDM · Pantheon+ + DESI BAO + CMB"
    print(f"\n  --- Pair 1b: {TITLE_2_1B}")
    df_2_1b, REF_2_1b = build_dataset(chi2_panth_bao_cmb, SECTION_2_1B, FEATURES_4, RANGES_4)
    model_2_1b, info_2_1b, shap_v_2_1b, X_s_2_1b = train_and_shap(
        df_2_1b, FEATURES_4, SECTION_2_1B, title=TITLE_2_1B,
    )
    samples_2_1b, _ = run_mcmc_and_getdist(
        model_2_1b, FEATURES_4, RANGES_4, REF_2_1b, SECTION_2_1B,
        labels=LABELS, markers=MARKERS_DE, title=TITLE_2_1B,
    )

    # Pair 1: comparativo ML
    plot_getdist_comparison(
        [samples_2_1a, samples_2_1b],
        ["Pantheon+ + BAO", "Pantheon+ + BAO + CMB"],
        FEATURES_4, LABELS,
        markers=MARKERS_DE,
        title="w0waCDM — Pair 1: Pantheon+ + BAO  vs  + CMB",
        save_path=FIGURES_DIR / "6_2_pair1_comparison.png",
        filled=[True, True],
        ranges=RANGES_4,
    )

    # Pair 2a: DES-2024 + BAO
    SECTION_2_2A = "6_2_2a"
    TITLE_2_2A   = "w0waCDM · DES-SN5YR 2024 + DESI BAO"
    print(f"\n  --- Pair 2a: {TITLE_2_2A}")
    df_2_2a, REF_2_2a = build_dataset(chi2_des2024_bao, SECTION_2_2A, FEATURES_4, RANGES_4)
    model_2_2a, info_2_2a, shap_v_2_2a, X_s_2_2a = train_and_shap(
        df_2_2a, FEATURES_4, SECTION_2_2A, title=TITLE_2_2A,
    )
    samples_2_2a, _ = run_mcmc_and_getdist(
        model_2_2a, FEATURES_4, RANGES_4, REF_2_2a, SECTION_2_2A,
        labels=LABELS, markers=MARKERS_DE, title=TITLE_2_2A,
    )

    # Pair 2b: DES-2024 + BAO + CMB
    SECTION_2_2B = "6_2_2b"
    TITLE_2_2B   = "w0waCDM · DES-SN5YR 2024 + DESI BAO + CMB"
    print(f"\n  --- Pair 2b: {TITLE_2_2B}")
    df_2_2b, REF_2_2b = build_dataset(chi2_des2024_bao_cmb, SECTION_2_2B, FEATURES_4, RANGES_4)
    model_2_2b, info_2_2b, shap_v_2_2b, X_s_2_2b = train_and_shap(
        df_2_2b, FEATURES_4, SECTION_2_2B, title=TITLE_2_2B,
    )
    samples_2_2b, _ = run_mcmc_and_getdist(
        model_2_2b, FEATURES_4, RANGES_4, REF_2_2b, SECTION_2_2B,
        labels=LABELS, markers=MARKERS_DE, title=TITLE_2_2B,
    )

    # Pair 2: comparativo ML
    plot_getdist_comparison(
        [samples_2_2a, samples_2_2b],
        ["DES-2024 + BAO", "DES-2024 + BAO + CMB"],
        FEATURES_4, LABELS,
        markers=MARKERS_DE,
        title="w0waCDM — Pair 2: DES-SN5YR 2024 + BAO  vs  + CMB",
        save_path=FIGURES_DIR / "6_2_pair2_comparison.png",
        filled=[True, True],
        ranges=RANGES_4,
    )

    # Pair 3a: DES-2025 + BAO
    SECTION_2_3A = "6_2_3a"
    TITLE_2_3A   = "w0waCDM · DES-SN5YR 2025 + DESI BAO"
    print(f"\n  --- Pair 3a: {TITLE_2_3A}")
    df_2_3a, REF_2_3a = build_dataset(chi2_des2025_bao, SECTION_2_3A, FEATURES_4, RANGES_4)
    model_2_3a, info_2_3a, shap_v_2_3a, X_s_2_3a = train_and_shap(
        df_2_3a, FEATURES_4, SECTION_2_3A, title=TITLE_2_3A,
    )
    samples_2_3a, _ = run_mcmc_and_getdist(
        model_2_3a, FEATURES_4, RANGES_4, REF_2_3a, SECTION_2_3A,
        labels=LABELS, markers=MARKERS_DE, title=TITLE_2_3A,
    )

    # Pair 3b: DES-2025 + BAO + CMB
    SECTION_2_3B = "6_2_3b"
    TITLE_2_3B   = "w0waCDM · DES-SN5YR 2025 + DESI BAO + CMB"
    print(f"\n  --- Pair 3b: {TITLE_2_3B}")
    df_2_3b, REF_2_3b = build_dataset(chi2_des2025_bao_cmb, SECTION_2_3B, FEATURES_4, RANGES_4)
    model_2_3b, info_2_3b, shap_v_2_3b, X_s_2_3b = train_and_shap(
        df_2_3b, FEATURES_4, SECTION_2_3B, title=TITLE_2_3B,
    )
    samples_2_3b, _ = run_mcmc_and_getdist(
        model_2_3b, FEATURES_4, RANGES_4, REF_2_3b, SECTION_2_3B,
        labels=LABELS, markers=MARKERS_DE, title=TITLE_2_3B,
    )

    # Pair 3: comparativo ML
    plot_getdist_comparison(
        [samples_2_3a, samples_2_3b],
        ["DES-2025 + BAO", "DES-2025 + BAO + CMB"],
        FEATURES_4, LABELS,
        markers=MARKERS_DE,
        title="w0waCDM — Pair 3: DES-SN5YR 2025 + BAO  vs  + CMB",
        filled=[True, True],
        save_path=FIGURES_DIR / "6_2_pair3_comparison.png",
        ranges=RANGES_4,
    )

    # ── Section 3: BAO-only + resumen final + timing benchmark ───────────────
    SECTION_3_BAO = "6_3_bao"
    TITLE_3_BAO   = "w0waCDM · DESI BAO only"
    print(f"\n{'='*60}\n=== Section 3: {TITLE_3_BAO} ===")

    df_3_bao, REF_3_bao = build_dataset(chi2_bao_only, SECTION_3_BAO, FEATURES_4, RANGES_4)
    model_3_bao, info_3_bao, shap_v_3_bao, X_s_3_bao = train_and_shap(
        df_3_bao, FEATURES_4, SECTION_3_BAO, title=TITLE_3_BAO,
    )
    samples_3_bao, ml_meta_bao = run_mcmc_and_getdist(
        model_3_bao, FEATURES_4, RANGES_4, REF_3_bao, SECTION_3_BAO,
        labels=LABELS, markers=MARKERS_DE, title=TITLE_3_BAO,
    )

    # Resumen 3-way ML
    print("\n  --- Section 3: Summary 3-way GetDist (ML)")
    _chain_panthbao = CHAINS_DIR / "6_2_1a_samples.npy"
    _chain_full     = CHAINS_DIR / "6_2_1b_samples.npy"
    if not _chain_panthbao.exists() or not _chain_full.exists():
        raise FileNotFoundError(
            "Chains de Section 2 Pair 1 no encontrados. Verifica que Section 2 corrió bien."
        )
    plot_getdist_comparison(
        [samples_3_bao, np.load(_chain_panthbao), np.load(_chain_full)],
        ["BAO", "BAO + Pantheon+", "BAO + Pantheon+ + CMB"],
        FEATURES_4, LABELS,
        markers=MARKERS_DE,
        title="w0waCDM — Summary: BAO / BAO+Pantheon+ / BAO+Pantheon++CMB",
        save_path=FIGURES_DIR / "6_3_summary_getdist.png",
        filled=[True, True, True],
        ranges=RANGES_4,
    )

    # ── Section 3: Theory MCMC benchmark ─────────────────────────────────────
    print(f"\n{'='*60}\n=== Section 3: Theory MCMC benchmark ===")

    # BAO-only theory
    print("\n  --- BAO-only (theory)")
    _, th_meta_bao = run_theory_mcmc(
        chi2_bao_only, FEATURES_4, RANGES_4, REF_3_bao, SECTION_3_BAO,
        labels=LABELS, markers=MARKERS_DE, title=TITLE_3_BAO,
    )

    # Pantheon+ + BAO theory — cargado desde caché generada en Section 2
    _chain_pb_ml  = CHAINS_DIR / "6_2_1a_samples.npy"
    _chain_pbc_ml = CHAINS_DIR / "6_2_1b_samples.npy"
    _, th_meta_pb = run_theory_mcmc(
        chi2_panth_bao, FEATURES_4, RANGES_4, REF_2_1a, SECTION_2_1A,
        labels=LABELS, markers=MARKERS_DE, title=TITLE_2_1A,
    )
    _, th_meta_pbc = run_theory_mcmc(
        chi2_panth_bao_cmb, FEATURES_4, RANGES_4, REF_2_1b, SECTION_2_1B,
        labels=LABELS, markers=MARKERS_DE, title=TITLE_2_1B,
    )
    ml_meta_pb  = {"cached": True, "wall_s": None, "n_steps_actual": None,
                   "ess_final": None, "tau_max": None,
                   "n_samples": len(np.load(_chain_pb_ml))}
    ml_meta_pbc = {"cached": True, "wall_s": None, "n_steps_actual": None,
                   "ess_final": None, "tau_max": None,
                   "n_samples": len(np.load(_chain_pbc_ml))}

    # ── 3-way theory ──────────────────────────────────────────────────────────
    _chain_th_bao = CHAINS_DIR / f"{SECTION_3_BAO}_samples_theory.npy"
    _chain_th_pb  = CHAINS_DIR / f"{SECTION_2_1A}_samples_theory.npy"
    _chain_th_pbc = CHAINS_DIR / f"{SECTION_2_1B}_samples_theory.npy"

    if all(p.exists() for p in [_chain_th_bao, _chain_th_pb, _chain_th_pbc]):
        plot_getdist_comparison(
            [np.load(_chain_th_bao), np.load(_chain_th_pb), np.load(_chain_th_pbc)],
            ["BAO", "BAO + Pantheon+", "BAO + Pantheon+ + CMB"],
            FEATURES_4, LABELS,
            markers=MARKERS_DE,
            title="w0waCDM — Summary Theory: BAO / BAO+Pantheon+ / BAO+Pantheon++CMB",
            save_path=FIGURES_DIR / "6_3_summary_getdist_theory.png",
            filled=[True, True, True],
            ranges=RANGES_4,
        )
    else:
        print("  [!] Chains theory incompletos — 3-way theory omitido.")

    # ── Section 3: Theory MCMC GPU benchmark (JAX) ────────────────────────────
    th_gpu_meta_bao = th_gpu_meta_pb = th_gpu_meta_pbc = None
    if _JAX_THEORY_OK:
        print(f"\n{'='*60}\n=== Section 3: Theory MCMC JAX/GPU benchmark ===")

        print("\n  Compilando funciones JAX (warm-up)...")
        gpu_fn_bao = _make_chi2_gpu_fn(panth=None,  bao=bao)
        gpu_fn_pb  = _make_chi2_gpu_fn(panth=panth, bao=bao)
        gpu_fn_pbc = _make_chi2_gpu_fn(panth=panth, bao=bao, planck_prior=True)
        print("  JAX JIT listo.")

        print("\n  --- BAO-only (theory GPU)")
        _, th_gpu_meta_bao = run_theory_mcmc_gpu(
            gpu_fn_bao, FEATURES_4, RANGES_4, REF_3_bao, SECTION_3_BAO,
            labels=LABELS, markers=MARKERS_DE, title=TITLE_3_BAO,
        )

        print("\n  --- Pantheon+ + BAO (theory GPU)")
        _, th_gpu_meta_pb = run_theory_mcmc_gpu(
            gpu_fn_pb, FEATURES_4, RANGES_4, REF_2_1a, SECTION_2_1A,
            labels=LABELS, markers=MARKERS_DE, title=TITLE_2_1A,
        )

        print("\n  --- Pantheon+ + BAO + CMB (theory GPU)")
        _, th_gpu_meta_pbc = run_theory_mcmc_gpu(
            gpu_fn_pbc, FEATURES_4, RANGES_4, REF_2_1b, SECTION_2_1B,
            labels=LABELS, markers=MARKERS_DE, title=TITLE_2_1B,
        )

        # 3-way comparison theory GPU
        _chain_tg_bao = CHAINS_DIR / f"{SECTION_3_BAO}_samples_theory_gpu.npy"
        _chain_tg_pb  = CHAINS_DIR / f"{SECTION_2_1A}_samples_theory_gpu.npy"
        _chain_tg_pbc = CHAINS_DIR / f"{SECTION_2_1B}_samples_theory_gpu.npy"
        if all(p.exists() for p in [_chain_tg_bao, _chain_tg_pb, _chain_tg_pbc]):
            plot_getdist_comparison(
                [np.load(_chain_tg_bao), np.load(_chain_tg_pb), np.load(_chain_tg_pbc)],
                ["BAO", "BAO + Pantheon+", "BAO + Pantheon+ + CMB"],
                FEATURES_4, LABELS,
                markers=MARKERS_DE,
                title="w0waCDM — Summary Theory GPU: BAO / BAO+Pantheon+ / BAO+Pantheon++CMB",
                save_path=FIGURES_DIR / "6_3_summary_getdist_theory_gpu.png",
                filled=[True, True, True],
                ranges=RANGES_4,
            )
    else:
        print("\n  [!] JAX no disponible — benchmark GPU-theory omitido.")
        print("      Instala con: pip install 'jax[cuda]'")

    # ── Timings JSON ──────────────────────────────────────────────────────────
    def _speedup(ml_m, th_m):
        w_ml, w_th = ml_m.get("wall_s"), th_m.get("wall_s")
        return round(w_th / w_ml, 2) if (w_ml and w_th) else None

    def _meta_or_skip(m):
        return m if m is not None else {"wall_s": None, "ess_final": None,
                                        "n_samples": None, "cached": True}

    timings = {
        "ml_engine":         "RWMH paralelo 1024 chains, GPU booster (LogChi2)",
        "theory_cpu_engine": "RWMH paralelo 1024 chains, CPU ProcessPoolExecutor",
        "theory_gpu_engine": "RWMH paralelo 1024 chains, JAX vmap GPU (chi2 exacta)",
        "common": {"n_chains": 1024, "n_steps_cap": 10_000,
                   "burn_in": 500, "seed": 42, "ess_target": 10_000},
        "runs": {
            "bao_only": {
                "ml":         ml_meta_bao,
                "theory_cpu": th_meta_bao,
                "theory_gpu": _meta_or_skip(th_gpu_meta_bao),
                "speedup_cpu_vs_ml":  _speedup(ml_meta_bao, th_meta_bao),
                "speedup_gpu_vs_ml":  _speedup(ml_meta_bao, _meta_or_skip(th_gpu_meta_bao)),
                "speedup_gpu_vs_cpu": _speedup(_meta_or_skip(th_gpu_meta_bao), th_meta_bao),
            },
            "panth_bao": {
                "ml":         ml_meta_pb,
                "theory_cpu": th_meta_pb,
                "theory_gpu": _meta_or_skip(th_gpu_meta_pb),
                "speedup_cpu_vs_ml":  _speedup(ml_meta_pb, th_meta_pb),
                "speedup_gpu_vs_ml":  _speedup(ml_meta_pb, _meta_or_skip(th_gpu_meta_pb)),
                "speedup_gpu_vs_cpu": _speedup(_meta_or_skip(th_gpu_meta_pb), th_meta_pb),
            },
            "panth_bao_cmb": {
                "ml":         ml_meta_pbc,
                "theory_cpu": th_meta_pbc,
                "theory_gpu": _meta_or_skip(th_gpu_meta_pbc),
                "speedup_cpu_vs_ml":  _speedup(ml_meta_pbc, th_meta_pbc),
                "speedup_gpu_vs_ml":  _speedup(ml_meta_pbc, _meta_or_skip(th_gpu_meta_pbc)),
                "speedup_gpu_vs_cpu": _speedup(_meta_or_skip(th_gpu_meta_pbc), th_meta_pbc),
            },
        },
    }
    timings_path = PAPER_DIR / "timings.json"
    with open(timings_path, "w") as f:
        json.dump(timings, f, indent=2)
    print(f"\n  Timings guardados: {timings_path}")

    # Tabla stdout
    def _fs(v):
        return f"{v:.1f}s" if v is not None else "cached"
    def _fsp(v):
        return f"{v:.1f}×" if v is not None else "–"
    rows = [
        ("bao_only",      ml_meta_bao, th_meta_bao, th_gpu_meta_bao),
        ("panth_bao",     ml_meta_pb,  th_meta_pb,  th_gpu_meta_pb),
        ("panth_bao_cmb", ml_meta_pbc, th_meta_pbc, th_gpu_meta_pbc),
    ]
    print("\n" + "─" * 90)
    print(f"{'section':<18}{'ml [s]':>10}{'th-cpu [s]':>12}{'th-gpu [s]':>12}"
          f"{'×cpu/ml':>9}{'×gpu/ml':>9}{'×gpu/cpu':>10}")
    print("─" * 90)
    for name, ml_m, th_m, tg_m in rows:
        tg = _meta_or_skip(tg_m)
        print(
            f"{name:<18}"
            f"{_fs(ml_m['wall_s']):>10}"
            f"{_fs(th_m['wall_s']):>12}"
            f"{_fs(tg['wall_s']):>12}"
            f"{_fsp(_speedup(ml_m, th_m)):>9}"
            f"{_fsp(_speedup(ml_m, tg)):>9}"
            f"{_fsp(_speedup(tg, th_m)):>10}"
        )
    print("─" * 90)

    print("\n=== DONE ===")
    print(f"  Figuras  : {FIGURES_DIR}/")
    print(f"  Datasets : {DATASETS_DIR}/")
    print(f"  Modelos  : {MODELS_DIR}/")
    print(f"  Chains   : {CHAINS_DIR}/")
    print(f"  Timings  : {timings_path}")


if __name__ == "__main__":
    main()
