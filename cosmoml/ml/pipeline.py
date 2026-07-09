"""High-level pipeline helpers: best-fit location, dataset building, train+SHAP."""
from __future__ import annotations

import numpy as np
from pathlib import Path

from ..sampling import build_chi2_dataset, load_or_build


def locate_bestfit(chi2_fn, features: list[str], ranges: dict):
    """Locate chi2 minimum with Minuit and compute Hessian covariance.

    Returns
    -------
    REF : dict  param -> best-fit value
    chi2_min : float
    errors : dict  param -> 1-sigma error
    cov_matrix : ndarray (ndim, ndim) or None
    """
    from iminuit import Minuit

    init = {f: (ranges[f][0] + ranges[f][1]) / 2.0 for f in features}
    if "w0" in features: init["w0"] = -1.0
    if "wa" in features: init["wa"] =  0.0
    if "Om" in features: init["Om"] =  0.3
    if "H0" in features: init["H0"] = 68.0
    m = Minuit(chi2_fn, **init)
    m.limits = [ranges[f] for f in features]
    m.migrad()

    cov_matrix = None
    try:
        m.hesse()
        errors = {f: float(m.errors[f]) for f in features}
        if m.covariance is not None:
            cov_matrix = np.array(m.covariance)
    except Exception:
        errors = {f: 0.1 * (ranges[f][1] - ranges[f][0]) for f in features}

    REF = {f: float(m.values[f]) for f in features}
    print(f"  Best-fit : {', '.join(f'{f}={v:.4f}' for f, v in REF.items())}")
    print(f"  Errors   : {', '.join(f'{f}={errors[f]:.4f}' for f in features)}")
    print(f"  chi2_min : {m.fval:.2f}")
    if cov_matrix is not None:
        print("  Covariance: OK (correlated sampling enabled)")
    return REF, m.fval, errors, cov_matrix


def build_pipeline_dataset(
    chi2_fn,
    section: str,
    features: list[str],
    ranges: dict,
    datasets_dir: Path,
    n_gaussian: int = 150_000,
    gaussian_sigma_scale: float = 3.0,
    n_random: int = 80_000,
    force_retrain: bool = False,
):
    """Locate best-fit, build slices + random + Gaussian dataset, cache to CSV.

    Returns
    -------
    df : DataFrame
    REF : dict  param -> best-fit value
    cov_matrix : ndarray (ndim, ndim) or None
    """
    csv_path = Path(datasets_dir) / f"{section}_dataset.csv"
    REF, _, errors, cov_matrix = locate_bestfit(chi2_fn, features, ranges)
    ndim = len(features)

    def builder():
        slices = []
        for _i in range(ndim):
            for _j in range(_i + 1, ndim):
                fi, fj = features[_i], features[_j]
                fixed = {f: REF[f] for f in features if f not in (fi, fj)}
                slices.append({fi: ranges[fi], fj: ranges[fj], **fixed, "_n": 20_000})
        gclouds = None
        if n_gaussian > 0:
            if cov_matrix is not None:
                gclouds = [{"center": REF, "cov": cov_matrix.tolist(),
                            "scale": gaussian_sigma_scale, "n": n_gaussian,
                            "bounds": {f: ranges[f] for f in features}}]
            else:
                sigma = {f: gaussian_sigma_scale * errors[f] for f in features}
                gclouds = [{"center": REF, "sigma": sigma, "n": n_gaussian,
                            "bounds": {f: ranges[f] for f in features}}]
        return build_chi2_dataset(
            chi2_fn=chi2_fn, param_names=features,
            slices=slices,
            random_box={f: ranges[f] for f in features},
            n_random=n_random,
            gaussian_clouds=gclouds,
            save_to=csv_path, seed=42,
        )

    df = load_or_build(csv_path, builder, force=force_retrain)
    print(f"  Dataset  : {len(df):,} rows | chi2 [{df['chi2'].min():.2f}, {df['chi2'].max():.2f}]")
    return df, REF, cov_matrix


def train_and_shap(
    df,
    features: list[str],
    section: str,
    models_dir: Path,
    figures_dir: Path,
    title: str = "",
    force_retrain: bool = False,
):
    """Train XGBoost with log-chi2 target and generate full SHAP suite.

    Returns
    -------
    model, info, shap_values, X_shap
    """
    from .train import train_xgb
    from .curve import plot_learning_curve
    from .shap_utils import shap_summary, shap_waterfall, shap_dependence_all

    model, info = train_xgb(
        df, features=features, log_target=True,
        hp_overrides=dict(n_estimators=5000, learning_rate=0.03,
                          max_depth=10, device="cuda"),
        cache_path=Path(models_dir) / f"{section}_model.ubj",
        force_retrain=force_retrain,
    )
    plot_learning_curve(info,
        title=f"{title} — Learning Curve (R²={info['r2']:.5f})", show=True)
    shap_v, X_s = shap_summary(model, info["X_val"],
        title=title, save_dir=figures_dir, prefix=section, show=True)
    shap_waterfall(shap_v, idx=0, title=f"{title} — SHAP waterfall", show=True)
    shap_dependence_all(shap_v, X_s, save_dir=figures_dir, prefix=section, show=True)
    return model, info, shap_v, X_s
