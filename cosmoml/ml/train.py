"""XGBoost regression on chi2 datasets.

Key design choices:
  - ``log_target=True`` applies a SHIFTED-log10 transform,
        y = log10(chi2 - chi2_min + 1)
    that maps the chi2 minimum to 0 in training space, amplifying small chi2
    differences near the best-fit by ~3 orders of magnitude. This is the
    transform that lets the surrogate resolve 1-sigma / 2-sigma contours.
  - ``LogChi2Model`` wraps the raw XGBoost regressor with the exact inverse
    transform so downstream code (contours, SHAP) sees linear chi2 again.
  - GPU is auto-detected at import time by attempting a tiny ``device='cuda'``
    fit; falls back to CPU silently.
"""
from __future__ import annotations
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.callback import EarlyStopping
from sklearn.model_selection import train_test_split

from ..config import (
    DEFAULT_XGB_PARAMS,
    DEFAULT_EARLY_STOPPING_ROUNDS,
    DEFAULT_EARLY_STOPPING_MIN_DELTA,
)

try:
    xgb.XGBRegressor(tree_method="hist", device="cuda").fit(np.zeros((1, 1)), np.zeros(1))
    _HAS_GPU = True
    print("[train.py] NVIDIA GPU detected. CUDA enabled by default.")
except Exception:
    _HAS_GPU = False
    print("[train.py] No GPU detected. Using CPU.")


class LogChi2Model:
    """Wrapper that inverts the shifted-log10 transform on prediction.

    ``predict(X)`` returns LINEAR chi2 again, so plot_contour_2d and SHAP
    work without further conversion. ``raw_model`` exposes the underlying
    XGBRegressor (SHAP must use this because SHAP values live in the
    training space — shifted-log10 here).
    """
    def __init__(self, inner_model, y_min: float = 0.0):
        self.raw_model = inner_model
        self.y_min = y_min

    def predict(self, X):
        # Inverse of y = log10(chi2 - y_min + 1)
        return (10 ** self.raw_model.predict(X)) - 1.0 + self.y_min

    def __getattr__(self, name):
        return getattr(self.raw_model, name)


def train_xgb(
    df: pd.DataFrame,
    features: list[str],
    target: str = "chi2",
    *,
    log_target: bool = False,
    chi2_clip: float | None = None,
    drop_chi2_bad: float | None = 90000.0,
    test_size: float = 0.15,
    random_state: int = 42,
    hp_overrides: dict | None = None,
    early_stopping_rounds: int | None = DEFAULT_EARLY_STOPPING_ROUNDS,
    early_stopping_min_delta: float | None = None,
    cache_path: str | Path | None = None,
    force_retrain: bool = False,
    verbose: bool = True,
) -> tuple[xgb.XGBRegressor | LogChi2Model, dict]:
    """Train an XGBRegressor on (features -> chi2).

    Parameters
    ----------
    log_target : bool
        If True, train on shifted-log10 of chi2 (see module docstring) and
        return a ``LogChi2Model``. This is the default strategy for
        cosmology likelihoods — it compresses the chi2 dynamic range and
        amplifies the gradient near the minimum.
    chi2_clip : float | None
        Legacy alternative to log_target: cap the target at ``min + clip``
        without dropping rows. Breaks SHAP and is incompatible with log_target.
    drop_chi2_bad : float | None
        Drop rows with ``chi2 >= drop_chi2_bad`` (default 90000). These are
        ``_CHI2_BAD`` sentinels emitted on numerical failures.
    test_size : float
        Validation fraction (default 0.15).
    hp_overrides : dict
        Override ``DEFAULT_XGB_PARAMS``.
    early_stopping_rounds : int | None
        Stop training if validation has not improved by at least
        ``early_stopping_min_delta`` for N rounds. None disables.
    early_stopping_min_delta : float | None
        If None, defaults to 1e-4 for log_target=True and to
        ``DEFAULT_EARLY_STOPPING_MIN_DELTA`` for linear training.
    cache_path : str | Path | None
        Save model (.ubj) and info (.info.pkl) here. On the next run, if both
        files exist and ``force_retrain`` is False, the cache is loaded
        instead of retraining.
    force_retrain : bool
        If True, ignore the cache and retrain (overwriting cache_path).

    Returns
    -------
    model, info
        ``info`` includes ``time, r2, X_val, y_val, X_train, y_train, n_train,
        n_val, eval_results, best_iteration, best_score, eval_metric,
        log_target, y_min``. ``y_min`` is the shift applied when log_target
        is True (0 otherwise).

    Notes
    -----
    In log_target mode, ``r2`` and the curves in ``eval_results`` are in
    the shifted-log10 space, not in linear chi2.
    """
    if log_target and chi2_clip is not None:
        raise ValueError("log_target=True and chi2_clip are incompatible.")

    # Cache: short-circuit if both files exist.
    if cache_path is not None:
        cache_path = Path(cache_path)
        info_path = cache_path.with_suffix(".info.pkl")
        if not force_retrain and cache_path.exists() and info_path.exists():
            if verbose:
                print(f"  loading cached model: {cache_path.name}")
            raw = xgb.XGBRegressor()
            raw.load_model(str(cache_path))
            with open(info_path, "rb") as f:
                info = pickle.load(f)
            model = LogChi2Model(raw, y_min=info.get("y_min", 0.0)) if info.get("log_target") else raw
            if verbose:
                print(f"  R2={info['r2']:.5f} | best_iter={info['best_iteration']}"
                      f" | n_train={info['n_train']:,} | n_val={info['n_val']:,}")
            return model, info

    df_use = df.drop_duplicates() if df.duplicated().any() else df.copy()

    if drop_chi2_bad is not None:
        n_before = len(df_use)
        df_use = df_use[df_use[target] < drop_chi2_bad]
        n_dropped = n_before - len(df_use)
        if n_dropped > 0 and verbose:
            print(f"  dropping {n_dropped} rows with chi2 >= {drop_chi2_bad} (sentinels)")

    X = df_use[features]
    y = df_use[target].copy()

    y_min_val = 0.0
    if log_target:
        if (y <= 0).any():
            raise ValueError("log_target=True requires chi2 > 0 in every row")
        y_min_val = float(y.min())
        # Shifted-log10: maps chi2_min to 0, amplifies near-minimum gradient.
        y = np.log10(y - y_min_val + 1.0)
        if verbose:
            print(f"  target in shifted-log10: range [{y.min():.3f}, {y.max():.3f}]")

    if chi2_clip is not None:
        cap = float(y.min() + chi2_clip)
        n_capped = int((y > cap).sum())
        y = y.clip(upper=cap)
        if verbose:
            print(f"  target clipped at min+{chi2_clip}={cap:.1f} ({n_capped} rows affected)")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    auto_device = {"device": "cuda" if _HAS_GPU else "cpu"}
    params = {**DEFAULT_XGB_PARAMS, **auto_device, **(hp_overrides or {})}

    if early_stopping_min_delta is None:
        early_stopping_min_delta = 1e-4 if log_target else DEFAULT_EARLY_STOPPING_MIN_DELTA

    callbacks = []
    if early_stopping_rounds is not None:
        callbacks.append(
            EarlyStopping(
                rounds=early_stopping_rounds,
                min_delta=early_stopping_min_delta,
                save_best=True,
                maximize=False,
                metric_name=params.get("eval_metric", "rmse"),
                data_name="validation_1",
            )
        )

    model = xgb.XGBRegressor(**params, callbacks=callbacks or None)

    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    elapsed = time.time() - t0
    r2 = float(model.score(X_val, y_val))

    eval_results = model.evals_result()
    best_iter = int(getattr(model, "best_iteration", model.n_estimators - 1))
    best_score = float(getattr(model, "best_score", float("nan")))

    if verbose:
        n_used = best_iter + 1
        n_total = params.get("n_estimators", 0)
        msg = f"  training: {elapsed:.2f}s | R2={r2:.5f} | best_iter={best_iter}/{n_total}"
        if early_stopping_rounds and best_iter < n_total - 1:
            saved = (n_total - n_used) / n_total
            msg += f"  (early stop, ~{saved:.0%} saved)"
        print(msg)

    final_model = LogChi2Model(model, y_min=y_min_val) if log_target else model

    info = dict(
        time=elapsed, r2=r2,
        X_val=X_val, y_val=y_val,
        X_train=X_train, y_train=y_train,
        n_train=len(X_train), n_val=len(X_val),
        eval_results=eval_results,
        best_iteration=best_iter,
        best_score=best_score,
        eval_metric=params.get("eval_metric", "rmse"),
        log_target=log_target,
        y_min=y_min_val if log_target else 0.0,
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        raw_to_save = final_model.raw_model if log_target else final_model
        raw_to_save.save_model(str(cache_path))
        with open(cache_path.with_suffix(".info.pkl"), "wb") as f:
            pickle.dump(info, f)
        if verbose:
            print(f"  model cached: {cache_path.name}")

    return final_model, info
