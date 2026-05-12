"""Entrenamiento XGBoost para regresión χ²."""
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
    # Intenta entrenar un mini-modelo basura en la gráfica
    xgb.XGBRegressor(tree_method="hist", device="cuda").fit(np.zeros((1,1)), np.zeros(1))
    _HAS_GPU = True
    print("[train.py] GPU NVIDIA detectada. Aceleración CUDA activada por defecto.")
except Exception:
    _HAS_GPU = False
    print("[train.py] No se detectó GPU. Se usará la CPU para XGBoost.")

class LogChi2Model:
    def __init__(self, inner_model, y_min=0.0):
        self.raw_model = inner_model
        self.y_min = y_min

    def predict(self, X):
        # Deshace el shifted-log: 10^pred - 1 + y_min
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
    """Entrena XGBRegressor sobre (features → chi2).

    Parameters
    ----------
    log_target : bool
        Si True, entrena sobre log10(chi²) en lugar de chi² lineal. Devuelve
        un `LogChi2Model` cuyo `.predict()` reconvierte automáticamente a chi²
        lineal. Es la estrategia ELEGIDA para escenarios con rango dinámico
        extremo (BAO): comprime χ²∈[8,10⁵] → log∈[0.9,5], el MSE deja de estar
        dominado por puntos lejos del best-fit y NO hace falta filtrar ni
        clipear. SHAP funciona directamente sin filtrado posterior (los
        valores SHAP están en log10(χ²) — ver docstring de LogChi2Model).
    chi2_clip : float | None
        [LEGACY, prefiere log_target=True] Si se pasa, capa el TARGET a
        `min(chi², chi².min() + chi2_clip)` SIN descartar filas. Mantiene rango
        finito para el MSE pero deja al modelo memorizando una meseta plana en
        ~98% de las filas y rompe SHAP. Sólo útil si por alguna razón no
        quieres usar log_target.
    drop_chi2_bad : float | None
        Descarta filas con chi² >= drop_chi2_bad (default 90000). Estos son
        FAIL de la integración numérica (`_CHI2_BAD = 99999`), no datos reales.
        Pásalo a None para no filtrar nada.
    hp_overrides : dict
        Sobrescribe los DEFAULT_XGB_PARAMS.
    early_stopping_rounds : int | None
        Para el entrenamiento si la val no mejora en N rondas. None = desactivar.
    early_stopping_min_delta : float | None
        Mínima mejora de RMSE para que la ronda cuente como "mejora". Si None,
        se elige automáticamente según el espacio del target:
          - chi² lineal: 0.01 (ruido <0.01 es invisible para los contornos)
          - log10(χ²):   1e-4 (≈0.02% de rango log; con 0.01 paras a ~190 iter)
        Pasa 0.0 para reproducir el comportamiento clásico sin tolerancia.
    cache_path : str | Path | None
        Si se pasa, guarda el modelo entrenado en `cache_path` (formato XGBoost
        nativo `.ubj`) y el `info` en `cache_path.with_suffix('.info.pkl')`.
        En la SIGUIENTE ejecución, si los dos archivos existen y `force_retrain`
        es False, se cargan en lugar de re-entrenar (se devuelven (model, info)
        idénticos a la primera ejecución).

        Convención: `cache_path = OUTPUTS_DIR / "models" / f"{SCENARIO}.ubj"`.

        IMPORTANTE: la caché NO se invalida sola. Si cambias hiperparámetros,
        la receta de generación del dataset, o `log_target`, pasa
        `force_retrain=True` para reentrenar.
    force_retrain : bool
        Si True, ignora la caché y reentrena (reescribiendo el archivo si
        `cache_path` está definido).

    Returns
    -------
    model, info  (info contiene: time, r2, X_val, y_val, n_train, n_val,
                  eval_results, best_iteration, best_score, log_target)

    Notas sobre métricas en modo log_target:
        - r2, eval_results, best_score se calculan en el ESPACIO DE
          ENTRENAMIENTO (log10). Es lo coherente con la curva de aprendizaje.
        - y_val, y_train en `info` están en log10. Para tener χ² lineal usar
          `10**info["y_val"]`.
    """
    if log_target and chi2_clip is not None:
        raise ValueError("log_target=True y chi2_clip son incompatibles. "
                         "log_target ya soluciona el rango dinámico — quita el clip.")

    # --- Caché: cargar modelo previo si existe ---
    if cache_path is not None:
        cache_path = Path(cache_path)
        info_path = cache_path.with_suffix(".info.pkl")
        if not force_retrain and cache_path.exists() and info_path.exists():
            if verbose:
                print(f"  cargando modelo cacheado: {cache_path.name}")
            raw = xgb.XGBRegressor()
            raw.load_model(str(cache_path))
            with open(info_path, "rb") as f:
                info = pickle.load(f)
            model = LogChi2Model(raw) if info.get("log_target") else raw
            if verbose:
                print(f"  R²={info['r2']:.5f} | best_iter={info['best_iteration']}"
                      f" | n_train={info['n_train']:,} | n_val={info['n_val']:,}")
            return model, info

    df_use = df.drop_duplicates() if df.duplicated().any() else df.copy()

    # Filtrar fallos numéricos (no son datos reales, son sentinels)
    if drop_chi2_bad is not None:
        n_before = len(df_use)
        df_use = df_use[df_use[target] < drop_chi2_bad]
        n_dropped = n_before - len(df_use)
        if n_dropped > 0 and verbose:
            print(f"  descartando {n_dropped} filas con chi² >= {drop_chi2_bad} (fallos integración)")

    X = df_use[features]
    y = df_use[target].copy()

    # --- INICIO DEL NUEVO BLOQUE SHIFTED-LOG ---
    y_min_val = 0.0
    if log_target:
        if (y <= 0).any():
            raise ValueError("log_target=True requiere chi² > 0 en todas las filas")
        y_min_val = float(y.min())
        # Aplicamos el Shifted-Log
        y = np.log10(y - y_min_val + 1.0)
        if verbose:
            print(f"  target en Shifted-Log10: rango [{y.min():.3f}, {y.max():.3f}]")
    # --- FIN DEL NUEVO BLOQUE ---

    # Clip del target para escenarios con rango dinámico extremo (BAO).
    # NO descarta filas: las filas con chi² alto siguen ahí pero su target queda
    # capado al mismo valor (min+chi2_clip). El modelo aprende "aquí no hay señal".
    if chi2_clip is not None:
        cap = float(y.min() + chi2_clip)
        n_capped = int((y > cap).sum())
        y = y.clip(upper=cap)
        if verbose:
            print(f"  target clipeado a min+{chi2_clip}={cap:.1f}  ({n_capped} filas afectadas)")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 1. Definimos el dispositivo dinámicamente según lo que detectó train.py al cargar
    auto_device = {"device": "cuda" if _HAS_GPU else "cpu"}
    
    # 2. Fusionamos: [Configuración Base] + [Dispositivo Automático] + [Sobrescrituras del Usuario]
    # El orden es importante: lo que está más a la derecha manda.
    params = {**DEFAULT_XGB_PARAMS, **auto_device, **(hp_overrides or {})}

    # min_delta default: depende del espacio del target (lineal vs log10)
    if early_stopping_min_delta is None:
        early_stopping_min_delta = 1e-4 if log_target else DEFAULT_EARLY_STOPPING_MIN_DELTA

    callbacks = []
    if early_stopping_rounds is not None:
        # Usamos el callback (en vez del kwarg `early_stopping_rounds`) para
        # poder pasar `min_delta`: sólo se cuenta como mejora un descenso de
        # RMSE >= min_delta. Sin esto el modelo entrena las 3000 rondas
        # completas porque la val baja eternamente en cantidades de 1e-5.
        callbacks.append(
            EarlyStopping(
                rounds=early_stopping_rounds,
                min_delta=early_stopping_min_delta,
                save_best=True,
                maximize=False,  # RMSE: minimizar
                metric_name=params.get("eval_metric", "rmse"),
                data_name="validation_1",  # eval_set[1] es la val
            )
        )

    model = xgb.XGBRegressor(**params, callbacks=callbacks or None)

    t0 = time.time()
    # Pasamos train+val al eval_set para tener las DOS curvas en eval_results
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    elapsed = time.time() - t0
    r2 = float(model.score(X_val, y_val))

    eval_results = model.evals_result()  # dict con 'validation_0' (train) y 'validation_1' (val)
    best_iter = int(getattr(model, "best_iteration", model.n_estimators - 1))
    best_score = float(getattr(model, "best_score", float("nan")))

    if verbose:
        n_used = best_iter + 1
        n_total = params.get("n_estimators", 0)
        msg = f"  entrenamiento: {elapsed:.2f}s | R²={r2:.5f} | best_iter={best_iter}/{n_total}"
        if early_stopping_rounds and best_iter < n_total - 1:
            saved = (n_total - n_used) / n_total
            msg += f"  (early stop, ahorro ~{saved:.0%})"
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

    # --- Caché: guardar modelo + info ---
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Siempre guardamos el XGBRegressor "raw"; en load decidimos si envolver
        raw_to_save = final_model.raw_model if log_target else final_model
        raw_to_save.save_model(str(cache_path))
        with open(cache_path.with_suffix(".info.pkl"), "wb") as f:
            pickle.dump(info, f)
        if verbose:
            print(f"  modelo cacheado: {cache_path.name}")

    return final_model, info
