"""Constantes globales y rutas del proyecto."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATASETS_DIR = OUTPUTS_DIR / "datasets"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"

C_LIGHT = 299792.458  # km/s

# Valores fiduciales (Planck 2018) usados como referencia
PLANCK_H0 = 67.36
PLANCK_OM = 0.3153
PLANCK_RD = 147.09
PLANCK_RD_ERR = 0.26
PLANCK_H0_ERR = 0.54
PLANCK_OM_ERR = 0.0073

# Niveles de Δχ² para contornos 2D (1σ y 2σ con 2 dof)
CONF_LEVELS_2D = (2.30, 6.18)

# Hiperparámetros XGBoost por defecto
# - n_estimators / learning_rate / max_depth: igual que los scripts originales
#   (mantienen la comparación justa de tiempos con la versión legacy).
# - tree_method='hist': implementación CPU rápida basada en histogramas; mismo
#   resultado que 'exact' en términos prácticos pero ~2-3× más rápida en datasets
#   de 10k-1M filas. Sin caché ni atajos en el χ², no cambia el benchmark.
# - eval_metric='rmse': métrica natural para regresión χ²; usada por el early
#   stopping y la curva de aprendizaje.
DEFAULT_XGB_PARAMS = dict(
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=6,
    tree_method="hist",
    eval_metric="rmse",
    n_jobs=-1,
)

# Early stopping: paramos cuando la val no mejora en N rondas, considerando
# sólo mejoras "significativas" (mayores que MIN_DELTA en RMSE).
#
# Con min_delta=0 (default de XGBoost) la val baja eternamente — incluso un
# Δ=1e-6 se cuenta como mejora y el modelo entrena las 3000 rondas completas.
# Para una función χ² con valores ~1700-2600 una mejora <0.01 en RMSE es ruido
# numérico invisible para los contornos. Pasamos a 0.01 como umbral de
# significancia → el entrenamiento corta cuando ha convergido de verdad.
DEFAULT_EARLY_STOPPING_ROUNDS = 50
DEFAULT_EARLY_STOPPING_MIN_DELTA = 0.01
