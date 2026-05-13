"""Project-wide constants and paths."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATASETS_DIR = OUTPUTS_DIR / "datasets"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"

C_LIGHT = 299792.458  # km/s

# Planck 2018 fiducial values used as reference points.
PLANCK_H0 = 67.36
PLANCK_OM = 0.3153
PLANCK_RD = 147.09
PLANCK_RD_ERR = 0.26
PLANCK_H0_ERR = 0.54
PLANCK_OM_ERR = 0.0073

# Delta-chi2 levels for 2D contours (1-sigma and 2-sigma with 2 dof).
CONF_LEVELS_2D = (2.30, 6.18)

# Default XGBoost hyperparameters. tree_method='hist' is the fast CPU histogram
# implementation; eval_metric='rmse' is used by both early stopping and the
# learning curve plot.
DEFAULT_XGB_PARAMS = dict(
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=6,
    tree_method="hist",
    eval_metric="rmse",
    n_jobs=-1,
)

# Early stopping: stop when validation has not improved by at least MIN_DELTA
# for ROUNDS consecutive iterations. The min_delta default is overridden in
# train_xgb based on whether log_target is used (1e-4 for log10, else 0.01).
DEFAULT_EARLY_STOPPING_ROUNDS = 50
DEFAULT_EARLY_STOPPING_MIN_DELTA = 0.01
