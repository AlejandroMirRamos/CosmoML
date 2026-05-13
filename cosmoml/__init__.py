"""CosmoML — ML-based analysis of cosmological likelihoods (SNe Ia + BAO).

Typical notebook usage:

    from cosmoml.data import load_pantheon_plus, load_desi_bao
    from cosmoml.theory.sne import chi2_sne
    from cosmoml.theory.bao import chi2_bao
    from cosmoml.sampling import build_chi2_dataset
    from cosmoml.ml import train_xgb, plot_contour_2d, shap_summary
"""
from .config import (
    PROJECT_ROOT, DATA_DIR, OUTPUTS_DIR,
    DEFAULT_XGB_PARAMS, CONF_LEVELS_2D,
)

__all__ = [
    "PROJECT_ROOT", "DATA_DIR", "OUTPUTS_DIR",
    "DEFAULT_XGB_PARAMS", "CONF_LEVELS_2D",
]
