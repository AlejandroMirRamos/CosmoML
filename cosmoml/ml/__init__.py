"""Entrenamiento, contornos, curva de aprendizaje y análisis SHAP."""
import matplotlib.pyplot as plt

from .train import train_xgb, LogChi2Model
from .contour import plot_contour_2d, predict_grid
from .curve import plot_learning_curve
from .shap_utils import (
    explain, shap_summary, shap_waterfall, shap_dependence_all, shap_dependence,
)


def use_paper_style() -> None:
    """Estilo común a todos los plots (serif, sin TeX)."""
    plt.rcParams["text.usetex"] = False
    plt.rcParams["font.family"] = "serif"


__all__ = [
    "train_xgb", "LogChi2Model",
    "plot_contour_2d", "predict_grid",
    "plot_learning_curve",
    "explain", "shap_summary", "shap_waterfall", "shap_dependence_all",
    "shap_dependence",
    "use_paper_style",
]
