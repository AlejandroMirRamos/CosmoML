"""XGBoost training, contours, learning curve and SHAP analysis."""
from .train import train_xgb, LogChi2Model
from .contour import plot_contour_2d, predict_grid
from .curve import plot_learning_curve
from .shap_utils import (
    explain, shap_summary, shap_waterfall, shap_dependence_all, shap_dependence,
)
from .marginal import plot_corner_marginal, run_mcmc_and_getdist, plot_getdist_comparison
from .pipeline import locate_bestfit, build_pipeline_dataset, train_and_shap
from .style import (
    apply_paper_style, texify, feature_labels, style_getdist,
    reassert_usetex, USETEX, PARAM_LABELS,
)


def use_paper_style() -> None:
    """Apply the shared large serif (Palatino/mathpazo) style to all plots."""
    apply_paper_style()


__all__ = [
    "train_xgb", "LogChi2Model",
    "plot_contour_2d", "predict_grid",
    "plot_learning_curve",
    "explain", "shap_summary", "shap_waterfall", "shap_dependence_all",
    "shap_dependence",
    "plot_corner_marginal", "run_mcmc_and_getdist", "plot_getdist_comparison",
    "locate_bestfit", "build_pipeline_dataset", "train_and_shap",
    "use_paper_style", "apply_paper_style", "texify", "feature_labels",
    "style_getdist", "reassert_usetex", "USETEX", "PARAM_LABELS",
]
