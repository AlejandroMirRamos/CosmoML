"""SHAP analysis for trained XGBoost chi2 models.

Provides the full panel used across the notebooks:
  - summary (beeswarm + bar): global feature importance.
  - waterfall: per-sample additive decomposition.
  - scatter/dependence: per-feature effect coloured by SHAP interactions.

Note: when the model is a ``LogChi2Model`` (shifted-log10 target), the SHAP
values describe contributions to log10(chi2 - chi2_min + 1), NOT to linear chi2.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import shap


def explain(model, X, n_sample: int = 1000, seed: int = 42):
    """Build a TreeExplainer and return (shap_values, X_sampled).

    Uses ``check_additivity=False`` because ``LogChi2Model.predict`` applies an
    inverse-log transform that SHAP cannot trace.
    """
    inner = getattr(model, "raw_model", model)
    explainer = shap.TreeExplainer(inner)
    X_s = X.sample(n=min(n_sample, len(X)), random_state=seed)
    return explainer(X_s, check_additivity=False), X_s


def shap_summary(
    model, X, *,
    save_dir: str | Path | None = None,
    prefix: str = "",
    n_sample: int = 1000,
    title: str = "",
    show: bool = False,
):
    """Beeswarm + bar plot. Returns ``(shap_values, X_sampled)`` for reuse."""
    if save_dir is not None:
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    shap_v, X_s = explain(model, X, n_sample=n_sample)

    plt.figure()
    if title:
        plt.title(f"{title} (beeswarm)")
    shap.plots.beeswarm(shap_v, show=False)
    plt.tight_layout()
    if save_dir is not None:
        out_bs = save_dir / f"{prefix}_shap_beeswarm.png"
        plt.savefig(out_bs, dpi=300, bbox_inches="tight"); print(f"  saved: {out_bs}")
    plt.show() if show else plt.close()

    plt.figure()
    if title:
        plt.title(f"{title} (bar)")
    shap.plots.bar(shap_v, show=False)
    plt.tight_layout()
    if save_dir is not None:
        out_bar = save_dir / f"{prefix}_shap_bar.png"
        plt.savefig(out_bar, dpi=300, bbox_inches="tight"); print(f"  saved: {out_bar}")
    plt.show() if show else plt.close()

    return shap_v, X_s


def shap_waterfall(
    shap_values, *,
    idx: int = 0,
    save_path: str | Path | None = None,
    title: str = "",
    show: bool = False,
):
    """Waterfall plot for one sample (per-feature additive decomposition)."""
    plt.figure()
    if title:
        plt.title(title)
    shap.plots.waterfall(shap_values[idx], show=False)
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  saved: {save_path}")
    plt.show() if show else plt.close()


def shap_dependence_all(
    shap_values, X, *,
    save_dir: str | Path | None = None,
    prefix: str = "",
    show: bool = False,
):
    """One scatter/dependence plot per feature, coloured by all SHAP interactions."""
    if save_dir is not None:
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for col in X.columns:
        plt.figure()
        shap.plots.scatter(shap_values[:, col], color=shap_values, show=False)
        plt.tight_layout()
        if save_dir is not None:
            out = save_dir / f"{prefix}_shap_{col}.png"
            plt.savefig(out, dpi=300, bbox_inches="tight")
            print(f"  saved: {out}")
            out_paths.append(out)
        plt.show() if show else plt.close()
    return out_paths


def shap_dependence(
    shap_values, X, feature: str, *,
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Single dependence plot for a named feature."""
    plt.figure()
    shap.plots.scatter(shap_values[:, feature], color=shap_values, show=False)
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  saved: {save_path}")
    plt.show() if show else plt.close()
