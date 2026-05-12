"""Resúmenes y dependencias SHAP para modelos XGBoost ya entrenados.

Implementa el panel completo que usamos en los notebooks:
- summary beeswarm + bar (vista global)
- waterfall en 1 muestra (descomposición local)
- scatter/dependence por feature (cómo afecta cada parámetro al χ²)
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import shap


def explain(model, X, n_sample: int = 1000, seed: int = 42):
    """Crea TreeExplainer y devuelve (shap_values, X_sampled).

    Si el modelo es un LogChi2Model (entrenado en log10(χ²)), opera sobre el
    XGBRegressor subyacente — los valores SHAP describen contribuciones a
    log10(χ²), no a χ² lineal. Una contribución de 0.5 ≈ factor 3× en χ².
    """
    inner = getattr(model, "raw_model", model)
    explainer = shap.TreeExplainer(inner)
    X_s = X.sample(n=min(n_sample, len(X)), random_state=seed)
    return explainer(X_s, check_additivity=False), X_s


def shap_summary(
    model, X, *,
    save_dir: str | Path,
    prefix: str,
    n_sample: int = 1000,
    title: str = "",
    show: bool = False,
):
    """Beeswarm + bar plot. Devuelve (shap_values, X_sampled) para reutilizar."""
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    shap_v, X_s = explain(model, X, n_sample=n_sample)

    plt.figure()
    if title:
        plt.title(f"{title} (beeswarm)")
    shap.plots.beeswarm(shap_v, show=False)
    plt.tight_layout()
    out_bs = save_dir / f"{prefix}_shap_beeswarm.png"
    plt.savefig(out_bs, dpi=300, bbox_inches="tight"); print(f"  guardado: {out_bs}")
    plt.show() if show else plt.close()

    plt.figure()
    if title:
        plt.title(f"{title} (bar)")
    shap.plots.bar(shap_v, show=False)
    plt.tight_layout()
    out_bar = save_dir / f"{prefix}_shap_bar.png"
    plt.savefig(out_bar, dpi=300, bbox_inches="tight"); print(f"  guardado: {out_bar}")
    plt.show() if show else plt.close()

    return shap_v, X_s


def shap_waterfall(
    shap_values, *,
    idx: int = 0,
    save_path: str | Path,
    title: str = "",
    show: bool = False,
):
    """Waterfall plot para una muestra (descomposición local del χ² predicho)."""
    plt.figure()
    if title:
        plt.title(title)
    shap.plots.waterfall(shap_values[idx], show=False)
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  guardado: {save_path}")
    plt.show() if show else plt.close()


def shap_dependence_all(
    shap_values, X, *,
    save_dir: str | Path,
    prefix: str,
    show: bool = False,
):
    """Un scatter/dependence por feature, coloreado por todas las SHAP (interacciones).

    Guarda como `{prefix}_{col}.png` (sin "shap_dep_" intermedio para mantener
    nombres cortos compatibles con la convención de figuras del notebook 03).
    """
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    for col in X.columns:
        plt.figure()
        shap.plots.scatter(shap_values[:, col], color=shap_values, show=False)
        plt.tight_layout()
        out = save_dir / f"{prefix}_shap_{col}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  guardado: {out}")
        plt.show() if show else plt.close()
        out_paths.append(out)
    return out_paths


# Alias retro-compatible: quien usaba shap_dependence(feature, ...) sigue funcionando
def shap_dependence(
    shap_values, X, feature: str, *,
    save_path: str | Path,
    show: bool = False,
):
    plt.figure()
    shap.plots.scatter(shap_values[:, feature], color=shap_values, show=False)
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  guardado: {save_path}")
    plt.show() if show else plt.close()
