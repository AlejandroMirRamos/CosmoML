"""Curva de aprendizaje (train + val por iteración)."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt


def plot_learning_curve(
    info: dict,
    *,
    save_path: str | Path | None = None,
    title: str = "Curva de aprendizaje",
    yscale: str = "log",
    figsize: tuple[float, float] = (9, 4.5),
    show: bool = False,
):
    """Dibuja la métrica de eval vs iteración para train y val.

    `info` es el dict que devuelve `train_xgb` (debe contener 'eval_results',
    'eval_metric', 'best_iteration').
    """
    eval_results = info["eval_results"]
    metric = info.get("eval_metric", "rmse")
    best_iter = info.get("best_iteration", None)

    # XGBoost nombra los eval_set como validation_0, validation_1, ...
    keys = list(eval_results.keys())
    if len(keys) < 2:
        raise ValueError(
            "eval_results sólo tiene un eval_set; pasa eval_set=[(train),(val)] en train_xgb"
        )
    train_curve = eval_results[keys[0]][metric]
    val_curve = eval_results[keys[1]][metric]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(train_curve, label=f"Train {metric.upper()}", lw=1.4)
    ax.plot(val_curve, label=f"Validation {metric.upper()}", lw=1.4)

    if best_iter is not None:
        ax.axvline(best_iter, color="gray", ls="--", alpha=0.6,
                   label=f"best_iter={best_iter}")

    if yscale:
        ax.set_yscale(yscale)
    ax.set_xlabel("Iteración (boosting round)")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        print(f"  guardado: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
