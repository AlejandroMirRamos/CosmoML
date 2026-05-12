# CosmoML

Análisis ML (XGBoost + SHAP) de likelihoods cosmológicos: supernovas Ia
(Pantheon+SH0ES, DES-SN5YR 2024/2025) y BAO (DESI DR2), en distintos modelos
(ΛCDM, wCDM, w₀wₐCDM) y combinaciones (SNe + BAO).

## Estructura

```
CosmoML/
├── cosmoml/            # librería importable (lo que antes se duplicaba en cada script)
│   ├── data/          # loaders: pantheon.py, des.py, desi_bao.py
│   ├── theory/        # χ²: sne.py, bao.py, joint.py
│   ├── ml/            # train.py, contour.py, shap_utils.py
│   ├── sampling.py    # generador de datasets χ² (rodajas + nube + ancla)
│   ├── priors.py      # priors gaussianos Planck
│   └── config.py      # rutas, constantes, fiduciales
├── data/              # datos observacionales (input — no se modifican)
│   ├── pantheon/      # Pantheon+SH0ES.dat / .cov
│   ├── des/           # DES-SN5YR 2024 y 2025
│   └── desi_bao/      # DESI DR2 mean + cov
├── outputs/           # generado (gitignored)
│   ├── datasets/      # CSVs χ² para entrenar XGBoost
│   ├── figures/       # PNGs por escenario
│   └── models/        # (opcional) modelos XGBoost cacheados
├── notebooks/         # un notebook por escenario (a generar en Fase 2)
│   └── figures/       # notebooks de figuras finales (Fig8, Fig12, Fig13)
├── scripts/           # generadores headless (CLI), opcionales
└── legacy/            # scripts originales archivados (a borrar tras validar)
```

## Filosofía

- **`cosmoml/`**: todo lo común (carga de datos, χ² del modelo, sampling, training,
  contornos, SHAP). Importable desde cualquier notebook.
- **`notebooks/<escenario>.ipynb`**: configuración específica de cada escenario
  (modelo, dataset, rangos de parámetros, priors). Carga el CSV cacheado si
  existe, si no lo regenera.
- **`outputs/`**: todo lo que se puede regenerar. Se ignora en git.
- **`data/`**: read-only.

## Setup

```bash
pip install -r requirements.txt
# o, si quieres instalar la librería editable:
pip install -e .
```

## Patrón de uso (notebook)

```python
from cosmoml.data import load_pantheon_plus
from cosmoml.theory.sne import chi2_sne
from cosmoml.sampling import build_chi2_dataset, load_or_build
from cosmoml.ml import train_xgb, plot_contour_2d, shap_summary, use_paper_style

use_paper_style()
sne = load_pantheon_plus()

# Genera o carga el CSV cacheado
df = load_or_build(
    "outputs/datasets/wCDM_Pantheon.csv",
    builder=lambda: build_chi2_dataset(
        chi2_fn=lambda Om, H0, w: chi2_sne(sne, "FlatwCDM", Om=Om, H0=H0, w0=w),
        param_names=["Om", "H0", "w"],
        slices=[
            dict(Om=(0.0, 0.6), w=(-1.8, -0.2), H0=73.04, _n=10000),
            dict(Om=(0.0, 0.6), H0=(60, 85), w=-1.0,    _n=10000),
            dict(H0=(60, 85),   w=(-1.8, -0.2), Om=0.334, _n=10000),
        ],
        random_box=dict(Om=(0.0, 0.65), H0=(55, 90), w=(-2.2, -0.1)),
        n_random=50000,
        anchor=dict(Om=0.334, H0=73.04, w=-1.0),
        n_anchor=2000,
        save_to="outputs/datasets/wCDM_Pantheon.csv",
    ),
)

model, info = train_xgb(df, features=["Om", "H0", "w"], chi2_cut=200)

plot_contour_2d(
    model, features=["Om", "H0", "w"],
    x_param="Om", y_param="w",
    x_range=(0.1, 0.5), y_range=(-2, -0.5),
    fixed=dict(H0=73.04),
    theory_fn=lambda Om, H0, w: chi2_sne(sne, "FlatwCDM", Om=Om, H0=H0, w0=w),
    save_path="outputs/figures/wCDM/wCDM_Om_w.png",
    x_label=r"$\Omega_m$", y_label=r"$w$",
)
```

## Estado de la migración

- ✅ **Fase 1**: librería + reorganización de carpetas (este commit).
- 🚧 **Fase 2**: 1-2 notebooks piloto (próximo paso).
- ⏳ **Fase 3**: el resto de los ~14 escenarios.
- ⏳ **Fase 4**: borrar `legacy/` cuando todos los escenarios estén validados.
