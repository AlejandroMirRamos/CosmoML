# CosmoML

ML analysis (XGBoost + SHAP) of cosmological likelihoods: Type Ia supernovae
(Pantheon+SH0ES, DES-SN5YR 2024/2025) and BAO (DESI DR2), across multiple models
(ΛCDM, wCDM, w₀wₐCDM) and dataset combinations (SNe + BAO).

## Notebooks

| # | Notebook | Scenarios |
|---|---|---|
| 01 | [SNe Pantheon+SH0ES](notebooks/01_SN_Panth+.ipynb) | [1.1 FlatLCDM 2D](notebooks/01_SN_Panth+.ipynb#1-1) · [1.2 Non-flat ΛCDM 3D](notebooks/01_SN_Panth+.ipynb#1-2) · [1.3 FlatLCDM+M 3D](notebooks/01_SN_Panth+.ipynb#1-3) · [1.4 SALT2 α/β 4D](notebooks/01_SN_Panth+.ipynb#1-4) · [1.5 wCDM 3D](notebooks/01_SN_Panth+.ipynb#1-5) · [1.6 w₀wₐCDM+SH0ES 4D](notebooks/01_SN_Panth+.ipynb#1-6) · [1.7 w₀wₐCDM no SH0ES 4D](notebooks/01_SN_Panth+.ipynb#1-7) · [1.8 w₀wₐCDM z>0.25 4D](notebooks/01_SN_Panth+.ipynb#1-8) |
| 02 | [SNe DES SN5YR](notebooks/02_SN_DES.ipynb) | [2.1 wCDM DES2024 2D](notebooks/02_SN_DES.ipynb#2-1) · [2.2 wCDM DES2025 2D](notebooks/02_SN_DES.ipynb#2-2) · [2.3 w₀wₐCDM DES2024 3D](notebooks/02_SN_DES.ipynb#2-3) · [2.4 w₀wₐCDM DES2025 3D](notebooks/02_SN_DES.ipynb#2-4) |
| 03 | [BAO DESI DR2](notebooks/03_BAO.ipynb) | [3.1 wCDM 2D](notebooks/03_BAO.ipynb#3-1) · [3.2 wCDM+H0 3D](notebooks/03_BAO.ipynb#3-2) · [3.3 w₀wₐCDM full 3D](notebooks/03_BAO.ipynb#3-3) · [3.4 w₀wₐCDM z<2 3D](notebooks/03_BAO.ipynb#3-4) · [3.5 w₀wₐCDM+rd Planck 4D](notebooks/03_BAO.ipynb#3-5) · [3.6 H0·rd 4D](notebooks/03_BAO.ipynb#3-6) · [3.7 H0–rd degeneracy 5D](notebooks/03_BAO.ipynb#3-7) · [3.8 +priors Om/H0/rd 5D](notebooks/03_BAO.ipynb#3-8) · [3.9 +5 Planck priors 5D](notebooks/03_BAO.ipynb#3-9) |
| 04 | [Joint SNe + BAO](notebooks/04_SN+BAO.ipynb) | [4.1 Joint w₀wₐCDM 4D](notebooks/04_SN+BAO.ipynb#4-1) · [4.2 SNe vs BAO Ωm=0.40 prior](notebooks/04_SN+BAO.ipynb#4-2) · [4.3 w₀–wₐ overlay ellipses](notebooks/04_SN+BAO.ipynb#4-3) · [4.4 1D constraint wₐ](notebooks/04_SN+BAO.ipynb#4-4) |
| 05 | [Special paper figures](notebooks/05_Figures.ipynb) | [5.1 Fig 8 — ΛCDM BAO (Ωm, H0·rd)](notebooks/05_Figures.ipynb#5-1) · [5.2 Fig 12 — wCDM overlay 3 datasets](notebooks/05_Figures.ipynb#5-2) · [5.3 Fig 13 — μ(z) residuals](notebooks/05_Figures.ipynb#5-3) · [5.4 ML test: CPL vs Taylor 4th order](notebooks/05_Figures.ipynb#5-4) |
| 06 | [Paper: full pipeline](notebooks/06_Paper.ipynb) | [6.1 ΛCDM · Pantheon+ + BAO](notebooks/06_Paper.ipynb#section-1) · [6.2.1a w₀wₐCDM · Pantheon+ + BAO](notebooks/06_Paper.ipynb#section-2) · [6.2.1b + CMB](notebooks/06_Paper.ipynb#section-2) · [6.2.2a DES-2024 + BAO](notebooks/06_Paper.ipynb#section-2) · [6.2.2b + CMB](notebooks/06_Paper.ipynb#section-2) · [6.2.3a DES-2025 + BAO](notebooks/06_Paper.ipynb#section-2) · [6.2.3b + CMB](notebooks/06_Paper.ipynb#section-2) · [6.3 Final summary: BAO / BAO+Pantheon+ / BAO+Pantheon++CMB](notebooks/06_Paper.ipynb#section-3) |

## Structure

```
CosmoML/
├── cosmoml/            # importable library (shared across all notebooks)
│   ├── data/          # loaders: pantheon.py, des.py, desi_bao.py
│   ├── theory/        # χ²: sne.py, bao.py, joint.py
│   ├── ml/            # train.py, contour.py, shap_utils.py
│   ├── sampling.py    # χ² dataset generator (slices + cloud + anchor)
│   ├── priors.py      # Planck Gaussian priors
│   └── config.py      # paths, constants, fiducials
├── data/              # observational data (input — read-only)
│   ├── pantheon/      # Pantheon+SH0ES.dat / .cov
│   ├── des/           # DES-SN5YR 2024 and 2025
│   └── desi_bao/      # DESI DR2 mean + cov
├── outputs/           # generated (gitignored)
│   ├── datasets/      # χ² CSVs for XGBoost training
│   ├── figures/       # PNGs per scenario
│   └── models/        # (optional) cached XGBoost models
├── notebooks/         # one notebook per scenario
│   └── figures/       # final paper figures (Fig8, Fig12, Fig13)
├── scripts/           # headless generators (CLI), optional
└── legacy/            # archived original scripts (to be deleted after validation)
```

## Design principles

- **`cosmoml/`**: all shared logic (data loading, model χ², sampling, training,
  contours, SHAP). Importable from any notebook.
- **`notebooks/<scenario>.ipynb`**: scenario-specific configuration
  (model, dataset, parameter ranges, priors). Loads the cached CSV if it
  exists, otherwise regenerates it.
- **`outputs/`**: everything that can be regenerated. Git-ignored.
- **`data/`**: read-only.

## Setup

```bash
pip install -r requirements.txt
# or, to install the library in editable mode:
pip install -e .
```

## Usage pattern (notebook)

```python
from cosmoml.data import load_pantheon_plus
from cosmoml.theory.sne import chi2_sne
from cosmoml.sampling import build_chi2_dataset, load_or_build
from cosmoml.ml import train_xgb, plot_contour_2d, shap_summary, use_paper_style

use_paper_style()
sne = load_pantheon_plus()

# Generate or load the cached CSV
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

## Migration status

- ✅ **Phase 1**: library + folder reorganisation (this commit).
- ✅ **Phase 2**: pilot notebooks 01–04.
- ✅ **Phase 3**: special figures notebook (05) + full paper pipeline (06).
- ⏳ **Phase 4**: delete `legacy/` once all scenarios are validated.
