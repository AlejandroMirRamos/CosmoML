# CosmoML

ML analysis (XGBoost + SHAP) of cosmological likelihoods: Type Ia supernovae
(Pantheon+SH0ES, DES-SN5YR 2024/2025) and BAO (DESI DR2), across multiple models
(ΛCDM, wCDM, w₀wₐCDM) and dataset combinations (SNe + BAO, optionally with a
compressed Planck CMB prior). Because the exact cosmological χ² is cheap enough
to sample directly, this application doubles as a controlled validation of the
emulation method: the surrogate posteriors are checked to be statistically
indistinguishable from the exact ones (notebook 07).

## Notebooks

| # | Notebook | Scenarios |
|---|---|---|
| 01 | [SNe Pantheon+SH0ES](notebooks/01_SN_Panth+.ipynb) | [1.1 FlatLCDM 2D](notebooks/01_SN_Panth+.ipynb#1-1) · [1.2 Non-flat ΛCDM 3D](notebooks/01_SN_Panth+.ipynb#1-2) · [1.3 FlatLCDM+M 3D](notebooks/01_SN_Panth+.ipynb#1-3) · [1.4 SALT2 α/β 4D](notebooks/01_SN_Panth+.ipynb#1-4) · [1.5 wCDM 3D](notebooks/01_SN_Panth+.ipynb#1-5) · [1.6 w₀wₐCDM+SH0ES 4D](notebooks/01_SN_Panth+.ipynb#1-6) · [1.7 w₀wₐCDM no SH0ES 4D](notebooks/01_SN_Panth+.ipynb#1-7) · [1.8 w₀wₐCDM z>0.25 4D](notebooks/01_SN_Panth+.ipynb#1-8) |
| 02 | [SNe DES SN5YR](notebooks/02_SN_DES.ipynb) | [2.1 wCDM DES2024 2D](notebooks/02_SN_DES.ipynb#2-1) · [2.2 wCDM DES2025 2D](notebooks/02_SN_DES.ipynb#2-2) · [2.3 w₀wₐCDM DES2024 3D](notebooks/02_SN_DES.ipynb#2-3) · [2.4 w₀wₐCDM DES2025 3D](notebooks/02_SN_DES.ipynb#2-4) |
| 03 | [BAO DESI DR2](notebooks/03_BAO.ipynb) | [3.1 wCDM 2D](notebooks/03_BAO.ipynb#3-1) · [3.2 wCDM+H0 3D](notebooks/03_BAO.ipynb#3-2) · [3.3 w₀wₐCDM full 3D](notebooks/03_BAO.ipynb#3-3) · [3.4 w₀wₐCDM z<2 3D](notebooks/03_BAO.ipynb#3-4) · [3.5 w₀wₐCDM+rd Planck 4D](notebooks/03_BAO.ipynb#3-5) · [3.6 H0·rd 4D](notebooks/03_BAO.ipynb#3-6) · [3.7 H0–rd degeneracy 5D](notebooks/03_BAO.ipynb#3-7) · [3.8 +priors Om/H0/rd 5D](notebooks/03_BAO.ipynb#3-8) · [3.9 +5 Planck priors 5D](notebooks/03_BAO.ipynb#3-9) |
| 04 | [Joint SNe + BAO](notebooks/04_SN+BAO.ipynb) | [4.1 Joint w₀wₐCDM 4D](notebooks/04_SN+BAO.ipynb#4-1) · [4.2 SNe vs BAO Ωm=0.40 prior](notebooks/04_SN+BAO.ipynb#4-2) · [4.3 w₀–wₐ overlay ellipses](notebooks/04_SN+BAO.ipynb#4-3) · [4.4 1D constraint wₐ](notebooks/04_SN+BAO.ipynb#4-4) |
| 05 | [Special paper figures](notebooks/05_Figures.ipynb) | [5.1 Fig 8 — ΛCDM BAO (Ωm, H0·rd)](notebooks/05_Figures.ipynb#5-1) · [5.2 Fig 12 — wCDM overlay 3 datasets](notebooks/05_Figures.ipynb#5-2) · [5.3 Fig 13 — μ(z) residuals](notebooks/05_Figures.ipynb#5-3) · [5.4 CPL-sufficiency test: CPL vs 4th-order Taylor w(z), exact-χ² cross-check](notebooks/05_Figures.ipynb#5-4) |
| 06 | [Paper: full pipeline](notebooks/06_Paper.ipynb) | [6.1 ΛCDM · Pantheon+ + BAO](notebooks/06_Paper.ipynb#section-1) · [6.2.1a w₀wₐCDM · Pantheon+ + BAO](notebooks/06_Paper.ipynb#section-2) · [6.2.1b + CMB](notebooks/06_Paper.ipynb#section-2) · [6.2.2a DES-2024 + BAO](notebooks/06_Paper.ipynb#section-2) · [6.2.2b + CMB](notebooks/06_Paper.ipynb#section-2) · [6.2.3a DES-2025 + BAO](notebooks/06_Paper.ipynb#section-2) · [6.2.3b + CMB](notebooks/06_Paper.ipynb#section-2) · [6.3 Final summary: BAO / BAO+Pantheon+ / BAO+Pantheon++CMB](notebooks/06_Paper.ipynb#section-3) |
| 07 | [Benchmark: ML vs Theory](notebooks/07_Benchmark.ipynb) | ML surrogate MCMC vs exact-χ² MCMC with interchangeable back-ends — Astropy reference (`ProcessPoolExecutor`), vectorized NumPy (CPU) and JAX `vmap` (GPU) — plus GPU-vs-CPU dataset generation, across the BAO / +Pantheon+ / +CMB scenarios; all back-ends give statistically identical posteriors |

## Structure

```
CosmoML/
├── cosmoml/            # importable library (shared across all notebooks)
│   ├── data/          # loaders: pantheon.py, des.py, desi_bao.py
│   ├── theory/        # χ²: sne.py, bao.py, joint.py, numpy_theory.py, jax_theory.py (GPU)
│   ├── ml/            # train.py, contour.py, shap_utils.py
│   ├── sampling.py    # χ² dataset generator (slices + cloud + anchor)
│   ├── priors.py      # Planck Gaussian priors
│   └── config.py      # paths, constants, fiducials
├── data/              # observational data (input — read-only)
│   ├── pantheon/      # Pantheon+SH0ES.dat / .cov
│   ├── des/           # DES-SN5YR 2024 and 2025
│   └── desi_bao/      # DESI DR2 mean + cov
├── notebooks/         # one notebook per scenario (01–07)
└── outputs/           # generated (gitignored)
    ├── datasets/      # χ² CSVs for XGBoost training
    ├── figures/       # PNGs per scenario
    └── models/        # (optional) cached XGBoost models
```

## Methodology (paper pipeline, notebooks 06–07)

- **Training design**: for each data combination, the χ² training set combines
  2D slices through the best fit, a uniform space-filling draw over the full
  prior box (Ωm ∈ [0.1, 0.9], H0 ∈ [20, 100], w₀ ∈ [−3, 0.2], wₐ ∈ [−3, 2]) and
  a Gaussian cloud whose covariance is the Hessian of the fit, so the narrow
  w₀–wₐ degeneracy is sampled densely. Each design has 2.5–5×10⁵ points, with
  the exact distance integrals evaluated in parallel.
- **Shifted-log₁₀ training target** (the key problem-specific ingredient,
  `cosmoml/ml/train.py`):

  `y = log10(χ² − χ²_min + 1)`

  The cosmological χ² spans from ~10³ at the best fit to ≳10⁶ at the box edges,
  while the physics lives in the Δχ² ~ 2–6 band near the minimum; χ²_min itself
  varies from ≈ 5.6 (BAO only) to ≈ 1700 (SN+BAO). Training on raw χ² lets the
  tail dominate the loss, and a plain log₁₀ only works when χ²_min happens to be
  small. The shift maps every best fit to y = 0 (same steep, well-resolved part
  of the transform regardless of χ²_min) and the +1 avoids log(0) at the densely
  sampled minimum. Predictions are mapped back to linear χ² through the exact
  inverse. With this target the emulators reach R² = 0.997–0.9999 on held-out
  validation sets across all data combinations.
- **Compressed CMB prior** (`cosmoml/priors.py`): instead of recomputing CMB
  spectra, the Planck PR4+lensing constraint enters as a Gaussian prior with a
  full 4×4 covariance for (H0, ωm, w₀, wₐ), obtained from the published chain
  covariance by collapsing the (H0, ωb, ωcdm) block onto (H0, ωm) via linear
  error propagation. The H0–ωm correlation is retained; w₀ and wₐ enter with
  σ(w₀) = 0.02, σ(wₐ) = 0.05.
- **Posterior sampling**: a parallel Random-Walk Metropolis–Hastings sampler
  (1024 chains, proposal covariance set to the fit Hessian, effective-sample-size
  stopping) runs directly on the emulated likelihood; contours are rendered with
  GetDist.

## Results (paper)

- **ΛCDM baseline** (Pantheon+ & DESI BAO): Ωm = 0.310 ± 0.008,
  H0 = 68.4 ± 0.5 km/s/Mpc at χ²_min = 1703.6 (R² = 0.99988), matching the
  concordance picture.
- **w₀wₐCDM**: without the CMB prior all three SN samples (Pantheon+, DES-2024,
  DES-2025) combined with BAO prefer dynamical dark energy (w₀ > −1, wₐ < 0,
  e.g. w₀ = −0.766, wₐ = −0.785 for Pantheon+ & BAO), the same hint reported by
  DESI. Adding the CMB prior pulls every fit back to ΛCDM (w₀ ≈ −0.96, wₐ ≈ 0.02)
  and shrinks σ(wₐ) by ~10× (0.42 → 0.045).
- **Benchmark** (07): with identical sampler settings, the surrogate completes
  each posterior run in 4.8–7.4 s vs the exact χ² backends — 3.6–22.4× faster
  than vectorized NumPy on CPU and 7.2–12.1× faster than the JAX GPU kernel —
  and all four back-ends give statistically identical posteriors.
- **SHAP**: the importance ranking follows the expected physics (w₀ above wₐ
  from low-redshift leverage; background parameters dominate without the CMB
  prior) and reorganises when the CMB prior is added (w₀, wₐ jump to the top).

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
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Or install the library in editable mode (includes all dependencies)
pip install -e .
```

For GPU support (JAX — optional, needed for the GPU sections of the 07 benchmark):

```bash
pip install "jax[cuda]"          # NVIDIA GPU (CUDA)
# pip install "jax[cpu]"         # CPU-only fallback
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

## Status

- ✅ Library + folder reorganisation (`cosmoml/`).
- ✅ Scenario notebooks 01–04.
- ✅ Special figures (05) + full paper pipeline (06).
- ✅ ML-vs-theory benchmark (07); legacy scripts removed.
