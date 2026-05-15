"""JAX/GPU chi2 for w0waCDM cosmology.

Drop-in GPU counterpart of cosmoml/theory/{sne,bao,joint}.py.
No astropy — comoving distances via cumulative trapezoid on a fixed grid
(N_GRID=500 points up to Z_MAX=2.6, relative error < 1e-5).

Public API
----------
make_chi2_gpu_fn(panth=None, bao=None, planck_prior=False, rd=PLANCK_RD)
    Returns predict_fn(arr: np.ndarray[N, 4]) -> np.ndarray[N]
    (columns: Om, H0, w0, wa). JIT-compiled; runs on GPU if JAX finds one.
    A warm-up call is made inside the factory so JIT compile time is not
    counted in the benchmark.
"""
from __future__ import annotations
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    _JAX_OK = True
except ImportError:
    _JAX_OK = False

from ..config import (
    C_LIGHT,
    PLANCK_H0, PLANCK_H0_ERR,
    PLANCK_OM, PLANCK_OM_ERR,
    PLANCK_RD,
)

N_GRID = 500
Z_MAX  = 2.6   # safely above max DESI BAO z (~2.33)


def _encode_types(types: np.ndarray) -> np.ndarray:
    """Encode BAO observable types to int: DM→0, DH→1, DV→2."""
    out = np.empty(len(types), dtype=np.int32)
    for i, t in enumerate(types):
        q = str(t).upper()
        out[i] = 0 if "DM" in q else (1 if "DH" in q else 2)
    return out


def make_chi2_gpu_fn(panth=None, bao=None,
                     planck_prior: bool = False,
                     rd: float = PLANCK_RD):
    """Build a batched, JIT-compiled GPU predict_fn.

    Parameters
    ----------
    panth : PantheonData | None
        SNe contribution. If None, only BAO (and optionally Planck) are used.
        The JAX implementation assumes use_cepheid_calibrators=False (all SNe
        receive the model luminosity distance) and use_zhel_correction=True.
    bao : DesiBaoData | None
        BAO contribution. If None, only SNe (and optionally Planck) are used.
    planck_prior : bool
        Add Gaussian Planck priors on H0 and Om.
    rd : float
        Sound horizon in Mpc. Defaults to PLANCK_RD.

    Returns
    -------
    predict_fn : (arr: np.ndarray[N, 4]) -> np.ndarray[N]
        arr columns: [Om, H0, w0, wa]. Returns chi2 for each row.
    """
    if not _JAX_OK:
        raise ImportError(
            "JAX is not installed. Install with: pip install 'jax[cuda]'"
        )
    if panth is None and bao is None:
        raise ValueError("At least one of panth or bao must be provided.")

    # ── Fixed integration grid (built once, treated as constants in JIT) ───────
    z_grid_np = np.linspace(0.0, Z_MAX, N_GRID, dtype=np.float32)
    z_grid    = jnp.array(z_grid_np)
    dz        = float(Z_MAX / (N_GRID - 1))

    # ── SNe static arrays ──────────────────────────────────────────────────────
    if panth is not None:
        _z_sne     = jnp.array(panth.z_hd,   dtype=jnp.float32)
        _z_hel_sne = jnp.array(panth.z_hel,  dtype=jnp.float32)
        _mb_sne    = jnp.array(panth.mb,      dtype=jnp.float32)
        _inv_cov_s = jnp.array(panth.inv_cov, dtype=jnp.float32)
        _s_inv_cov = float(panth.sum_inv_cov)  # = 1^T C^-1 1

    # ── BAO static arrays ──────────────────────────────────────────────────────
    if bao is not None:
        _z_bao     = jnp.array(bao.z,       dtype=jnp.float32)
        _val_bao   = jnp.array(bao.val,     dtype=jnp.float32)
        _inv_cov_b = jnp.array(bao.inv_cov, dtype=jnp.float32)
        _type_int  = jnp.array(_encode_types(bao.type), dtype=jnp.int32)
        _rd_f      = float(rd)

    # ── Planck prior constants ─────────────────────────────────────────────────
    _H0_mu, _H0_sig = float(PLANCK_H0), float(PLANCK_H0_ERR)
    _Om_mu, _Om_sig = float(PLANCK_OM), float(PLANCK_OM_ERR)

    # ── Core function (scalar params → scalar chi2) ────────────────────────────
    @jax.jit
    def _chi2_single(Om, H0, w0, wa):
        # 1. E(z) = H(z)/H0 on the fixed grid (CPL)
        f_de   = ((1.0 + z_grid) ** (3.0 * (1.0 + w0 + wa))
                  * jnp.exp(-3.0 * wa * z_grid / (1.0 + z_grid)))
        E_grid = jnp.sqrt(Om * (1.0 + z_grid) ** 3 + (1.0 - Om) * f_de)

        # 2. Cumulative comoving distance via trapezoid rule (Mpc)
        inv_E  = 1.0 / E_grid                            # (N_GRID,)
        traps  = (inv_E[:-1] + inv_E[1:]) * 0.5 * dz   # (N_GRID-1,) slab areas
        dc_grid = jnp.concatenate([
            jnp.zeros(1),
            jnp.cumsum(traps),
        ]) * (C_LIGHT / H0)                              # (N_GRID,) in Mpc

        # 3. SNe chi2 — analytic marginalization over M
        #    Derivation: chi2_marg = delta^T C^-1 delta - (1^T C^-1 delta)^2 / (1^T C^-1 1)
        if panth is not None:
            dc_sne = jnp.interp(_z_sne, z_grid, dc_grid)
            # z_hel correction: dl = (1+z_hd)*dc * (1+z_hel)/(1+z_hd) = dc*(1+z_hel)
            dl_sne = dc_sne * (1.0 + _z_hel_sne)
            mu_th  = 5.0 * jnp.log10(jnp.clip(dl_sne, 1e-8, None)) + 25.0
            delta  = _mb_sne - mu_th
            x      = _inv_cov_s @ delta          # C^-1 @ delta
            chi2_s = delta @ x - jnp.sum(x) ** 2 / _s_inv_cov
        else:
            chi2_s = 0.0

        # 4. BAO chi2 — DM, DH, DV observables
        if bao is not None:
            dc_bao = jnp.interp(_z_bao, z_grid, dc_grid)
            # E at BAO redshifts computed analytically (more accurate than interp)
            f_de_b = ((1.0 + _z_bao) ** (3.0 * (1.0 + w0 + wa))
                      * jnp.exp(-3.0 * wa * _z_bao / (1.0 + _z_bao)))
            E_bao  = jnp.sqrt(Om * (1.0 + _z_bao) ** 3 + (1.0 - Om) * f_de_b)
            dh_bao = C_LIGHT / (H0 * E_bao)
            dv_bao = (_z_bao * dc_bao ** 2 * dh_bao) ** (1.0 / 3.0)
            th_vec = jnp.where(_type_int == 0, dc_bao / _rd_f,
                     jnp.where(_type_int == 1, dh_bao / _rd_f,
                                               dv_bao / _rd_f))
            diff_b = _val_bao - th_vec
            chi2_b = diff_b @ (_inv_cov_b @ diff_b)
        else:
            chi2_b = 0.0

        # 5. Planck Gaussian priors on H0 and Om
        if planck_prior:
            chi2_p = (((H0 - _H0_mu) / _H0_sig) ** 2
                      + ((Om - _Om_mu) / _Om_sig) ** 2)
        else:
            chi2_p = 0.0

        return chi2_s + chi2_b + chi2_p

    # Vectorize over a batch of parameter sets and JIT the whole batch
    _chi2_batched = jax.jit(jax.vmap(_chi2_single, in_axes=(0, 0, 0, 0)))

    def predict_fn(arr: np.ndarray) -> np.ndarray:
        """arr: (N, 4) — columns [Om, H0, w0, wa]. Returns chi2 array (N,)."""
        x = jnp.array(arr, dtype=jnp.float32)
        return np.asarray(_chi2_batched(x[:, 0], x[:, 1], x[:, 2], x[:, 3]))

    # Warm-up: trigger JIT compilation now so it is not counted in the benchmark
    _dummy = np.array([[0.3, 68.0, -1.0, 0.0]], dtype=np.float32)
    predict_fn(_dummy)

    return predict_fn
