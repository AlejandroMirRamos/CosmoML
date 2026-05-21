"""Fast vectorized numpy chi2 for w0waCDM (no astropy, no JAX).

Mirrors jax_theory.py but uses numpy broadcasting so it runs on any CPU
without JAX installed. predict_fn(arr) accepts an (N, 4) array and returns
(N,) chi2 values in a single batched call — much faster than N independent
astropy/scipy evaluations.

Public API
----------
make_chi2_numpy_fn(panth=None, bao=None, planck_prior=False, rd=PLANCK_RD)
    Returns predict_fn(arr: np.ndarray[N, 4]) -> np.ndarray[N]
    (columns: Om, H0, w0, wa).
"""
from __future__ import annotations
import numpy as np

from ..config import (
    C_LIGHT, PLANCK_RD,
    PLANCK_H0, PLANCK_OM, PLANCK_CMB_W0, PLANCK_CMB_WA,
)
from ..priors import _build_cmb_inv_cov

N_GRID = 500
Z_MAX  = 2.6


def _encode_types(types: np.ndarray) -> np.ndarray:
    out = np.empty(len(types), dtype=np.int32)
    for i, t in enumerate(types):
        q = str(t).upper()
        out[i] = 0 if "DM" in q else (1 if "DH" in q else 2)
    return out


def make_chi2_numpy_fn(panth=None, bao=None,
                       planck_prior: bool = False,
                       rd: float = PLANCK_RD):
    """Build a batched numpy predict_fn equivalent to make_chi2_gpu_fn.

    All static arrays (covariance matrices, redshifts, interpolation indices)
    are precomputed once at factory time so predict_fn only does arithmetic.
    """
    if panth is None and bao is None:
        raise ValueError("At least one of panth or bao must be provided.")

    z_grid = np.linspace(0.0, Z_MAX, N_GRID, dtype=np.float64)
    dz     = float(Z_MAX / (N_GRID - 1))

    if panth is not None:
        _z_sne     = panth.z_hd.astype(np.float64)
        _z_hel_sne = panth.z_hel.astype(np.float64)
        _mb_sne    = panth.mb.astype(np.float64)
        _inv_cov_s = panth.inv_cov.astype(np.float64)
        _s_inv_cov = float(panth.sum_inv_cov)
        # Precompute fixed interpolation indices for SNe redshifts
        _idx_s = np.searchsorted(z_grid, _z_sne, side='right').clip(1, N_GRID - 1)
        _t_s   = ((_z_sne - z_grid[_idx_s - 1])
                  / (z_grid[_idx_s] - z_grid[_idx_s - 1]))

    if bao is not None:
        _z_bao    = bao.z.astype(np.float64)
        _val_bao  = bao.val.astype(np.float64)
        _inv_cov_b = bao.inv_cov.astype(np.float64)
        _type_int  = _encode_types(bao.type)
        _rd_f      = float(rd)
        # Precompute fixed interpolation indices for BAO redshifts
        _idx_b = np.searchsorted(z_grid, _z_bao, side='right').clip(1, N_GRID - 1)
        _t_b   = ((_z_bao - z_grid[_idx_b - 1])
                  / (z_grid[_idx_b] - z_grid[_idx_b - 1]))

    # Precompute CMB inverse-covariance and best-fit omega_m once at factory time.
    if planck_prior:
        _cmb_inv_cov  = _build_cmb_inv_cov()
        _omega_m_bf   = float(PLANCK_OM * (PLANCK_H0 / 100.0) ** 2)
        _H0_bf        = float(PLANCK_H0)
        _w0_bf        = float(PLANCK_CMB_W0)
        _wa_bf        = float(PLANCK_CMB_WA)

    def predict_fn(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float64)
        N   = len(arr)
        Om  = arr[:, 0]  # (N,)
        H0  = arr[:, 1]
        w0  = arr[:, 2]
        wa  = arr[:, 3]

        # E(z) on the fixed grid — (N, N_GRID)
        z      = z_grid[None, :]                          # (1, N_GRID)
        f_de   = ((1.0 + z) ** (3.0 * (1.0 + w0[:, None] + wa[:, None]))
                  * np.exp(-3.0 * wa[:, None] * z / (1.0 + z)))
        E_grid = np.sqrt(Om[:, None] * (1.0 + z) ** 3
                         + (1.0 - Om[:, None]) * f_de)   # (N, N_GRID)

        # Cumulative comoving distance via trapezoid — (N, N_GRID)
        inv_E  = 1.0 / E_grid
        slabs  = (inv_E[:, :-1] + inv_E[:, 1:]) * 0.5 * dz
        dc_grid = np.concatenate(
            [np.zeros((N, 1)), np.cumsum(slabs, axis=1)], axis=1
        ) * (C_LIGHT / H0[:, None])                      # (N, N_GRID)

        chi2 = np.zeros(N)

        # SNe chi2 with analytic marginalization over M
        if panth is not None:
            dc_sne = (dc_grid[:, _idx_s - 1] * (1.0 - _t_s)
                      + dc_grid[:, _idx_s]   * _t_s)     # (N, M)
            dl_sne = dc_sne * (1.0 + _z_hel_sne)
            mu_th  = 5.0 * np.log10(np.clip(dl_sne, 1e-8, None)) + 25.0
            delta  = _mb_sne[None, :] - mu_th            # (N, M)
            # x = C^-1 @ delta_i for all i via (M,M)@(M,N)
            x      = _inv_cov_s @ delta.T                 # (M, N)
            chi2  += (np.sum(delta.T * x, axis=0)
                      - np.sum(x, axis=0) ** 2 / _s_inv_cov)

        # BAO chi2
        if bao is not None:
            dc_bao  = (dc_grid[:, _idx_b - 1] * (1.0 - _t_b)
                       + dc_grid[:, _idx_b]   * _t_b)    # (N, B)
            f_de_b  = ((1.0 + _z_bao) ** (3.0 * (1.0 + w0[:, None] + wa[:, None]))
                       * np.exp(-3.0 * wa[:, None] * _z_bao / (1.0 + _z_bao)))
            E_bao   = np.sqrt(Om[:, None] * (1.0 + _z_bao) ** 3
                              + (1.0 - Om[:, None]) * f_de_b)
            dh_bao  = C_LIGHT / (H0[:, None] * E_bao)
            dv_bao  = (_z_bao * dc_bao ** 2 * dh_bao) ** (1.0 / 3.0)
            th_vec  = np.where(_type_int == 0, dc_bao / _rd_f,
                      np.where(_type_int == 1, dh_bao / _rd_f,
                                               dv_bao / _rd_f))
            diff    = _val_bao - th_vec                   # (N, B)
            x_b     = diff @ _inv_cov_b                   # (N, B)
            chi2   += np.sum(x_b * diff, axis=1)

        # CMB covariance-matrix prior: chi2 = Delta^T InvCov Delta
        # Delta = (H0 - H0_bf, omega_m - omega_m_bf, w0 - w0_bf, wa - wa_bf)
        if planck_prior:
            omega_m   = Om * (H0 / 100.0) ** 2
            delta_cmb = np.stack([
                H0 - _H0_bf,
                omega_m - _omega_m_bf,
                w0 - _w0_bf,
                wa - _wa_bf,
            ], axis=1)                                  # (N, 4)
            chi2 += np.einsum('ni,ij,nj->n', delta_cmb, _cmb_inv_cov, delta_cmb)

        return chi2

    return predict_fn
