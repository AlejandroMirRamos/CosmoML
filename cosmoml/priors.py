"""CMB Gaussian priors using the full Planck PR4 covariance matrix (CPL)."""
from pathlib import Path

from .config import (
    PLANCK_H0, PLANCK_OM,
    PLANCK_CMB_W0, PLANCK_CMB_WA,
)

_COVMAT_PATH = Path(__file__).resolve().parent / 'data' / 'PR4_lensing_CPL.covmat'

# Cached inverse-covariance (lazy singleton).
_CMB_INV_COV = None


def _build_cmb_inv_cov():
    """Parse PR4_lensing_CPL.covmat and return InvCov for (H0, omega_m, w0, wa).

    The covmat contains 25 parameters.  We extract the 5×5 block for
    (H0, omega_b, omega_cdm, w0_fld, wa_fld) at column/row indices [2,3,4,5,6],
    then project (omega_b, omega_cdm) → omega_m = omega_b + omega_cdm to obtain
    a 4×4 covariance for (H0, omega_m, w0, wa), and return its inverse.
    """
    import numpy as np

    with open(_COVMAT_PATH) as f:
        f.readline()                          # skip header
        rows = [list(map(float, line.split())) for line in f if line.strip()]
    C = np.array(rows)

    # 5×5 submatrix: H0(2), omega_b(3), omega_cdm(4), w0(5), wa(6)
    idx = [2, 3, 4, 5, 6]
    C_sub = C[np.ix_(idx, idx)]

    # Project 3×3 (H0, omega_b, omega_cdm) block → 2×2 (H0, omega_m)
    C3 = C_sub[:3, :3]
    A = np.array([[1., 0., 0.], [0., 1., 1.]])
    C2 = A @ C3 @ A.T                        # 2×2 cov for (H0, omega_m)

    # Assemble full 4×4 covariance for (H0, omega_m, w0, wa).
    # w0 and wa entries have zero off-diagonal terms in the covmat.
    C4 = np.zeros((4, 4))
    C4[:2, :2] = C2
    C4[2, 2] = C_sub[3, 3]                  # var(w0)
    C4[3, 3] = C_sub[4, 4]                  # var(wa)

    return np.linalg.inv(C4)


def _get_cmb_inv_cov():
    global _CMB_INV_COV
    if _CMB_INV_COV is None:
        _CMB_INV_COV = _build_cmb_inv_cov()
    return _CMB_INV_COV


def cmb_prior_chi2(Om: float, H0: float, w0: float, wa: float) -> float:
    """CMB chi2 using the full Planck covariance matrix.

    chi2_cmb = Delta^T InvCov Delta

    where Delta_i = param_i - param_BF_i and InvCov is the inverse of the
    projected PR4-lensing covariance for (H0, omega_m, w0, wa).
    omega_m = Om * (H0/100)^2 is computed from the sampled parameters.
    """
    import numpy as np
    inv_cov = _get_cmb_inv_cov()
    omega_m_bf = PLANCK_OM * (PLANCK_H0 / 100.0) ** 2
    omega_m    = Om * (H0 / 100.0) ** 2
    delta = np.array([
        H0  - PLANCK_H0,
        omega_m - omega_m_bf,
        w0  - PLANCK_CMB_W0,
        wa  - PLANCK_CMB_WA,
    ])
    return float(delta @ inv_cov @ delta)


# ---------------------------------------------------------------------------
# Vectorised version for batched numpy predict_fn
# ---------------------------------------------------------------------------

def cmb_prior_chi2_batch(Om, H0, w0, wa, inv_cov=None):
    """Vectorised CMB covariance prior for arrays of shape (N,).

    Returns an (N,) array of chi2 contributions.
    """
    import numpy as np
    if inv_cov is None:
        inv_cov = _get_cmb_inv_cov()
    omega_m_bf = PLANCK_OM * (PLANCK_H0 / 100.0) ** 2
    omega_m    = Om * (H0 / 100.0) ** 2
    # delta: (N, 4)
    delta = np.stack([
        H0  - PLANCK_H0,
        omega_m - omega_m_bf,
        w0  - PLANCK_CMB_W0,
        wa  - PLANCK_CMB_WA,
    ], axis=1)
    return np.einsum('ni,ij,nj->n', delta, inv_cov, delta)
