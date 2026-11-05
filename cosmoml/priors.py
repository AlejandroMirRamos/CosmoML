"""CMB Gaussian priors using the full Planck PR4 covariance matrix (CPL)."""
from pathlib import Path

from .config import (
    CMB_REF_H0, CMB_REF_OMEGA_M,
    CMB_REF_W0, CMB_REF_WA,
)

_COVMAT_PATH = Path(__file__).resolve().parent / 'data' / 'PR4_lensing_CPL.covmat'

# Cached inverse-covariance (lazy singleton).
_CMB_INV_COV = None


def _build_cmb_inv_cov():
    """Parse PR4_lensing_CPL.covmat and return InvCov for (H0, omega_m, w0, wa).

    The covmat contains 25 parameters.  We extract the 5×5 block for
    (H0, omega_b, omega_cdm, w0_fld, wa_fld) at column/row indices [2,3,4,5,6]
    and project the two physical densities onto their sum with the linear map

        y = A x,  x = (H0, omega_b, omega_cdm, w0, wa)
                  y = (H0, omega_m,            w0, wa),   omega_m = omega_b+omega_cdm

    so that Sigma_y = A Sigma_x A^T keeps EVERY cross-correlation.  This matters
    for the w0waCDM PR4+lensing chain, where w0 and wa are strongly correlated
    with each other (rho ~ -0.75) and with H0 (rho ~ -0.83): the previous code
    zeroed those off-diagonals, which is only valid for a LambdaCDM covmat with
    fixed w0, wa.  The 4×4 inverse is the CMB precision matrix of the prior.
    """
    import numpy as np

    with open(_COVMAT_PATH) as f:
        f.readline()                          # skip header
        rows = [list(map(float, line.split())) for line in f if line.strip()]
    C = np.array(rows)

    # 5×5 submatrix: H0(2), omega_b(3), omega_cdm(4), w0(5), wa(6)
    idx = [2, 3, 4, 5, 6]
    C5 = C[np.ix_(idx, idx)]

    # Full projection to (H0, omega_m, w0, wa): omega_m = omega_b + omega_cdm,
    # all other rows passed through unchanged. All correlations are preserved.
    A = np.array([
        [1., 0., 0., 0., 0.],   # H0
        [0., 1., 1., 0., 0.],   # omega_m = omega_b + omega_cdm
        [0., 0., 0., 1., 0.],   # w0
        [0., 0., 0., 0., 1.],   # wa
    ])
    C4 = A @ C5 @ A.T                        # 4×4 cov for (H0, omega_m, w0, wa)

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
    omega_m = Om * (H0 / 100.0) ** 2
    delta = np.array([
        H0      - CMB_REF_H0,
        omega_m - CMB_REF_OMEGA_M,
        w0      - CMB_REF_W0,
        wa      - CMB_REF_WA,
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
    omega_m = Om * (H0 / 100.0) ** 2
    # delta: (N, 4)
    delta = np.stack([
        H0      - CMB_REF_H0,
        omega_m - CMB_REF_OMEGA_M,
        w0      - CMB_REF_W0,
        wa      - CMB_REF_WA,
    ], axis=1)
    return np.einsum('ni,ij,nj->n', delta, inv_cov, delta)
