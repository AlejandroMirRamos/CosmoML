"""Gaussian priors (Planck 2018)."""
from .config import (
    PLANCK_H0, PLANCK_H0_ERR,
    PLANCK_OM, PLANCK_OM_ERR,
    PLANCK_RD, PLANCK_RD_ERR,
)


def gaussian_prior(value: float, mean: float, sigma: float) -> float:
    """Return ((value - mean) / sigma)**2, the chi2 contribution of a Gaussian prior."""
    return ((value - mean) / sigma) ** 2


PLANCK_PRIORS = {
    "H0": (PLANCK_H0, PLANCK_H0_ERR),
    "Om": (PLANCK_OM, PLANCK_OM_ERR),
    "rd": (PLANCK_RD, PLANCK_RD_ERR),
}


def planck_prior_chi2(*, H0=None, Om=None, rd=None) -> float:
    """Sum of Planck Gaussian priors for the parameters that are not None."""
    chi2 = 0.0
    if H0 is not None:
        chi2 += gaussian_prior(H0, *PLANCK_PRIORS["H0"])
    if Om is not None:
        chi2 += gaussian_prior(Om, *PLANCK_PRIORS["Om"])
    if rd is not None:
        chi2 += gaussian_prior(rd, *PLANCK_PRIORS["rd"])
    return chi2
