"""Type Ia supernovae chi2 with Cepheid calibration (Pantheon+SH0ES).

Supports:
  - FlatLambdaCDM, LambdaCDM, FlatwCDM, Flatw0waCDM (via astropy.cosmology).
  - Absolute magnitude M analytically marginalized (default) or passed explicitly.
  - Optional SALT2 alpha/beta correction (mb_corr += (alpha-alpha0)*x1 - (beta-beta0)*c).
"""
from __future__ import annotations
from typing import Literal

import numpy as np
from astropy.cosmology import FlatLambdaCDM, FlatwCDM, Flatw0waCDM, LambdaCDM

from ..data.pantheon import PantheonData
from ..data.des import DesData


_CHI2_BAD = 99999.9
_FID_ALPHA = 0.14
_FID_BETA = 3.1


def _build_cosmo(model: str, H0: float, Om: float,
                 w0: float = -1.0, wa: float = 0.0,
                 Ode: float | None = None):
    if model == "FlatLambdaCDM":
        return FlatLambdaCDM(H0=H0, Om0=Om)
    if model == "LambdaCDM":
        if Ode is None:
            raise ValueError("LambdaCDM requires Ode")
        return LambdaCDM(H0=H0, Om0=Om, Ode0=Ode)
    if model == "FlatwCDM":
        return FlatwCDM(H0=H0, Om0=Om, w0=w0)
    if model == "Flatw0waCDM":
        return Flatw0waCDM(H0=H0, Om0=Om, w0=w0, wa=wa)
    raise ValueError(f"Unknown model: {model}")


def mu_theory_sne(
    data: PantheonData,
    model: str,
    Om: float, H0: float,
    w0: float = -1.0, wa: float = 0.0,
    Ode: float | None = None,
    *,
    use_cepheid_calibrators: bool = True,
    use_zhel_correction: bool = True,
) -> np.ndarray:
    """Theoretical distance modulus per SN.

    Parameters
    ----------
    use_cepheid_calibrators : bool
        If True (default), calibrators receive ``CEPH_DIST`` and the rest get
        the cosmological model.
        If False, ALL points receive the model (no SH0ES).
    use_zhel_correction : bool
        If True (default), apply ``dl * (1 + z_hel) / (1 + z_hd)``.

    Returns ``None`` if the input is unphysical or astropy produces non-finite
    luminosity distances (caller treats this as a chi2 failure sentinel).
    """
    if Om <= 0 or H0 <= 0:
        return None
    mu = np.zeros_like(data.mb)
    if use_cepheid_calibrators:
        mu[data.is_calib] = data.ceph_dist[data.is_calib]
        flow = ~data.is_calib
    else:
        flow = np.ones_like(data.is_calib, dtype=bool)
    if np.any(flow):
        cosmo = _build_cosmo(model, H0, Om, w0, wa, Ode)
        dl = cosmo.luminosity_distance(data.z_hd[flow]).value
        # Non-physical cosmologies (e.g. LambdaCDM with Om~0 and Ode>>1) can
        # produce NaN/inf without raising; treat as invalid.
        if not np.all(np.isfinite(dl)):
            return None
        dl[dl <= 0] = 1e-10
        if use_zhel_correction:
            dl = dl * (1 + data.z_hel[flow]) / (1 + data.z_hd[flow])
        mu[flow] = 5 * np.log10(dl) + 25
    return mu


def chi2_sne(
    data: PantheonData,
    model: str = "FlatLambdaCDM",
    *,
    Om: float, H0: float,
    w0: float = -1.0, wa: float = 0.0, Ode: float | None = None,
    M: float | Literal["marginalize"] = "marginalize",
    alpha: float | None = None,
    beta: float | None = None,
    fid_alpha: float = _FID_ALPHA,
    fid_beta: float = _FID_BETA,
    use_cepheid_calibrators: bool = True,
    use_zhel_correction: bool = True,
) -> float:
    """Pantheon+ chi2 with full STAT+SYS covariance.

    Parameters
    ----------
    M : float | "marginalize"
        - "marginalize" (default): analytic marginalization,
          ``m_best = sum(C^-1 . delta) / sum(C^-1)``.
        - float: use the given M (e.g. for the M-free scenario).
    alpha, beta : float | None
        If both are given, apply the SALT2 correction. Requires
        ``include_x1c=True`` when loading Pantheon.
    use_cepheid_calibrators, use_zhel_correction
        See ``mu_theory_sne``.
    """
    try:
        mb = data.mb.copy()
        if alpha is not None and beta is not None:
            if data.x1 is None or data.c is None:
                raise ValueError("alpha/beta require include_x1c=True when loading Pantheon")
            mb = mb + (alpha - fid_alpha) * data.x1 - (beta - fid_beta) * data.c

        mu = mu_theory_sne(
            data, model, Om, H0, w0=w0, wa=wa, Ode=Ode,
            use_cepheid_calibrators=use_cepheid_calibrators,
            use_zhel_correction=use_zhel_correction,
        )
        if mu is None:
            return _CHI2_BAD

        delta = mb - mu
        if M == "marginalize":
            m_best = float(np.sum(data.inv_cov @ delta) / data.sum_inv_cov)
            diff = delta - m_best
        else:
            diff = delta - float(M)
        return float(diff @ data.inv_cov @ diff)
    except Exception:
        return _CHI2_BAD


def chi2_sne_des(
    data: DesData,
    model: str = "FlatwCDM",
    *,
    Om: float, H0: float = 70.0,
    w0: float = -1.0, wa: float = 0.0, Ode: float | None = None,
) -> float:
    """DES SN5YR chi2 with analytic marginalization of the absolute magnitude offset.

    Differences with ``chi2_sne``:
      - DES has no Cepheid calibrators (no external anchor in the HD).
      - The HD already provides ``mu`` directly, so there is no separate M term;
        the marginalization acts on the offset between model and observed mu.
      - Analytic marginalization:
            chi2 = sum(d . C^-1 . d) - (sum(C^-1 . d))^2 / sum(C^-1)
                   + log(sum(C^-1) / (2 pi))

    Only flat cosmologies are supported (DES paper convention).
    """
    if Om <= 0 or H0 <= 0:
        return _CHI2_BAD
    try:
        cosmo = _build_cosmo(model, H0, Om, w0=w0, wa=wa, Ode=Ode)
        da = cosmo.angular_diameter_distance(data.z).value
        if np.any(da <= 0) or not np.all(np.isfinite(da)):
            return _CHI2_BAD
        # mu = 5 * log10[(1 + zHD) * (1 + zHEL) * D_A] + 25
        mu_model = 5.0 * np.log10((1.0 + data.z) * (1.0 + data.z_hel) * da) + 25.0
        delta = mu_model - data.mu
        chit2 = float(delta @ data.inv_cov @ delta)
        B = float(np.sum(data.inv_cov @ delta))
        C = data.sum_inv_cov
        return chit2 - (B * B / C) + float(np.log(C / (2.0 * np.pi)))
    except Exception:
        return _CHI2_BAD
