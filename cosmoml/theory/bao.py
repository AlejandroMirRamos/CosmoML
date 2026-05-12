"""χ² para BAO DESI DR2 (Gaussian likelihood).

Función teórica común a todos los escenarios BAO:
  - DM/rs, DH/rs y DV/rs en parametrización CPL (w₀, wₐ).
  - H₀ y rd tratados como parámetros (puedes fijarlos al fiducial Planck o dejarlos libres).
"""
from __future__ import annotations
import numpy as np
import scipy.integrate as integrate

from ..config import C_LIGHT, PLANCK_H0, PLANCK_RD
from ..data.desi_bao import DesiBaoData


_CHI2_BAD = 99999.0


def E_w0wa(z: np.ndarray | float, Om: float, w0: float = -1.0, wa: float = 0.0):
    """E(z) = H(z)/H0 para CPL w(z)=w0 + wa·z/(1+z)."""
    f_de = (1 + z) ** (3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z))
    return np.sqrt(Om * (1 + z) ** 3 + (1 - Om) * f_de)


def DM_w0wa(z: float, Om: float, w0: float = -1.0, wa: float = 0.0,
            H0: float = PLANCK_H0) -> float:
    """Distancia comóvil transversal D_M(z) = (c/H₀) ∫₀ᶻ dz'/E(z')."""
    if z == 0:
        return 0.0
    integral, _ = integrate.quad(lambda zp: 1.0 / E_w0wa(zp, Om, w0, wa), 0, z)
    return (C_LIGHT / H0) * integral


def bao_theory_vector(
    data: DesiBaoData,
    Om: float, w0: float = -1.0, wa: float = 0.0,
    H0: float = PLANCK_H0, rd: float = PLANCK_RD,
) -> np.ndarray:
    """Vector teórico ordenado igual que data.val (mezcla DM/rs, DH/rs, DV/rs)."""
    dm_map = {z: DM_w0wa(z, Om, w0, wa, H0=H0) for z in data.unique_z}
    out = np.empty(len(data))
    for i in range(len(data)):
        z = data.z[i]
        q = str(data.type[i]).upper()
        ez = E_w0wa(z, Om, w0, wa)
        dh = C_LIGHT / (H0 * ez)
        dm = dm_map[z]
        if "DM" in q:
            out[i] = dm / rd
        elif "DH" in q:
            out[i] = dh / rd
        elif "DV" in q:
            out[i] = (z * dm * dm * dh) ** (1 / 3) / rd
        else:
            out[i] = np.nan
    return out


def chi2_bao(
    data: DesiBaoData,
    Om: float, w0: float = -1.0, wa: float = 0.0,
    H0: float = PLANCK_H0, rd: float = PLANCK_RD,
) -> float:
    """χ² BAO DESI."""
    if not (0 < Om < 1) or H0 <= 0 or rd <= 0:
        return _CHI2_BAD
    try:
        th = bao_theory_vector(data, Om, w0, wa, H0=H0, rd=rd)
        diff = data.val - th
        return float(diff @ data.inv_cov @ diff)
    except Exception:
        return _CHI2_BAD
