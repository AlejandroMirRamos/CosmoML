"""χ² para supernovas tipo Ia con calibración Cepheid (Pantheon+SH0ES).

Soporta:
- FlatLambdaCDM, FlatwCDM, Flatw0waCDM (vía astropy.cosmology)
- M absoluta marginalizada analíticamente (por defecto) o pasada explícita
- Corrección α/β opcional (mb_corr += (α-α₀) x1 - (β-β₀) c)

Cálculo riguroso: una llamada a astropy por punto, sin cachés cruzados —
la comparación ML vs teoría en los notebooks debe ser justa.
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
    if model == "LambdaCDM":  # no plano: Ode libre
        if Ode is None:
            raise ValueError("LambdaCDM requiere Ode")
        return LambdaCDM(H0=H0, Om0=Om, Ode0=Ode)
    if model == "FlatwCDM":
        return FlatwCDM(H0=H0, Om0=Om, w0=w0)
    if model == "Flatw0waCDM":
        return Flatw0waCDM(H0=H0, Om0=Om, w0=w0, wa=wa)
    raise ValueError(f"Modelo no reconocido: {model}")


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
    """μ teórica por SNe.

    Parameters
    ----------
    use_cepheid_calibrators : bool
        True (default): los calibradores reciben CEPH_DIST y el resto recibe el modelo.
        False: TODOS los puntos reciben μ del modelo (equivale al modo "Pantheon
        plano sin SH0ES" usado en el script original FlatCDM_generator.py).
    use_zhel_correction : bool
        True (default): se aplica dl·(1+z_hel)/(1+z_hd). False: se usa dl directamente.
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
        # Cosmologías no físicas (e.g. LambdaCDM con Ω_m≈0 y Ω_Λ alto) producen
        # NaN/inf en astropy sin lanzar excepción. Marca como inválido.
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
    """χ² SNe Pantheon+ con cov completa.

    M
        - "marginalize" (default): se marginaliza analíticamente
          (m_best = sum(C⁻¹ Δ) / sum(C⁻¹)).
        - float: se usa ese valor explícito (escenario M variable o calibrado).
    alpha, beta
        Si se pasan ambos, se aplica la corrección α/β (requiere include_x1c=True).
    use_cepheid_calibrators, use_zhel_correction
        Ver mu_theory_sne. Para reproducir el FlatCDM "simple" del script original
        usa False en ambos junto con M=-19.23904 y carga Pantheon con apply_mask=False.
    """
    try:
        mb = data.mb.copy()
        if alpha is not None and beta is not None:
            if data.x1 is None or data.c is None:
                raise ValueError("alpha/beta requieren cargar Pantheon con include_x1c=True")
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
    """χ² SNe DES SN5YR con marginalización analítica de la M absoluta.

    Diferencias con `chi2_sne`:
    - DES no incluye Cefeidas (no hay calibradores externos en el Hubble Diagram).
    - El Hubble Diagram ya contiene `μ` directamente (no `m_b`), así que no hay
      que añadir el término M; la marginalización se hace SOBRE el offset entre
      μ_modelo y μ_observado.
    - Marginalización analítica idéntica al script DES oficial:
        χ² = Σ Δ·C⁻¹·Δ  -  (Σ C⁻¹·Δ)² / (Σ C⁻¹)  +  log(Σ C⁻¹ / 2π)
      (el último término es el Jacobiano de la marginalización; constante).

    Modelo soporta `FlatLambdaCDM`, `FlatwCDM`, `Flatw0waCDM` (no LambdaCDM
    no-plana — DES sólo se usa con cosmologías planas en el paper).
    """
    if Om <= 0 or H0 <= 0:
        return _CHI2_BAD
    try:
        cosmo = _build_cosmo(model, H0, Om, w0=w0, wa=wa, Ode=Ode)
        da = cosmo.angular_diameter_distance(data.z).value
        if np.any(da <= 0) or not np.all(np.isfinite(da)):
            return _CHI2_BAD
        # μ = 5·log10[(1+zHD)·(1+zHEL)·DA] + 25  (relación distancia-luminosidad)
        mu_model = 5.0 * np.log10((1.0 + data.z) * (1.0 + data.z_hel) * da) + 25.0
        delta = mu_model - data.mu
        chit2 = float(delta @ data.inv_cov @ delta)
        B = float(np.sum(data.inv_cov @ delta))
        C = data.sum_inv_cov
        return chit2 - (B * B / C) + float(np.log(C / (2.0 * np.pi)))
    except Exception:
        return _CHI2_BAD

