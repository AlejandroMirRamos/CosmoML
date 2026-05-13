"""Joint SNe + BAO chi2."""
from .sne import chi2_sne
from .bao import chi2_bao
from ..data.pantheon import PantheonData
from ..data.desi_bao import DesiBaoData


def chi2_joint(
    sne_data: PantheonData,
    bao_data: DesiBaoData,
    *,
    Om: float, H0: float, w0: float = -1.0, wa: float = 0.0,
    rd: float | None = None,
    sne_kwargs: dict | None = None,
) -> float:
    """SNe + BAO chi2 with a shared cosmology (Flat w0waCDM).

    ``rd=None`` uses the Planck fiducial inside ``chi2_bao``.
    ``sne_kwargs`` lets the caller pass explicit M or alpha/beta to ``chi2_sne``.
    """
    sne_kwargs = sne_kwargs or {}
    chi2_s = chi2_sne(sne_data, model="Flatw0waCDM",
                      Om=Om, H0=H0, w0=w0, wa=wa, **sne_kwargs)
    bao_kwargs = dict(Om=Om, w0=w0, wa=wa, H0=H0)
    if rd is not None:
        bao_kwargs["rd"] = rd
    chi2_b = chi2_bao(bao_data, **bao_kwargs)
    return chi2_s + chi2_b
