"""DESI BAO DR2 loader (Gaussian likelihood)."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

from ..config import DATA_DIR


@dataclass
class DesiBaoData:
    z: np.ndarray            # redshift de cada medida
    val: np.ndarray          # valor observado (DM/rs, DH/rs o DV/rs)
    type: np.ndarray         # 'DM_over_rs' | 'DH_over_rs' | 'DV_over_rs'
    unique_z: np.ndarray
    cov: np.ndarray
    inv_cov: np.ndarray

    def __len__(self) -> int:
        return len(self.z)


def load_desi_bao(z_max: float | None = None) -> DesiBaoData:
    """Carga DESI BAO DR2.

    Parameters
    ----------
    z_max : float | None
        Si se pasa, descarta medidas con z >= z_max y recorta la covarianza
        ANTES de invertirla (importante para mantener la coherencia entre
        índices de medidas y covarianza). Útil para escenarios z<2 que
        excluyen Lyman-α.
    """
    mean_path = DATA_DIR / "desi_bao" / "desi_gaussian_bao_ALL_GCcomb_mean.txt"
    cov_path = DATA_DIR / "desi_bao" / "desi_gaussian_bao_ALL_GCcomb_cov.txt"

    df = pd.read_csv(
        mean_path, sep=r"\s+", comment="#", header=None,
        names=["z", "val", "type"],
    )
    cov = np.loadtxt(cov_path)

    if z_max is not None:
        mask = df["z"].values < z_max
        df = df[mask].reset_index(drop=True)
        cov = cov[mask, :][:, mask]

    return DesiBaoData(
        z=df["z"].values,
        val=df["val"].values,
        type=df["type"].values,
        unique_z=np.unique(df["z"].values),
        cov=cov,
        inv_cov=np.linalg.inv(cov),
    )
