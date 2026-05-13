"""DESI BAO DR2 loader (Gaussian likelihood)."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

from ..config import DATA_DIR


@dataclass
class DesiBaoData:
    z: np.ndarray            # redshift of each measurement
    val: np.ndarray          # observed value (DM/rs, DH/rs or DV/rs)
    type: np.ndarray         # 'DM_over_rs' | 'DH_over_rs' | 'DV_over_rs'
    unique_z: np.ndarray
    cov: np.ndarray
    inv_cov: np.ndarray

    def __len__(self) -> int:
        return len(self.z)


def load_desi_bao(z_max: float | None = None) -> DesiBaoData:
    """Load DESI BAO DR2.

    Parameters
    ----------
    z_max : float | None
        If given, drop measurements with ``z >= z_max`` and crop the covariance
        BEFORE inverting. Useful for scenarios that exclude the Lyman-alpha
        bin (e.g. z < 2).
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
