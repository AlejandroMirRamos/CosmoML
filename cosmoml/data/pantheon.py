"""Pantheon+SH0ES loader.

Devuelve un objeto con z_hd, z_hel, mb, is_calib, ceph_dist, cov, inv_cov, sum_inv_cov,
ya con la mask estándar (zHD>0.01 o calibrador) aplicada.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

from ..config import DATA_DIR


@dataclass
class PantheonData:
    z_hd: np.ndarray
    z_hel: np.ndarray
    mb: np.ndarray
    is_calib: np.ndarray
    ceph_dist: np.ndarray
    cov: np.ndarray
    inv_cov: np.ndarray
    sum_inv_cov: float
    x1: np.ndarray | None = None
    c: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.z_hd)


def load_pantheon_plus(
    apply_mask: bool = True,
    z_min: float = 0.01,
    keep_calibrators: bool = True,
    include_x1c: bool = False,
) -> PantheonData:
    """Carga Pantheon+SH0ES.

    Parameters
    ----------
    apply_mask : bool
        Si True aplica una mask sobre las filas. El comportamiento depende de
        `keep_calibrators`.
    z_min : float
        Umbral para el corte de redshift bajo.
    keep_calibrators : bool
        - True (default): mask = (zHD > z_min) | IS_CALIBRATOR. Los calibradores
          Cefeidas se mantienen aunque tengan z<z_min — escenario "con SH0ES".
        - False: mask = (zHD > z_min) sólo. Los calibradores se descartan junto
          con el resto. Útil para los escenarios "sin SH0ES" (e.g. corte
          agresivo z>0.25, sin Cefeidas).
    include_x1c : bool
        Si True devuelve también x1 y c (necesario para análisis α/β).
    """
    dat_path = DATA_DIR / "pantheon" / "Pantheon+SH0ES.dat"
    cov_path = DATA_DIR / "pantheon" / "Pantheon+SH0ES_STAT+SYS.cov"

    df = pd.read_csv(dat_path, sep=r"\s+")
    z_hd = df["zHD"].values
    z_hel = df["zHEL"].values
    mb = df["m_b_corr"].values
    is_calib = df["IS_CALIBRATOR"].values.astype(bool)
    ceph_dist = df["CEPH_DIST"].values

    cov = np.genfromtxt(cov_path, skip_header=1).reshape((len(z_hd), len(z_hd)))

    x1 = df["x1"].values if (include_x1c and "x1" in df.columns) else None
    c = df["c"].values if (include_x1c and "c" in df.columns) else None

    if apply_mask:
        if keep_calibrators:
            mask = (z_hd > z_min) | is_calib
        else:
            mask = (z_hd > z_min)
        z_hd, z_hel, mb = z_hd[mask], z_hel[mask], mb[mask]
        is_calib, ceph_dist = is_calib[mask], ceph_dist[mask]
        cov = cov[mask, :][:, mask]
        if x1 is not None:
            x1 = x1[mask]
        if c is not None:
            c = c[mask]

    inv_cov = np.linalg.inv(cov)

    return PantheonData(
        z_hd=z_hd, z_hel=z_hel, mb=mb,
        is_calib=is_calib, ceph_dist=ceph_dist,
        cov=cov, inv_cov=inv_cov, sum_inv_cov=float(np.sum(inv_cov)),
        x1=x1, c=c,
    )
