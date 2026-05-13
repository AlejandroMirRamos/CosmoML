"""DES SN5YR loaders (2024 and 2025).

The two releases distribute their covariance matrices differently:

  - DES 2024 (``STAT+SYS_2024.txt.gz``): SYSTEMATIC covariance only — the
    statistical variance ``MUERR_FINAL**2`` must be added to the diagonal
    before inverting.
  - DES 2025 (``STAT+SYS_2025.npz``): contains the packed upper triangle of
    the INVERSE of the full STAT+SYS covariance. Reconstruct the symmetric
    matrix and use directly as inv_cov.
"""
from dataclasses import dataclass
import gzip
import numpy as np
import pandas as pd

from ..config import DATA_DIR


@dataclass
class DesData:
    z: np.ndarray            # zHD (CMB-frame, VPEC corrected)
    z_hel: np.ndarray        # zHEL (heliocentric, no VPEC)
    mu: np.ndarray
    inv_cov: np.ndarray
    sum_inv_cov: float       # cached for the analytic M marginalization

    def __len__(self) -> int:
        return len(self.z)


def _load_sys_cov_2024(path) -> np.ndarray:
    """First line is N, then N*N floats forming the systematic covariance."""
    with gzip.open(path, "rt") as f:
        n = int(f.readline().strip())
        flat = np.fromstring(f.read(), sep="\n")
    return flat.reshape((n, n))


def _load_inv_cov_2025(npz_path) -> np.ndarray:
    """Unpack the upper-triangle inv_cov from the .npz and return the symmetric matrix."""
    d = np.load(npz_path)
    n = int(d["nsn"][0])
    inv = np.zeros((n, n), dtype=np.float64)
    inv[np.triu_indices(n)] = d["cov"]
    inv[np.tril_indices(n, -1)] = inv.T[np.tril_indices(n, -1)]
    return inv


def _read_des_hd(hd_path, want_muerr: bool = False):
    """Read a DES Hubble Diagram. Supports both CSV (2024) and SNANA-style (2025).

    Returns (zHD, zHEL, MU) or (..., MUERR_FINAL) when `want_muerr` is True.
    """
    try:
        df = pd.read_csv(hd_path)
        if "zHD" in df.columns:
            out = (df["zHD"].values, df["zHEL"].values, df["MU"].values)
            if want_muerr:
                err_col = "MUERR_FINAL" if "MUERR_FINAL" in df.columns else "MUERR"
                out = (*out, df[err_col].values)
            return out
    except Exception:
        pass
    # SNANA-style fallback (whitespace separated, with VARNAMES:/SN: prefix columns).
    df = pd.read_csv(hd_path, comment="#", sep=r"\s+")
    if df.columns[0] in ("VARNAMES:", "SN:"):
        df = df.rename(columns={df.columns[0]: "_marker"}).drop(columns="_marker")
    out = (
        df["zHD"].values.astype(float),
        df["zHEL"].values.astype(float),
        df["MU"].values.astype(float),
    )
    if want_muerr:
        err_col = "MUERR_FINAL" if "MUERR_FINAL" in df.columns else "MUERR"
        out = (*out, df[err_col].values.astype(float))
    return out


def load_des_2024() -> DesData:
    """Load DES-SN5YR 2024 Hubble Diagram and STAT+SYS covariance.

    The 2024 .txt.gz holds the systematic covariance only, so we add
    ``MUERR_FINAL**2`` to the diagonal to get the full STAT+SYS before inverting.
    """
    hd_path = DATA_DIR / "des" / "DES-SN5YR_2024_HD.csv"
    cov_path = DATA_DIR / "des" / "STAT+SYS_2024.txt.gz"

    z_hd, z_hel, mu, mu_err = _read_des_hd(hd_path, want_muerr=True)
    sys_cov = _load_sys_cov_2024(cov_path)
    full_cov = sys_cov.copy()
    np.fill_diagonal(full_cov, np.diag(full_cov) + mu_err ** 2)
    inv_cov = np.linalg.inv(full_cov)
    return DesData(
        z=z_hd, z_hel=z_hel, mu=mu,
        inv_cov=inv_cov, sum_inv_cov=float(np.sum(inv_cov)),
    )


def load_des_2025() -> DesData:
    """Load DES 2025 Hubble Diagram and inverse-covariance.

    The 2025 .npz already provides the inverse of the full STAT+SYS covariance
    as a packed upper triangle; we just unpack and use it directly.
    """
    hd_path = DATA_DIR / "des" / "DES_2025_HD.csv"
    cov_path = DATA_DIR / "des" / "STAT+SYS_2025.npz"

    z_hd, z_hel, mu = _read_des_hd(hd_path)
    inv_cov = _load_inv_cov_2025(cov_path)
    return DesData(
        z=z_hd, z_hel=z_hel, mu=mu,
        inv_cov=inv_cov, sum_inv_cov=float(np.sum(inv_cov)),
    )
