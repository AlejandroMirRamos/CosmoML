"""DES SN5YR loaders (2024 y 2025).

⚠️ Notas sobre las matrices (los archivos no contienen lo que su nombre sugiere):

**DES 2024** (`STAT+SYS_2024.txt.gz`): pese al nombre, la matriz es SÓLO la
covarianza SISTEMÁTICA. Para tener la STAT+SYS hay que sumar la varianza
estadística diagonal (`MUERR_FINAL²` del HD). Después se invierte.

**DES 2025** (`STAT+SYS_2025.npz`): `nsn` (tamaño) + `cov` (triángulo superior
empacado de la **inversa** de covarianza ya completa STAT+SYS — no hay que
sumar nada). Reconstruimos la matriz simétrica y la usamos directamente como
inv_cov.
"""
from dataclasses import dataclass
import gzip
import numpy as np
import pandas as pd

from ..config import DATA_DIR


@dataclass
class DesData:
    z: np.ndarray            # zHD (CMB-frame con corrección VPEC)
    z_hel: np.ndarray        # zHEL (heliocéntrica, sin VPEC)
    mu: np.ndarray
    inv_cov: np.ndarray
    sum_inv_cov: float       # cacheado para la marginalización analítica de M

    def __len__(self) -> int:
        return len(self.z)


def _load_sys_cov_2024(path) -> np.ndarray:
    """STAT+SYS_2024.txt.gz: primera línea es N, luego N·N flotantes con la
    covarianza SISTEMÁTICA (sin la diagonal estadística)."""
    with gzip.open(path, "rt") as f:
        n = int(f.readline().strip())
        flat = np.fromstring(f.read(), sep="\n")
    return flat.reshape((n, n))


def _load_inv_cov_2025(npz_path) -> np.ndarray:
    """STAT+SYS_2025.npz: `nsn`=N y `cov`=triángulo superior empacado de la
    INV_COV (no la covarianza). Reconstruimos la matriz simétrica."""
    d = np.load(npz_path)
    n = int(d["nsn"][0])
    inv = np.zeros((n, n), dtype=np.float64)
    inv[np.triu_indices(n)] = d["cov"]
    inv[np.tril_indices(n, -1)] = inv.T[np.tril_indices(n, -1)]
    return inv


def _read_des_hd(hd_path, want_muerr: bool = False):
    """Carga el Hubble Diagram. Soporta:
    - CSV con columnas estándar (DES 2024).
    - SNANA-style con cabecera VARNAMES y filas SN: (DES 2025).
    Devuelve (zHD, zHEL, MU) o (zHD, zHEL, MU, MUERR_FINAL/MUERR) si want_muerr."""
    # Intento 1: CSV directo (DES 2024)
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
    # Intento 2: SNANA-style (comment='#', whitespace, descartar VARNAMES:/SN:)
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
    """DES-SN5YR 2024 Hubble Diagram + cov sistemática (.txt.gz).

    Como el archivo `STAT+SYS_2024.txt.gz` es SÓLO la cov sistemática, sumamos
    `MUERR_FINAL²` a la diagonal para obtener STAT+SYS y luego invertimos.
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
    """DES 2025 HD (SNANA-style CSV) + INV_COV (.npz triángulo empacado).

    El archivo .npz contiene la INVERSA de covarianza STAT+SYS ya completa —
    no hay que sumar nada. Sólo desempacar el triángulo.
    """
    hd_path = DATA_DIR / "des" / "DES_2025_HD.csv"
    cov_path = DATA_DIR / "des" / "STAT+SYS_2025.npz"

    z_hd, z_hel, mu = _read_des_hd(hd_path)
    inv_cov = _load_inv_cov_2025(cov_path)
    return DesData(
        z=z_hd, z_hel=z_hel, mu=mu,
        inv_cov=inv_cov, sum_inv_cov=float(np.sum(inv_cov)),
    )
