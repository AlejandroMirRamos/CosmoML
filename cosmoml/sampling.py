"""Generador de datasets χ² para entrenar XGBoost.

Patrón estándar usado en todos los escenarios:
  1. Rodajas: una dimensión fija + el resto aleatorio (varios planos).
  2. Nube random: todas las dimensiones aleatorias en rangos amplios.
  3. Anclaje: el best-fit repetido N veces para "dopar" el modelo.
"""
from __future__ import annotations
from collections.abc import Callable
from pathlib import Path
import time
import numpy as np
import pandas as pd
import concurrent.futures
import multiprocessing

def _sample_uniform(spec: dict[str, tuple[float, float] | float], n: int,
                    rng: np.random.Generator) -> dict[str, np.ndarray]:
    """spec: {param: (low, high) | valor_fijo}. Devuelve dict de arrays len=n."""
    out = {}
    for k, v in spec.items():
        if isinstance(v, tuple):
            out[k] = rng.uniform(v[0], v[1], n)
        else:
            out[k] = np.full(n, float(v))
    return out


# -------------------------------------------------------------------
# FUNCIÓN AUXILIAR (Debe ir FUERA para que Python pueda paralelizarla)
# -------------------------------------------------------------------
def _chi2_worker(task):
    """Desempaqueta la función y los argumentos y los evalúa."""
    func, kwargs = task
    return func(**kwargs)

# -------------------------------------------------------------------
# TU FUNCIÓN PRINCIPAL ACTUALIZADA
# -------------------------------------------------------------------
def build_chi2_dataset(
    chi2_fn: Callable[..., float],
    param_names: list[str],
    *,
    slices: list[dict] | None = None,
    random_box: dict | None = None,
    n_random: int = 30000,
    anchor: dict | None = None,
    n_anchor: int = 0,
    seed: int = 0,
    save_to: str | Path | None = None,
    progress_every: int = 5000, # Nota: en paralelo no usaremos esto, será todo de golpe
) -> pd.DataFrame:
    """Construye un DataFrame (params..., chi2).

    Parameters
    ----------
    chi2_fn : callable
        Función que recibe **kwargs con los param_names y devuelve χ².
    param_names : list[str]
        Orden de columnas en el DataFrame de salida.
    slices : list[dict]
        Cada dict tiene {param: (low, high) | valor_fijo, '_n': N}. Genera N puntos.
    random_box : dict
        {param: (low, high)} para la nube aleatoria N-dim.
    n_random : int
        Tamaño de la nube aleatoria.
    anchor : dict
        {param: valor_fijo} del best-fit a repetir n_anchor veces.
    n_anchor : int
        Veces que se repite el ancla (0 lo desactiva).
    save_to : str | Path
        Si se pasa, escribe el CSV resultante.
    """
    rng = np.random.default_rng(seed)
    blocks: list[dict[str, np.ndarray]] = []

    # 1. Rodajas
    for sp in slices or []:
        n = int(sp.pop("_n"))
        blocks.append(_sample_uniform(sp, n, rng))

    # 2. Nube random
    if random_box is not None and n_random > 0:
        blocks.append(_sample_uniform(random_box, n_random, rng))

    # Concatenar
    samples = {p: np.concatenate([b[p] for b in blocks]) for p in param_names}
    total = len(next(iter(samples.values())))
    print(f"Calculando χ² para {total} puntos (en paralelo)...")

    # --- INICIO DEL BLOQUE PARALELO ---
    t0 = time.time()
    
    # Preparamos las tareas: una tupla con (tu_funcion, diccionario_de_parametros)
    tasks = [(chi2_fn, {p: float(samples[p][i]) for p in param_names}) for i in range(total)]
    
    # Dejamos 1 núcleo libre para no colapsar el ordenador
    n_cores = max(1, multiprocessing.cpu_count() - 1)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        # chunksize=100 envía las integrales en paquetes de 100 a cada núcleo
        results = list(executor.map(_chi2_worker, tasks, chunksize=100))
        
    chi2s = np.array(results)
    print(f"  → terminado en {time.time()-t0:.1f}s")
    # --- FIN DEL BLOQUE PARALELO ---

    df = pd.DataFrame({**samples, "chi2": chi2s})

    # 3. Anclaje (Se mantiene exactamente igual que lo tenías)
    if anchor is not None and n_anchor > 0:
        kwargs = {p: float(anchor[p]) for p in param_names}
        chi2_anchor = float(chi2_fn(**kwargs))
        print(f"  ancla {anchor} → χ²={chi2_anchor:.3f} repetida {n_anchor}×")
        anchor_rows = pd.DataFrame({
            **{p: [anchor[p]] * n_anchor for p in param_names},
            "chi2": [chi2_anchor] * n_anchor,
        })
        df = pd.concat([df, anchor_rows], ignore_index=True)

    df = df[param_names + ["chi2"]]

    if save_to:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_to, index=False)
        print(f"  guardado: {save_to}  ({len(df)} filas)")

    return df

def load_or_build(
    csv_path: str | Path,
    builder: Callable[[], pd.DataFrame],
    force: bool = False,
) -> pd.DataFrame:
    """Si existe el CSV lo carga; si no (o force=True) lo construye."""
    csv_path = Path(csv_path)
    if csv_path.exists() and not force:
        print(f"Cargando dataset existente: {csv_path}")
        return pd.read_csv(csv_path)
    print(f"Generando dataset (no existe {csv_path})...")
    df = builder()
    return df
