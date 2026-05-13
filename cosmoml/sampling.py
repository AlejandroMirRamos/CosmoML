"""Chi-squared dataset builder for XGBoost training.

Generates a DataFrame of (parameters..., chi2) tuples by sampling the parameter
space, then evaluating chi2 in parallel. The sampling pattern combines:

    1. Slices: fix one or more dimensions to a constant (typically REF) and
       sample the rest uniformly. Provides dense coverage along 2D projections.
    2. Random box: sample all dimensions uniformly across a wider region.
    3. Anchor: optionally repeat the best-fit point n_anchor times.
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
    """Sample `n` rows per key. Tuple values → uniform draw; scalars → constant."""
    out = {}
    for k, v in spec.items():
        if isinstance(v, tuple):
            out[k] = rng.uniform(v[0], v[1], n)
        else:
            out[k] = np.full(n, float(v))
    return out


# Module-level so ProcessPoolExecutor can pickle it.
def _chi2_worker(task):
    func, kwargs = task
    return func(**kwargs)


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
    progress_every: int = 2000,
) -> pd.DataFrame:
    """Build a (params..., chi2) DataFrame via parallel chi2 evaluation.

    Parameters
    ----------
    chi2_fn : callable
        Receives kwargs matching `param_names` and returns chi2. Must be
        picklable (module-level function or notebook-cell closure over picklable
        data; lambdas are NOT picklable).
    param_names : list[str]
        Output column order.
    slices : list[dict]
        Each dict: ``{param: (low, high) | constant, '_n': N}``. Produces N
        rows per slice.
    random_box : dict
        ``{param: (low, high)}`` for the N-dim random cloud.
    n_random : int
        Cloud size.
    anchor : dict
        ``{param: constant}`` for the best-fit row, repeated `n_anchor` times.
    seed : int
        RNG seed.
    save_to : str | Path
        Optional CSV output path.
    progress_every : int
        Print progress (elapsed time, ETA, throughput) every N completed points.
        Set to 0 to disable.
    """
    rng = np.random.default_rng(seed)
    blocks: list[dict[str, np.ndarray]] = []

    for sp in slices or []:
        n = int(sp.pop("_n"))
        blocks.append(_sample_uniform(sp, n, rng))

    if random_box is not None and n_random > 0:
        blocks.append(_sample_uniform(random_box, n_random, rng))

    samples = {p: np.concatenate([b[p] for b in blocks]) for p in param_names}
    total = len(next(iter(samples.values())))

    n_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"Evaluating chi2 at {total} points across {n_cores} cores...")

    tasks = [(chi2_fn, {p: float(samples[p][i]) for p in param_names})
             for i in range(total)]

    chi2s = np.empty(total)
    t0 = time.time()
    last_report = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        # executor.map preserves submission order, so the index i matches tasks[i]
        for i, val in enumerate(executor.map(_chi2_worker, tasks, chunksize=100)):
            chi2s[i] = val
            done = i + 1
            if progress_every and done - last_report >= progress_every:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (total - done) / rate if rate > 0 else 0.0
                print(f"  {done:>7d}/{total} ({100*done/total:5.1f}%) | "
                      f"elapsed {elapsed:6.1f}s | ETA {eta:6.1f}s | "
                      f"{rate:6.0f} pts/s")
                last_report = done
    print(f"  done in {time.time() - t0:.1f}s")

    df = pd.DataFrame({**samples, "chi2": chi2s})

    if anchor is not None and n_anchor > 0:
        kwargs = {p: float(anchor[p]) for p in param_names}
        chi2_anchor = float(chi2_fn(**kwargs))
        print(f"  anchor {anchor} -> chi2={chi2_anchor:.3f} repeated {n_anchor}x")
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
        print(f"  saved: {save_to} ({len(df)} rows)")

    return df


def load_or_build(
    csv_path: str | Path,
    builder: Callable[[], pd.DataFrame],
    force: bool = False,
) -> pd.DataFrame:
    """Load `csv_path` if it exists (and not `force`); otherwise call `builder()`."""
    csv_path = Path(csv_path)
    if csv_path.exists() and not force:
        print(f"Loading cached dataset: {csv_path}")
        return pd.read_csv(csv_path)
    print(f"Building dataset (cache missing: {csv_path})")
    return builder()
