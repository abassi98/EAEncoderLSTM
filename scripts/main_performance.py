"""Evaluate ensemble and individual runs, reusing each basin's data and metrics."""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import pickle
import warnings

import numpy as np
import pandas as pd
from xarray import DataArray

from src.performance_batch import calculate_metrics
from src.datautils import load_forcings
from src.signatures import calculate_all_signatures


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="global, pub, global_linear or pub_linear")
    parser.add_argument("encoded_features")
    parser.add_argument("eval_period", choices=["val", "test"])
    parser.add_argument("--random_features", action="store_true")
    parser.add_argument("--camels-root", type=Path, default=Path("../Datasets/CAMELS-US"))
    parser.add_argument("--workers", type=int, default=1, help="Parallel basin workers (default: 1).")
    parser.add_argument(
        "--warmup-days", type=int, default=0,
        help=("Discard this many initial calendar days before dropping missing flow rows. "
              "Use 0 for no extra warmup. Omit to preserve legacy filtering: if any "
              "column contains NaNs, drop incomplete rows and then the first 365 rows."),
    )
    args = parser.parse_args()
    if args.workers < 1 or (args.warmup_days is not None and args.warmup_days < 0):
        parser.error("--workers must be positive and --warmup-days must be nonnegative")
    return args


def prepare_basin(frame, columns, warmup_days=None):
    """Keep the default evaluation record unchanged; never mutate the input."""
    if (not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.is_unique
            or not frame.index.is_monotonic_increasing or frame.index.hasnans):
        raise ValueError("Basin dates must be a unique, sorted DatetimeIndex.")
    if not frame.columns.is_unique:
        raise ValueError("Basin columns must be unique.")
    if warmup_days is None:
        # Retain the old rule for reproducibility, but make this surprising
        # evaluation-period change visible and provide an explicit alternative.
        missing = frame.isna().any(axis=1)
        if missing.any():
            warnings.warn("Legacy filtering drops incomplete rows and another 365 rows; "
                          "use --warmup-days 0 to disable the extra trim.", stacklevel=2)
            frame = frame.loc[~missing].iloc[365:]
    elif not frame.empty:
        start = frame.index[0] + pd.Timedelta(days=warmup_days)
        frame = frame.loc[start:, ["qobs", *columns]].dropna()
    data = frame[["qobs", *columns]].copy()
    data.attrs = {}
    if np.isinf(data.to_numpy(dtype=float)).any():
        raise ValueError("Infinite streamflow values are not supported.")
    return data


def compute_basin(basin, frame, columns, camels_root, forcing_path, warmup_days):
    data = prepare_basin(frame, columns, warmup_days)
    if data.empty:
        return basin, None, None
    metrics = calculate_metrics(data["qobs"].to_numpy(), data[columns].to_numpy())

    # Load once, retaining true dates even if streamflow has internal gaps.
    forcing, _ = load_forcings(camels_root, str(basin), "daymet", file_path=forcing_path)
    if not forcing.index.is_unique:
        raise ValueError(f"Basin {basin}: duplicate precipitation dates.")
    precipitation = forcing["prcp(mm/day)"].reindex(data.index)
    if not np.isfinite(precipitation.to_numpy()).all():
        raise ValueError(f"Basin {basin}: missing or nonfinite precipitation on evaluation dates.")
    xr_prcp = DataArray(precipitation.to_numpy(), dims=["date"], coords={"date": data.index})

    stats_row, signature_row = {}, {}
    for i, column in enumerate(columns):
        suffix = "" if column == "qsim" else f"_seed{column.removeprefix('qsim_')}"
        stats_row.update({f"{name}{suffix}": value for name, value in metrics.iloc[i].items()})
        flow = DataArray(data[column].to_numpy(), dims=["date"], coords={"date": data.index})
        signatures = calculate_all_signatures(flow, prcp=xr_prcp)
        signature_row.update({f"{name}{suffix}": value for name, value in signatures.items()})
    return basin, stats_row, signature_row


def _compute_task(task):
    return compute_basin(*task)


def _parallel_results(tasks, workers):
    """Bound queued basin copies rather than submit the entire dataset at once."""
    tasks = iter(tasks)
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        pending = deque()
        for _ in range(2 * workers):
            task = next(tasks, None)
            if task is not None:
                pending.append(pool.submit(_compute_task, task))
        while pending:
            yield pending.popleft().result()
            task = next(tasks, None)
            if task is not None:
                pending.append(pool.submit(_compute_task, task))


def compute_performance(ens_dict, camels_root=Path("../Datasets/CAMELS-US"),
                        warmup_days=None, workers=1):
    """Return statistics/signature tables in deterministic basin and seed order."""
    if not ens_dict:
        raise ValueError("No basins found in ensemble results.")
    if workers < 1 or (warmup_days is not None and warmup_days < 0):
        raise ValueError("workers must be positive and warmup_days nonnegative.")
    first = next(iter(ens_dict.values()))
    runs = sorted(col for col in first.columns if col.startswith("qsim_"))
    columns = ["qsim", *runs]
    for basin, frame in ens_dict.items():
        if sorted(col for col in frame.columns if col.startswith("qsim_")) != runs:
            raise ValueError(f"Basin {basin}: run seeds must match across basins.")

    camels_root = Path(camels_root)
    forcing_dir = camels_root / "basin_mean_forcing/daymet"
    forcing_paths = {}
    for path in sorted(forcing_dir.glob("**/*_forcing_leap.txt")):
        forcing_paths.setdefault(path.name[:8], path)
    missing = [str(basin) for basin in ens_dict if str(basin) not in forcing_paths]
    if missing:
        raise FileNotFoundError(f"Missing daymet forcing files in {forcing_dir}: {missing[:5]}")

    tasks = ((basin, frame, columns, camels_root, forcing_paths[str(basin)], warmup_days)
             for basin, frame in ens_dict.items())
    results = map(_compute_task, tasks) if workers == 1 else _parallel_results(tasks, workers)
    stats, signatures = {}, {}
    for basin, stats_row, signature_row in results:
        if stats_row is None:
            warnings.warn(f"Skipping basin {basin}: no evaluation rows after filtering.")
            continue
        stats[basin] = stats_row
        signatures[basin] = signature_row
        print(f"{basin} ({len(stats)} of {len(ens_dict)}) --- NSE: {stats_row['nse']} "
              f"--- mNSE: {stats_row['abs_nse']} --- FHV: {stats_row['FHV']}")
    if not stats:
        raise ValueError("No basins have evaluation rows after filtering.")
    stats = pd.DataFrame.from_dict(stats, orient="index").rename_axis("basin")
    signatures = pd.DataFrame.from_dict(signatures, orient="index").rename_axis("basin")
    print("Mean NSE:", stats["nse"].mean())
    print("Median NSE:", stats["nse"].median())
    print("Num Failures:", (stats["nse"] < 0).sum())
    return stats, signatures


def main():
    args = get_args()
    suffix = "_random_features" if args.random_features else ""
    experiment = args.experiment
    stem = f"{experiment}_es{args.encoded_features}"
    input_path = Path("analysis/results_data") / f"{stem}_{args.eval_period}{suffix}.pkl"
    with input_path.open("rb") as handle:
        ens_dict = pickle.load(handle)
    stats, signatures = compute_performance(ens_dict, args.camels_root, args.warmup_days, args.workers)
    for directory, table in (("stats", stats), ("signatures", signatures)):
        output_path = Path("analysis") / directory / args.eval_period / f"{stem}{suffix}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
