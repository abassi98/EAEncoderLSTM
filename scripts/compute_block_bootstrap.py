"""Paired NSE bootstrap using hydrological years or sliding 365-day blocks.

Separate model runs reuse one saved sequence of sampled blocks. Element i of
every output array therefore evaluates the same dates, including repetitions.
"""

import argparse
import hashlib
import json
import os
import pickle
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("encoded_features")
    parser.add_argument("eval_period", choices=["val", "test"])
    parser.add_argument("--random_features", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-bootstrap", "--n_bootstrap", type=int, default=5000)
    parser.add_argument(
        "--block-method", "--block_method",
        choices=["hydrological_year", "sliding"], default="sliding",
        help="Complete October--September years, or sliding 365-day blocks (default).",
    )
    parser.add_argument(
        "--samples-file", "--samples_file", type=Path,
        help="Shared block-sampling plan; created on the first run and reused thereafter.",
    )
    args = parser.parse_args()
    if args.seed < 0 or args.n_bootstrap < 1:
        parser.error("--seed must be nonnegative and --n-bootstrap must be positive")
    return args


def prepare_data(ens_dict):
    """Keep the daily calendar intact and validate a model-independent NSE mask."""
    if not ens_dict:
        raise ValueError("The results file contains no catchments.")
    dates = None
    data = {}
    observation_hashes = {}
    for basin in sorted(ens_dict, key=str):
        df = ens_dict[basin].sort_index()
        if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
            raise ValueError(f"Basin {basin}: expected a nonempty DatetimeIndex.")
        if df.index.tz is not None or df.index.has_duplicates or df.index.hasnans:
            raise ValueError(f"Basin {basin}: dates must be unique, timezone-naive and valid.")
        index = df.index.astype("datetime64[ns]")
        calendar = pd.date_range(index[0], index[-1], freq="D").astype("datetime64[ns]")
        if not index.equals(calendar) or not index.equals(index.normalize()):
            raise ValueError(f"Basin {basin}: expected consecutive daily dates; do not drop missing days.")
        if dates is None:
            dates = calendar
        elif not dates.equals(calendar):
            raise ValueError(f"Basin {basin}: all catchments must share the same daily calendar.")
        if not {"qobs", "qsim"}.issubset(df.columns):
            raise ValueError(f"Basin {basin}: qobs and ensemble-mean qsim columns are required.")
        obs = df["qobs"].to_numpy(dtype=np.float64, copy=True)
        sim = df["qsim"].to_numpy(dtype=np.float64, copy=True)
        valid = np.isfinite(obs) & (obs >= 0)
        if not valid.any():
            raise ValueError(f"Basin {basin}: no valid observed discharge.")
        if np.any(valid & (~np.isfinite(sim) | (sim < 0))):
            raise ValueError(
                f"Basin {basin}: missing, infinite or negative qsim on valid observation dates. "
                "Resolve prediction gaps across models before paired bootstrapping."
            )
        # All models must use the same observations, hence the same NSE denominator.
        # Different missing-value sentinels are equivalent; retain their calendar rows.
        obs[~valid] = np.nan
        observation_hashes[basin] = hashlib.sha256(obs.astype("<f8").tobytes()).hexdigest()
        data[basin] = (obs, sim)
    return dates, data, observation_hashes


def create_samples(dates, observation_hashes, n_bootstrap=5000, seed=0,
                   block_method="hydrological_year"):
    """Draw shared blocks with replacement; sliding samples retain the record length."""
    if n_bootstrap < 1 or seed < 0:
        raise ValueError("n_bootstrap must be positive and seed must be nonnegative.")
    rng = np.random.default_rng(seed)
    samples = {
        "version": 1,
        "dates": dates,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "observation_hashes": observation_hashes,
    }
    if block_method == "sliding":
        block_length = 365
        if len(dates) <= block_length:
            raise ValueError("Sliding blocks require more than 365 days for resampling variability.")
        n_blocks = (len(dates) + block_length - 1) // block_length
        block_lengths = np.full(n_blocks, block_length, dtype=int)
        block_lengths[-1] = len(dates) - block_length * (n_blocks - 1)
        samples.update({
            "method": "sliding_365_days",
            "block_length": block_length,
            "block_lengths": block_lengths,
            "block_starts": rng.integers(
                0, len(dates) - block_length + 1, size=(n_bootstrap, n_blocks),
            ),
        })
        return samples
    if block_method != "hydrological_year":
        raise ValueError(f"Unknown block method: {block_method}")
    if (dates[0].month, dates[0].day) != (10, 1) or (dates[-1].month, dates[-1].day) != (9, 30):
        raise ValueError("The evaluation calendar must contain complete October 1--September 30 years.")
    boundaries = pd.DatetimeIndex([
        pd.Timestamp(year=year, month=10, day=1)
        for year in range(dates[0].year, dates[-1].year + 1)
    ])
    year_offsets = dates.searchsorted(boundaries)
    n_years = len(year_offsets) - 1
    if n_years < 2:
        raise ValueError("At least two complete hydrological years are needed for resampling variability.")
    samples.update({
        "method": "october_september_years",
        "year_offsets": year_offsets,
        "sampled_years": rng.integers(0, n_years, size=(n_bootstrap, n_years)),
    })
    return samples


def load_or_create_samples(path, dates, observation_hashes, n_bootstrap=5000, seed=0,
                           block_method="hydrological_year"):
    """Publish the shared plan atomically, including when cluster jobs start together."""
    path = Path(path)
    expected = create_samples(dates, observation_hashes, n_bootstrap, seed, block_method)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            pickle.dump(expected, handle, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            # Unlike replacing the destination, this never overwrites another job's plan.
            os.link(temporary, path)
        except FileExistsError:
            pass
        finally:
            temporary.unlink()
    with path.open("rb") as handle:
        samples = pickle.load(handle)
    for key in ("version", "method", "n_bootstrap", "seed"):
        if samples.get(key) != expected[key]:
            raise ValueError(f"Shared samples {path}: incompatible {key}; use the same bootstrap settings.")
    array_keys = ("year_offsets", "sampled_years") if block_method == "hydrological_year" else (
        "block_length", "block_lengths", "block_starts",
    )
    for key in ("dates", *array_keys):
        if not np.array_equal(samples.get(key), expected[key]):
            raise ValueError(f"Shared samples {path}: {key} differ; align model calendars and sampling settings.")
    for basin, fingerprint in observation_hashes.items():
        if basin not in samples["observation_hashes"]:
            raise ValueError(f"Basin {basin} is absent from {path}; create the shared plan using all catchments.")
        if samples["observation_hashes"][basin] != fingerprint:
            raise ValueError(f"Basin {basin}: observations or valid dates differ from shared samples {path}.")
    return samples


def block_statistics(obs, sim, starts, lengths):
    """Return count, mean, within-block squared deviations and squared model error."""
    statistics = np.zeros((len(starts), 4))
    offsets = np.arange(np.max(lengths))
    # Bound temporary memory even when every day is a possible block start.
    for first in range(0, len(starts), 256):
        stop = min(first + 256, len(starts))
        indices = starts[first:stop, None] + offsets
        in_block = offsets < lengths[first:stop, None]
        indices = np.minimum(indices, len(obs) - 1)
        valid = in_block & np.isfinite(obs[indices])
        values = np.where(valid, obs[indices], 0.0)
        reference = values[np.arange(len(values)), valid.argmax(axis=1)]
        centered = np.where(valid, values - reference[:, None], 0.0)
        counts = valid.sum(axis=1)
        centered_mean = np.divide(
            centered.sum(axis=1), counts, out=np.zeros(len(counts)), where=counts > 0,
        )
        deviations = np.where(valid, centered - centered_mean[:, None], 0.0)
        errors = np.where(valid, values - sim[indices], 0.0)
        statistics[first:stop] = np.column_stack((
            counts, reference + centered_mean,
            (deviations**2).sum(axis=1), (errors**2).sum(axis=1),
        ))
    return statistics


def compute_bootstrap(data, samples):
    """Evaluate full-resample NSE efficiently from per-block sufficient statistics."""
    nse_dict = {}
    for basin, (obs, sim) in data.items():
        if samples["method"] == "october_september_years":
            offsets = samples["year_offsets"]
            statistics = block_statistics(obs, sim, offsets[:-1], np.diff(offsets))
            sampled_statistics = statistics[samples["sampled_years"]]
        elif samples["method"] == "sliding_365_days":
            length = samples["block_length"]
            starts = np.arange(len(obs) - length + 1)
            statistics = block_statistics(obs, sim, starts, np.full(len(starts), length))
            sampled_statistics = statistics[samples["block_starts"]]
            last_length = samples["block_lengths"][-1]
            if last_length != length:
                # Trim only the final sampled block, so each replicate has N calendar days.
                last_starts, inverse = np.unique(samples["block_starts"][:, -1], return_inverse=True)
                tail = block_statistics(obs, sim, last_starts, np.full(len(last_starts), last_length))
                sampled_statistics[:, -1] = tail[inverse]
        else:
            raise ValueError(f"Unknown sampling method: {samples['method']}")
        counts_by_block, means, within_block, errors = sampled_statistics.transpose(2, 0, 1)
        counts = counts_by_block.sum(axis=1)
        # Pool within- and between-block variance without subtracting large squares.
        # Center on a sampled block's mean, keeping constant-flow replicates exactly zero.
        relative_means = means - means.max(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            resample_mean = (counts_by_block * relative_means).sum(axis=1) / counts
            denominator = (
                within_block + counts_by_block * (relative_means - resample_mean[:, None])**2
            ).sum(axis=1)
            bootstrapped_nse = 1 - errors.sum(axis=1) / denominator
        undefined = (counts < 2) | (denominator <= 0) | ~np.isfinite(bootstrapped_nse)
        bootstrapped_nse[undefined] = np.nan
        if undefined.any():
            warnings.warn(
                f"Basin {basin}: {undefined.sum()} NSE replicates are undefined "
                "(insufficient observations or zero observed variance); retained as NaN to preserve pairing.",
                stacklevel=2,
            )
        nse_dict[basin] = bootstrapped_nse
        finite_scores = bootstrapped_nse[np.isfinite(bootstrapped_nse)]
        quantiles = np.quantile(finite_scores, [0.025, 0.5, 0.975]) if finite_scores.size else [np.nan] * 3
        print(f"Basin {basin}: 2.5%={quantiles[0]:.6f}, median={quantiles[1]:.6f}, 97.5%={quantiles[2]:.6f}")
    return nse_dict


def main():
    args = get_args()
    suffix = "_random_features" if args.random_features else ""
    input_path = Path(
        f"analysis/results_data/{args.experiment}_es{args.encoded_features}_{args.eval_period}{suffix}.pkl"
    )
    with input_path.open("rb") as handle:
        ens_dict = pickle.load(handle)
    dates, data, observation_hashes = prepare_data(ens_dict)
    plan_prefix = "hydrological_year" if args.block_method == "hydrological_year" else "sliding365"
    samples_path = args.samples_file or Path(
        f"analysis/bootstrap/{plan_prefix}_samples_{args.eval_period}"
        f"_seed{args.seed}_n{args.n_bootstrap}.pkl"
    )
    samples = load_or_create_samples(
        samples_path, dates, observation_hashes, args.n_bootstrap, args.seed, args.block_method,
    )
    print(f"Using shared samples: {samples_path}")
    nse_dict = compute_bootstrap(data, samples)
    method_suffix = "" if args.block_method == "hydrological_year" else "_sliding365"
    output_path = Path(
        f"analysis/bootstrap/nse_{args.experiment}_{args.encoded_features}_{args.eval_period}{method_suffix}{suffix}.pkl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(nse_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = {
        "method": samples["method"],
        "samples_file": str(samples_path),
        "samples_sha256": hashlib.sha256(samples_path.read_bytes()).hexdigest(),
        "seed": args.seed,
        "n_bootstrap": args.n_bootstrap,
        "start_date": str(dates[0].date()),
        "end_date": str(dates[-1].date()),
    }
    if args.block_method == "hydrological_year":
        metadata["year_lengths"] = np.diff(samples["year_offsets"]).tolist()
    else:
        metadata["block_length"] = samples["block_length"]
        metadata["block_lengths"] = samples["block_lengths"].tolist()
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved paired NSE replicates: {output_path}")


if __name__ == "__main__":
    main()
