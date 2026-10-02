
import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def get_args():
    from src.performance_batch import METRIC_NAMES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", help="For example, global, pub, global_linear or pub_linear")
    parser.add_argument("encoded_features")
    parser.add_argument("eval_period", choices=["val", "test"])
    parser.add_argument("metric", choices=METRIC_NAMES)
    parser.add_argument("--random_features", action="store_true")
    return parser.parse_args()


def compute_leave_one_out(ens_dict, metric="nse"):
    """Evaluate all variants on the same valid dates, without modifying inputs."""
    # Loading existing CSVs does not require the metric engine or Numba.
    from src.performance_batch import METRIC_NAMES, calculate_metrics

    if not ens_dict:
        raise ValueError("No basins found in the ensemble results.")
    if metric not in METRIC_NAMES:
        raise ValueError(f"Unknown metric: {metric}")

    first = next(iter(ens_dict.values()))
    run_columns = sorted(col for col in first.columns if col.startswith("qsim_"))
    if len(run_columns) < 2:
        raise ValueError("At least two qsim_<seed> columns are required.")

    scores = {}
    for basin, frame in ens_dict.items():
        columns = sorted(col for col in frame.columns if col.startswith("qsim_"))
        if columns != run_columns:
            raise ValueError(f"Basin {basin}: run seeds must match across basins.")
        if not frame.index.is_unique or not frame.columns.is_unique:
            raise ValueError(f"Basin {basin}: duplicate dates or columns.")
        data = frame[["qobs", "qsim", *run_columns]]
        valid = np.isfinite(data).all(axis=1) & (data >= 0).all(axis=1)
        data = data.loc[valid]
        if len(data) < 2:
            raise ValueError(f"Basin {basin}: fewer than two common valid dates.")
        if metric == "nse" and data["qobs"].nunique() < 2:
            raise ValueError(f"Basin {basin}: NSE is undefined for constant observations.")
        if not np.allclose(data["qsim"], data[run_columns].mean(axis=1)):
            raise ValueError(f"Basin {basin}: qsim differs from the mean of the runs.")

        # Batch the saved full ensemble and the mean prediction for each omission.
        # Never average individual-run metric scores.
        predictions = [data["qsim"].to_numpy()]
        variant_names = ["full"]
        for omitted in run_columns:
            remaining = [col for col in run_columns if col != omitted]
            predictions.append(data[remaining].mean(axis=1).to_numpy())
            seed = omitted.removeprefix("qsim_")
            variant_names.append(f"without_{seed}")
        metrics = calculate_metrics(data["qobs"].to_numpy(), np.column_stack(predictions))
        scores[basin] = dict(zip(variant_names, metrics[metric].to_numpy()))

    return pd.DataFrame.from_dict(scores, orient="index").rename_axis("basin")


def load_leave_one_out(model_specs, eval_period, metric, directory=Path("analysis/bootstrap")):
    """Keep the same finite basin population for every model and every variant."""
    directory = Path(directory)
    inputs = {}
    for spec in model_specs:
        experiment = spec["file_experiment"]
        suffix = "_random_features" if spec["random_features"] else ""
        path = directory / (
            f"leave_one_out_{metric}_{experiment}_{spec['encoded_features']}_{eval_period}{suffix}.csv"
        )
        if not path.is_file():
            if spec["random_features"]:
                warnings.warn(f"Skipping missing random-features leave-one-out file: {path}")
                continue
            raise FileNotFoundError(
                f"Missing {path}. Generate it with: python -m scripts.compute_leave_one_out "
                f"{experiment} {spec['encoded_features']} {eval_period} {metric}"
            )
        scores = pd.read_csv(path, dtype={"basin": str}, index_col="basin")
        omissions = [col for col in scores if col.startswith("without_")]
        if "full" not in scores or len(omissions) < 2 or len(scores.columns) != len(omissions) + 1:
            raise ValueError(f"Expected full and without_<seed> columns in {path}")
        if scores.index.has_duplicates or scores.index.hasnans:
            raise ValueError(f"Basin IDs must be unique and nonmissing in {path}")
        inputs[spec["label"]] = scores[["full", *omissions]]
    if not inputs:
        raise ValueError("No leave-one-run-out results found.")
    common = pd.concat(inputs, axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(common) < 2:
        raise ValueError("At least two common basins with finite scores in all variants are required.")
    print(f"Using {len(common)} common basins for leave-one-run-out {metric} analysis in the {eval_period} period.")
    return {label: common[label] for label in inputs}


def main():
    args = get_args()
    suffix = "_random_features" if args.random_features else ""
    input_path = Path("analysis/results_data") / (
        f"{args.experiment}_es{args.encoded_features}_{args.eval_period}{suffix}.pkl"
    )
    with input_path.open("rb") as handle:
        ens_dict = pickle.load(handle)
    scores = compute_leave_one_out(ens_dict, args.metric)

    # A separate, labelled format prevents sensitivity variants being mistaken
    # for the random bootstrap replicates saved by the previous script.
    output_path = Path("analysis/bootstrap") / (
        f"leave_one_out_{args.metric}_{args.experiment}_{args.encoded_features}_"
        f"{args.eval_period}{suffix}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output_path)
    print(f"Saved full ensemble and {len(scores.columns) - 1} omissions for "
          f"{len(scores)} basins: {output_path}")


if __name__ == "__main__":
    main()
