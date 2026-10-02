import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.datautils import CLIM_ATTRS, GEOL_ATTRS, SOIL_ATTRS, TOPO_ATTRS, VEGE_ATTRS, load_attributes
from src.models import Hydro_Attention
from src.utils import get_basin_list

KEEP_ATTRS = SOIL_ATTRS + CLIM_ATTRS + VEGE_ATTRS + TOPO_ATTRS + GEOL_ATTRS
DEFAULT_NRNS = 4
DEFAULT_NSPLITS = 12


def get_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute basin-wise encoded features for GLOBAL or PUB models. "
            "GLOBAL outputs have shape [n_runs, n_features]. PUB outputs have "
            "shape [n_runs, n_splits, n_features] because each split model "
            "encodes all basins."
        )
    )
    parser.add_argument(
        "--encoded_features",
        type=int,
        default=2,
        help="Encoded space dimension.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="global",
        choices=["global", "pub"],
        help="Model family to encode.",
    )
    parser.add_argument(
        "--nruns",
        type=int,
        default=DEFAULT_NRNS,
        help="Number of runs to process.",
    )
    parser.add_argument(
        "--nsplits",
        type=int,
        default=DEFAULT_NSPLITS,
        help="Number of PUB splits per run.",
    )
    return vars(parser.parse_args())


def parse_run_dir(report_file: Path) -> Path:
    """Parse the run directory from a training report file."""
    run_dir = None
    with report_file.open("r") as fp:
        for line in fp:
            if line.startswith("run_dir"):
                if "Attention4Hydro/" in line:
                    run_dir = line.split("Attention4Hydro/", 1)[1].strip()
                elif ":" in line:
                    run_dir = line.split(":", 1)[1].strip()
                else:
                    run_dir = line.replace("run_dir", "", 1).strip()
                break

    if run_dir is None:
        raise ValueError(f"Could not find a run_dir entry in {report_file}.")

    resolved = Path(run_dir)
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    if not resolved.is_dir():
        raise FileNotFoundError(f"Resolved run directory does not exist: {resolved}")
    return resolved


def load_model(report_file: Path, encoded_features: int) -> Hydro_Attention:
    """Load one trained encoder from its report file."""
    run_dir = parse_run_dir(report_file)
    ckpt_file = run_dir / "last.ckpt"
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Missing checkpoint file: {ckpt_file}")

    model = Hydro_Attention(
        input_dim=26,
        output_dim=encoded_features,
        hidden_layers=4 * [300],
        act=nn.LeakyReLU(),
    )
    checkpoint = torch.load(ckpt_file, map_location=torch.device("cpu"), weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def encode_basin_attributes(
    model: Hydro_Attention,
    attribute_frame,
) -> np.ndarray:
    """Encode a standardized basin-attribute matrix with one trained model."""
    attrs = torch.tensor(attribute_frame.to_numpy(dtype=np.float32), dtype=torch.float32)
    with torch.no_grad():
        encodings = model.encoder(attrs)
    return encodings.detach().cpu().numpy()


def standardize_attributes(attribute_frame, mean_values, std_values):
    """Standardize attributes with provided moments."""
    safe_stds = std_values.replace(0.0, 1.0)
    return (attribute_frame - mean_values) / safe_stds


def compute_global_features(
    all_basins: list[str],
    df_attr,
    encoded_features: int,
    nruns: int,
) -> dict:
    """Compute GLOBAL features for all runs on all basins."""
    features = {
        basin: np.full((nruns, encoded_features), np.nan, dtype=float)
        for basin in all_basins
    }

    mean_values = df_attr.mean(axis=0)
    std_values = df_attr.std(axis=0)
    standardized_attr = standardize_attributes(df_attr, mean_values, std_values)

    for run in tqdm(range(nruns), desc="GLOBAL runs"):
        report_file = Path(f"reports/global_ae_es{encoded_features}_run{run}.out")
        if not report_file.is_file():
            raise FileNotFoundError(f"Missing report file: {report_file}")

        model = load_model(report_file=report_file, encoded_features=encoded_features)
        encodings = encode_basin_attributes(model=model, attribute_frame=standardized_attr.loc[all_basins])
        for basin_idx, basin in enumerate(all_basins):
            features[basin][run, :] = encodings[basin_idx, :]

    features["__metadata__"] = {
        "layout": "[n_runs, n_features]",
        "nruns": nruns,
        "encoded_features": encoded_features,
    }
    return features


def compute_pub_features(
    all_basins: list[str],
    df_attr,
    encoded_features: int,
    nruns: int,
    nsplits: int,
) -> dict:
    """Compute PUB features for all run-split models on all basins."""
    features = {
        basin: np.full((nruns, nsplits, encoded_features), np.nan, dtype=float)
        for basin in all_basins
    }

    for run in tqdm(range(nruns), desc="PUB runs"):
        split_seed = 300 + run
        split_file = Path(f"data/kfold_splits_seed{split_seed}.p")
        if not split_file.is_file():
            raise FileNotFoundError(f"Missing split file: {split_file}")

        with split_file.open("rb") as fp:
            splits = pickle.load(fp)

        for split in range(nsplits):
            report_file = Path(f"reports/pub_ae_es{encoded_features}_run{run}_split{split}.out")
            if not report_file.is_file():
                raise FileNotFoundError(f"Missing report file: {report_file}")

            train_basins = [str(basin).zfill(8) for basin in splits[split]["train"]]
            mean_values = df_attr.loc[train_basins].mean(axis=0)
            std_values = df_attr.loc[train_basins].std(axis=0)
            standardized_attr = standardize_attributes(df_attr, mean_values, std_values)

            model = load_model(report_file=report_file, encoded_features=encoded_features)
            encodings = encode_basin_attributes(
                model=model,
                attribute_frame=standardized_attr.loc[all_basins],
            )
            for basin_idx, basin in enumerate(all_basins):
                features[basin][run, split, :] = encodings[basin_idx, :]

    features["__metadata__"] = {
        "layout": "[n_runs, n_splits, n_features]",
        "nruns": nruns,
        "nsplits": nsplits,
        "encoded_features": encoded_features,
        "note": (
            "PUB features are encoded for every run and every split on all basins. "
            "Downstream split-wise analysis should still select only the split "
            "matching each basin's test membership."
        ),
    }
    return features


if __name__ == "__main__":
    cfg = get_args()
    encoded_features = cfg["encoded_features"]
    experiment = cfg["experiment"]
    nruns = cfg["nruns"]
    nsplits = cfg["nsplits"]

    all_basins = get_basin_list()
    df_attr = load_attributes("data/attributes.db", all_basins, keep_attributes=KEEP_ATTRS)
    df_attr = df_attr.loc[all_basins, KEEP_ATTRS]
    df_attr.index = [str(basin).zfill(8) for basin in df_attr.index]
    all_basins = [str(basin).zfill(8) for basin in all_basins]

    if experiment == "global":
        features = compute_global_features(
            all_basins=all_basins,
            df_attr=df_attr,
            encoded_features=encoded_features,
            nruns=nruns,
        )
    else:
        features = compute_pub_features(
            all_basins=all_basins,
            df_attr=df_attr,
            encoded_features=encoded_features,
            nruns=nruns,
            nsplits=nsplits,
        )

    output_file = Path(f"analysis/encoded_features/{experiment}_es{encoded_features}.pkl")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as fp:
        pickle.dump(features, fp)

    print(f"Saved encoded features to {output_file}")
