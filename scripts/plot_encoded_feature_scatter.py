import argparse
import pickle
from functools import lru_cache
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr
from sklearn.decomposition import FastICA, PCA

from src.datautils import CLIM_ATTRS, GEOL_ATTRS, SOIL_ATTRS, TOPO_ATTRS, VEGE_ATTRS, load_attributes, load_signatures
from src.utils import clean_and_capitalize, compute_grid, get_basin_list, str2bool

mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

DEFAULT_OUTPUT_DIR = Path("analysis/figures")
BASIN_ATTR_DB = Path("data/attributes.db")
SIGNATURE_FILE = Path("analysis/signatures/camels_us_test.csv")
ATTRIBUTE_NAMES = SOIL_ATTRS + CLIM_ATTRS + VEGE_ATTRS + TOPO_ATTRS + GEOL_ATTRS
SIGNATURE_NAMES = [
    "high_q_freq",
    "high_q_dur",
    "low_q_freq",
    "low_q_dur",
    "zero_q_freq",
    "q95",
    "q5",
    "q_mean",
    "hfd_mean",
    "baseflow_index",
    "slope_fdc",
    "stream_elas",
    "runoff_ratio",
]
TARGET_NAMES = ATTRIBUTE_NAMES + SIGNATURE_NAMES


def get_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot scatter plots between each encoded feature of a model and one "
            "selected catchment attribute or streamflow signature."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="global",
        choices=["global", "pub"],
        help="Model family to plot.",
    )
    parser.add_argument(
        "--encoded_features",
        type=int,
        required=True,
        help="Latent space dimension used to load analysis/encoded_features/{model}_es{N}.pkl.",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help=(
            "One target variable among the 26 catchment attributes or 13 streamflow "
            "signatures."
        ),
    )
    parser.add_argument(
        "--run",
        type=int,
        default=None,
        help=(
            "Optional run index to plot. If omitted, the script aligns feature axes "
            "across runs and plots the mean value per basin."
        ),
    )
    parser.add_argument(
        "--with_ica",
        type=str2bool,
        default=False,
        help="After Procrustes alignment, rotate all runs with one ICA basis fitted on the consensus mean space.",
    )
    parser.add_argument(
        "--with_pca",
        type=str2bool,
        default=False,
        help="After Procrustes alignment, rotate all runs with one PCA basis fitted on the consensus mean space.",
    )
    parser.add_argument(
        "--marker_size",
        type=float,
        default=45.0,
        help="Marker size for scatter points.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.85,
        help="Scatter point alpha.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output figure DPI.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path. Defaults to analysis/figures/encoded_feature_scatter_*.png.",
    )
    return vars(parser.parse_args())


def validate_target(target: str) -> str:
    """Validate the chosen target name."""
    if target not in TARGET_NAMES:
        raise ValueError(
            f"Unknown target '{target}'. Choose one of: {', '.join(TARGET_NAMES)}"
        )
    return target


def load_split_file(run: int) -> dict:
    """Load basin splits for one PUB run."""
    split_seed = 300 + run
    split_file = Path(f"data/kfold_splits_seed{split_seed}.p")
    if not split_file.is_file():
        raise FileNotFoundError(f"Missing split file: {split_file}")

    with split_file.open("rb") as fp:
        return pickle.load(fp)


def load_raw_encoded_feature_tensor(
    model: str,
    encoded_features: int,
) -> tuple[list[str], np.ndarray]:
    """Load raw basin-wise encoded features from disk."""
    feature_file = Path(f"analysis/encoded_features/{model}_es{encoded_features}.pkl")
    if not feature_file.is_file():
        raise FileNotFoundError(f"Missing encoded feature file: {feature_file}")

    with feature_file.open("rb") as fp:
        feature_dict = pickle.load(fp)

    basins = [basin for basin in get_basin_list() if basin in feature_dict]
    if not basins:
        raise ValueError(f"No CAMELS basins found in {feature_file}.")

    features = []
    for basin in basins:
        values = np.asarray(feature_dict[basin], dtype=float)
        if values.ndim == 1:
            values = values[np.newaxis, :]
        features.append(values)

    feature_tensor = np.stack(features, axis=0)
    if feature_tensor.shape[-1] != encoded_features:
        raise ValueError(
            "Loaded encoded features do not match the requested latent dimension: "
            f"expected {encoded_features}, found {feature_tensor.shape[-1]}."
        )

    return basins, feature_tensor


def align_pub_split_tensor(feature_tensor: np.ndarray) -> np.ndarray:
    """Align PUB split models within each run using all basins."""
    aligned_tensor = np.zeros_like(feature_tensor)
    for run in range(feature_tensor.shape[1]):
        aligned_tensor[:, run, :, :] = align_feature_axes(feature_tensor[:, run, :, :])
    return aligned_tensor


def select_pub_test_split_features(
    basins: list[str],
    feature_tensor: np.ndarray,
) -> np.ndarray:
    """Select, for each run and basin, the encoding from that basin's test split."""
    selected_tensor = np.full(
        (len(basins), feature_tensor.shape[1], feature_tensor.shape[3]),
        np.nan,
        dtype=float,
    )
    basin_to_index = {basin: idx for idx, basin in enumerate(basins)}

    for run in range(feature_tensor.shape[1]):
        splits = load_split_file(run)
        for split in range(feature_tensor.shape[2]):
            test_basins = [str(basin).zfill(8) for basin in splits[split]["test"]]
            for basin in test_basins:
                basin_idx = basin_to_index.get(basin)
                if basin_idx is None:
                    continue
                selected_tensor[basin_idx, run, :] = feature_tensor[basin_idx, run, split, :]

    if np.any(~np.isfinite(selected_tensor)):
        missing_idx = np.argwhere(~np.isfinite(selected_tensor).all(axis=2))
        if missing_idx.size > 0:
            basin_idx, run_idx = missing_idx[0]
            raise ValueError(
                "Missing selected PUB test features after split selection for "
                f"basin {basins[basin_idx]} and run {run_idx}."
            )
    return selected_tensor


def load_pub_split_feature_tensor(
    encoded_features: int,
) -> tuple[list[str], np.ndarray]:
    """Load PUB split encodings and align split subspaces on all basins."""
    basins, feature_tensor = load_raw_encoded_feature_tensor(
        model="pub",
        encoded_features=encoded_features,
    )
    if feature_tensor.ndim != 4:
        raise ValueError(
            "Expected PUB encoded features with shape [n_basins, n_runs, n_splits, n_features]. "
            f"Found tensor with shape {feature_tensor.shape}."
        )
    aligned_tensor = align_pub_split_tensor(feature_tensor)
    aligned_tensor = align_feature_tensor_to_global_reference_signs(
        basins=basins,
        feature_tensor=aligned_tensor,
        encoded_features=encoded_features,
        with_ica=False,
        with_pca=False,
    )
    return basins, aligned_tensor


def load_encoded_feature_tensor(
    model: str,
    encoded_features: int,
) -> tuple[list[str], np.ndarray]:
    """Load basin-wise encoded features with shape [n_basins, n_runs, n_features]."""
    basins, feature_tensor = load_raw_encoded_feature_tensor(
        model=model,
        encoded_features=encoded_features,
    )
    if model == "pub" and feature_tensor.ndim == 4:
        aligned_split_tensor = align_pub_split_tensor(feature_tensor)
        aligned_split_tensor = align_feature_tensor_to_global_reference_signs(
            basins=basins,
            feature_tensor=aligned_split_tensor,
            encoded_features=encoded_features,
            with_ica=False,
            with_pca=False,
        )
        return basins, select_pub_test_split_features(basins, aligned_split_tensor)
    return basins, feature_tensor


def summarize_feature_tensor(feature_tensor: np.ndarray) -> np.ndarray:
    """Average a feature tensor over every model axis except basins and features."""
    if feature_tensor.ndim < 2:
        raise ValueError(
            "Expected at least a basin-by-feature matrix to summarize the feature tensor."
        )
    if feature_tensor.ndim == 2:
        return feature_tensor
    reduce_axes = tuple(range(1, feature_tensor.ndim - 1))
    return np.nanmean(feature_tensor, axis=reduce_axes)


@lru_cache(maxsize=None)
def build_global_reference_values(
    encoded_features: int,
    with_ica: bool,
    with_pca: bool,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Build the GLOBAL basin-wise reference used to orient other model signs."""
    basins, feature_tensor = load_raw_encoded_feature_tensor(
        model="global",
        encoded_features=encoded_features,
    )
    if feature_tensor.ndim != 3:
        raise ValueError(
            "Expected GLOBAL encoded features with shape [n_basins, n_runs, n_features]. "
            f"Found tensor with shape {feature_tensor.shape}."
        )

    transformed_tensor = align_feature_axes(feature_tensor)
    if with_ica:
        transformed_tensor = rotate_feature_tensor_to_reference_ica(transformed_tensor)
    if with_pca:
        transformed_tensor = rotate_feature_tensor_to_reference_pca(transformed_tensor)
    return tuple(basins), np.nanmean(transformed_tensor, axis=1)


def align_feature_tensor_to_reference_signs(
    basins: list[str],
    feature_tensor: np.ndarray,
    reference_basins: list[str] | tuple[str, ...],
    reference_values: np.ndarray,
) -> np.ndarray:
    """Flip feature signs so a tensor matches a basin-wise reference orientation."""
    reference_index = {basin: idx for idx, basin in enumerate(reference_basins)}
    basin_pairs = [
        (basin_idx, ref_idx)
        for basin_idx, basin in enumerate(basins)
        if (ref_idx := reference_index.get(basin)) is not None
    ]
    if not basin_pairs:
        return feature_tensor

    basin_idx = np.array([pair[0] for pair in basin_pairs], dtype=int)
    ref_idx = np.array([pair[1] for pair in basin_pairs], dtype=int)
    summary_values = summarize_feature_tensor(feature_tensor)[basin_idx, :]
    reference_subset = reference_values[ref_idx, :]
    aligned_tensor = feature_tensor.copy()

    for feature_idx in range(summary_values.shape[1]):
        target_feature = summary_values[:, feature_idx]
        reference_feature = reference_subset[:, feature_idx]
        mask = np.isfinite(target_feature) & np.isfinite(reference_feature)
        if np.count_nonzero(mask) < 2:
            continue
        if np.nanstd(target_feature[mask]) == 0.0 or np.nanstd(reference_feature[mask]) == 0.0:
            continue

        corr = np.corrcoef(target_feature[mask], reference_feature[mask])[0, 1]
        if np.isfinite(corr) and corr < 0.0:
            aligned_tensor[..., feature_idx] *= -1.0

    return aligned_tensor


def align_feature_tensor_to_global_reference_signs(
    basins: list[str],
    feature_tensor: np.ndarray,
    encoded_features: int,
    with_ica: bool,
    with_pca: bool,
) -> np.ndarray:
    """Orient one feature tensor so its component signs follow the GLOBAL reference."""
    reference_basins, reference_values = build_global_reference_values(
        encoded_features=encoded_features,
        with_ica=with_ica,
        with_pca=with_pca,
    )
    return align_feature_tensor_to_reference_signs(
        basins=basins,
        feature_tensor=feature_tensor,
        reference_basins=reference_basins,
        reference_values=reference_values,
    )


def apply_ica_per_run(feature_tensor: np.ndarray) -> np.ndarray:
    """Apply FastICA independently to each run on the basin-by-feature matrix."""
    transformed = np.zeros_like(feature_tensor)
    n_runs = feature_tensor.shape[1]
    n_features = feature_tensor.shape[2]

    for run in range(n_runs):
        ica = FastICA(
            n_components=n_features,
            whiten=False,
            random_state=0,
            max_iter=1000,
        )
        transformed[:, run, :] = ica.fit_transform(feature_tensor[:, run, :])

    return transformed


def fit_consensus_ica(aligned_tensor: np.ndarray) -> tuple[FastICA, np.ndarray]:
    """Fit one ICA basis on the Procrustes consensus mean and return oriented reference scores."""
    reference_values = np.nanmean(aligned_tensor, axis=1)
    ica = FastICA(
        n_components=aligned_tensor.shape[2],
        whiten=False,
        random_state=0,
        max_iter=1000,
    )
    reference_transformed = ica.fit_transform(reference_values)
    reference_transformed = orient_rotated_components(reference_transformed, reference_transformed)
    return ica, reference_transformed


def align_feature_axes(feature_tensor: np.ndarray) -> np.ndarray:
    """Align latent feature axes across runs with generalized orthogonal Procrustes."""
    aligned_tensor = feature_tensor.copy()
    if aligned_tensor.shape[1] <= 1:
        return aligned_tensor

    run_means = np.nanmean(aligned_tensor, axis=0, keepdims=True)
    centered_tensor = aligned_tensor - run_means
    reference = centered_tensor[:, 0, :].copy()
    aligned_centered = centered_tensor.copy()

    for _ in range(32):
        previous_reference = reference.copy()
        for run in range(centered_tensor.shape[1]):
            rotation, _ = orthogonal_procrustes(centered_tensor[:, run, :], reference)
            aligned_centered[:, run, :] = centered_tensor[:, run, :] @ rotation

        reference = np.nanmean(aligned_centered, axis=1)
        denominator = np.linalg.norm(previous_reference) + 1e-12
        if np.linalg.norm(reference - previous_reference) / denominator < 1e-6:
            break

    reference_run = centered_tensor[:, 0, :]
    for feature_idx in range(aligned_centered.shape[2]):
        ref_feature = reference_run[:, feature_idx]
        aligned_feature = reference[:, feature_idx]
        if np.std(ref_feature) == 0.0 or np.std(aligned_feature) == 0.0:
            continue

        corr = np.corrcoef(ref_feature, aligned_feature)[0, 1]
        if np.isfinite(corr) and corr < 0.0:
            aligned_centered[:, :, feature_idx] *= -1.0
            reference[:, feature_idx] *= -1.0

    consensus_mean = np.nanmean(run_means, axis=1, keepdims=True)
    return aligned_centered + consensus_mean


def reduce_feature_tensor(
    feature_tensor: np.ndarray,
    run: int | None,
    with_ica: bool,
    with_pca: bool = False,
    model: str | None = None,
    basins: list[str] | None = None,
    encoded_features: int | None = None,
) -> tuple[np.ndarray, str]:
    """Reduce [n_basins, n_runs, n_features] to [n_basins, n_features]."""
    n_runs = feature_tensor.shape[1]
    needs_common_basis = with_ica or with_pca or run is None
    transformed_tensor = feature_tensor
    if needs_common_basis:
        aligned_tensor = align_feature_axes(feature_tensor)
        transformed_tensor = aligned_tensor
        if with_ica:
            transformed_tensor = rotate_feature_tensor_to_reference_ica(transformed_tensor)
        if with_pca:
            transformed_tensor = rotate_feature_tensor_to_reference_pca(transformed_tensor)
    if model == "pub" and basins is not None and encoded_features is not None:
        transformed_tensor = align_feature_tensor_to_global_reference_signs(
            basins=basins,
            feature_tensor=transformed_tensor,
            encoded_features=encoded_features,
            with_ica=with_ica,
            with_pca=with_pca,
        )

    if run is None:
        feature_values = transformed_tensor.mean(axis=1)
        selection_label = "mean"
        if with_ica:
            selection_label = f"{selection_label}_ica"
        if with_pca:
            selection_label = f"{selection_label}_pca"
    else:
        if run < 0 or run >= n_runs:
            raise ValueError(f"Invalid run index {run}. Available runs: 0 to {n_runs - 1}.")

        feature_values = transformed_tensor[:, run, :]
        selection_label = f"run{run}"
        if with_ica:
            selection_label = f"{selection_label}_ica"
        if with_pca:
            selection_label = f"{selection_label}_pca"

    return feature_values, selection_label


def build_global_pca_reference(
    encoded_features: int,
    run: int | None,
    with_ica: bool,
) -> np.ndarray:
    """Build the GLOBAL reference matrix used to fit the PCA basis."""
    _, global_tensor = load_encoded_feature_tensor(
        model="global",
        encoded_features=encoded_features,
    )

    if with_ica:
        global_tensor = apply_ica_per_run(global_tensor)

    if run is None:
        global_tensor = align_feature_axes(global_tensor)
        return global_tensor.mean(axis=1)

    if run < 0 or run >= global_tensor.shape[1]:
        raise ValueError(
            f"Invalid run index {run} for GLOBAL PCA basis. Available runs: 0 to {global_tensor.shape[1] - 1}."
        )
    return global_tensor[:, run, :]


def orient_rotated_components(reference_values: np.ndarray, transformed_values: np.ndarray) -> np.ndarray:
    """Fix rotated-component signs using the reference transform."""
    oriented_values = transformed_values.copy()
    for component_idx in range(reference_values.shape[1]):
        reference_component = reference_values[:, component_idx]
        if np.nanmean(reference_component) < 0.0:
            oriented_values[:, component_idx] *= -1.0
    return oriented_values


def rotate_features_to_global_pca(
    feature_values: np.ndarray,
    encoded_features: int,
    run: int | None,
    with_ica: bool,
) -> np.ndarray:
    """Rotate features into a PCA basis fitted on the GLOBAL model."""
    global_reference = build_global_pca_reference(
        encoded_features=encoded_features,
        run=run,
        with_ica=with_ica,
    )
    pca = PCA(n_components=encoded_features, svd_solver="full")
    reference_transformed = pca.fit_transform(global_reference)
    rotated_values = pca.transform(feature_values)
    return orient_rotated_components(reference_transformed, rotated_values)


def fit_consensus_pca(aligned_tensor: np.ndarray) -> tuple[PCA, np.ndarray]:
    """Fit one PCA basis on the Procrustes consensus mean and return oriented reference scores."""
    reference_values = np.nanmean(aligned_tensor, axis=1)
    pca = PCA(n_components=aligned_tensor.shape[2], svd_solver="full")
    reference_transformed = pca.fit_transform(reference_values)
    reference_transformed = orient_rotated_components(reference_transformed, reference_transformed)
    return pca, reference_transformed


def rotate_feature_tensor_to_reference_ica(
    aligned_tensor: np.ndarray,
) -> np.ndarray:
    """Rotate each aligned run with a single ICA basis fitted on the consensus mean space."""
    ica, reference_transformed = fit_consensus_ica(aligned_tensor=aligned_tensor)
    rotated_tensor = np.zeros_like(aligned_tensor)
    for run in range(aligned_tensor.shape[1]):
        transformed = ica.transform(aligned_tensor[:, run, :])
        rotated_tensor[:, run, :] = orient_rotated_components(reference_transformed, transformed)
    return rotated_tensor


def rotate_feature_tensor_to_reference_pca(
    aligned_tensor: np.ndarray,
) -> np.ndarray:
    """Rotate each aligned run with a single PCA basis fitted on the consensus mean space."""
    pca, reference_transformed = fit_consensus_pca(aligned_tensor=aligned_tensor)
    rotated_tensor = np.zeros_like(aligned_tensor)
    for run in range(aligned_tensor.shape[1]):
        transformed = pca.transform(aligned_tensor[:, run, :])
        rotated_tensor[:, run, :] = orient_rotated_components(reference_transformed, transformed)
    return rotated_tensor


def rotate_feature_tensor_to_global_pca(
    feature_tensor: np.ndarray,
    encoded_features: int,
    with_ica: bool,
) -> np.ndarray:
    """Rotate each run of a feature tensor into the corresponding GLOBAL PCA basis."""
    rotated_tensor = np.zeros_like(feature_tensor)
    for run in range(feature_tensor.shape[1]):
        rotated_tensor[:, run, :] = rotate_features_to_global_pca(
            feature_values=feature_tensor[:, run, :],
            encoded_features=encoded_features,
            run=run,
            with_ica=with_ica,
        )
    return rotated_tensor


def load_attribute_target(basins: list[str], target: str) -> pd.Series:
    """Load one catchment attribute."""
    df_attr = load_attributes(
        BASIN_ATTR_DB,
        basins,
        keep_attributes=[target],
    )
    series = df_attr.loc[basins, target]
    series.index = [str(basin).zfill(8) for basin in series.index]
    return series


def load_signature_target(basins: list[str], target: str) -> pd.Series:
    """Load one streamflow signature."""
    if not SIGNATURE_FILE.is_file():
        raise FileNotFoundError(f"Missing signature file: {SIGNATURE_FILE}")

    df_sig = load_signatures(SIGNATURE_FILE)
    return df_sig.loc[basins, target]


def load_target_series(basins: list[str], target: str) -> tuple[pd.Series, str]:
    """Load the selected target as a basin-indexed series."""
    if target in ATTRIBUTE_NAMES:
        return load_attribute_target(basins, target), "attribute"
    return load_signature_target(basins, target), "signature"


def build_output_path(cfg: dict, selection_label: str, target_kind: str) -> Path:
    """Resolve output path for the figure."""
    if cfg["output"] is not None:
        return Path(cfg["output"])

    return DEFAULT_OUTPUT_DIR / (
        f"encoded_feature_scatter_{cfg['model']}_es{cfg['encoded_features']}_{target_kind}_"
        f"{cfg['target']}_{selection_label}.png"
    )


def plot_feature_scatter_grid(
    feature_values: np.ndarray,
    target_series: pd.Series,
    model: str,
    encoded_features: int,
    target_name: str,
    target_kind: str,
    selection_label: str,
    marker_size: float,
    alpha: float,
    output_path: Path,
    dpi: int,
) -> None:
    """Plot one scatter subplot per encoded feature."""
    n_features = feature_values.shape[1]
    nrows, ncols = compute_grid(n_features)
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(5.5 * ncols, 4.3 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    axes = axs.ravel()
    x_label = clean_and_capitalize(target_name).strip()
    x_values = target_series.to_numpy(dtype=float)

    for feature_idx, ax in enumerate(axes):
        if feature_idx >= n_features:
            ax.set_visible(False)
            continue

        y_values = feature_values[:, feature_idx]
        mask = np.isfinite(x_values) & np.isfinite(y_values)
        x_plot = x_values[mask]
        y_plot = y_values[mask]

        ax.scatter(
            x_plot,
            y_plot,
            s=marker_size,
            alpha=alpha,
            color="#2166AC",
            edgecolors="none",
        )

        if x_plot.size >= 2:
            slope, intercept = np.polyfit(x_plot, y_plot, 1)
            x_line = np.linspace(np.min(x_plot), np.max(x_plot), 200)
            ax.plot(x_line, slope * x_line + intercept, color="#111827", linewidth=1.5)
            corr, pvalue = spearmanr(x_plot, y_plot)
        else:
            corr, pvalue = np.nan, np.nan

        ax.grid(alpha=0.3)
        ax.set_title(f"EF {feature_idx + 1}", fontsize=18)
        ax.set_xlabel(x_label, fontsize=16)
        ax.set_ylabel("Encoded feature value", fontsize=16)
        ax.text(
            0.03,
            0.97,
            f"$\\rho$ = {corr:.2f}\np = {pvalue:.2e}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=13,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#d1d5db"),
        )

    figure_title = (
        f"{model.upper()} encoded features vs {target_kind} "
        f"{x_label} (ES={encoded_features}, {selection_label})"
    )
    fig.suptitle(figure_title, fontsize=22)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    cfg = get_args()
    cfg["target"] = validate_target(cfg["target"])

    basins, feature_tensor = load_encoded_feature_tensor(
        model=cfg["model"],
        encoded_features=cfg["encoded_features"],
    )
    feature_values, selection_label = reduce_feature_tensor(
        feature_tensor=feature_tensor,
        run=cfg["run"],
        with_ica=cfg["with_ica"],
        with_pca=cfg["with_pca"],
        model=cfg["model"],
        basins=basins,
        encoded_features=cfg["encoded_features"],
    )
    target_series, target_kind = load_target_series(basins, cfg["target"])
    output_path = build_output_path(cfg, selection_label, target_kind)

    plot_feature_scatter_grid(
        feature_values=feature_values,
        target_series=target_series,
        model=cfg["model"],
        encoded_features=cfg["encoded_features"],
        target_name=cfg["target"],
        target_kind=target_kind,
        selection_label=selection_label,
        marker_size=cfg["marker_size"],
        alpha=cfg["alpha"],
        output_path=output_path,
        dpi=cfg["dpi"],
    )

    print(f"Saved figure to {output_path}")
