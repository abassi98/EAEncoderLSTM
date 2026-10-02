import argparse
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
from matplotlib.patches import Polygon
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import FastICA

from src.plot_utils import add_basemap, load_state_polygons_local, load_us_states
from scripts.plot_encoded_feature_scatter import (
    align_feature_tensor_to_global_reference_signs,
    load_encoded_feature_tensor as load_shared_encoded_feature_tensor,
    rotate_feature_tensor_to_reference_ica,
    rotate_feature_tensor_to_reference_pca,
)
from src.datautils import load_attributes
from src.utils import compute_grid, get_basin_list, str2bool

mpl.rcParams["xtick.labelsize"] = 16
mpl.rcParams["ytick.labelsize"] = 16
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

DEFAULT_OUTPUT_DIR = Path("analysis/figures")
BASIN_ATTR_DB = Path("data/attributes.db")
BASEMAP_CHOICES = ["none", "light", "terrain", "satellite"]


def get_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot one US map per encoded feature for a selected model and latent "
            "space dimension."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="global",
        choices=["global", "pub", "both"],
        help="Model family to plot. Use 'both' to compare GLOBAL and PUB in one figure.",
    )
    parser.add_argument(
        "--encoded_features",
        type=int,
        required=True,
        help="Latent space dimension used to load analysis/encoded_features/{model}_es{N}.pkl.",
    )
    parser.add_argument(
        "--run",
        type=int,
        default=None,
        help=(
            "Optional run index to plot. If omitted, the script aligns feature signs "
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
        "--basemap",
        type=str,
        default="light",
        choices=BASEMAP_CHOICES,
        help="Background basemap style from src.plot_utils.",
    )
    parser.add_argument(
        "--marker_size",
        type=float,
        default=45.0,
        help="Marker size for basin points.",
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
        help="Optional output path. Defaults to analysis/figures/encoded_features_*.png.",
    )
    return vars(parser.parse_args())


def load_encoded_feature_tensor(
    model: str,
    encoded_features: int,
) -> tuple[list[str], np.ndarray]:
    """Load basin-wise latent features with shape [n_basins, n_runs, n_features]."""
    return load_shared_encoded_feature_tensor(model=model, encoded_features=encoded_features)


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
    with_pca: bool,
    model: str,
    basins: list[str],
    encoded_features: int,
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
    if model == "pub":
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


def load_basin_locations(basins: list[str]):
    """Load gauge latitude and longitude for the requested basins."""
    locations = load_attributes(
        BASIN_ATTR_DB,
        basins,
        keep_attributes=["gauge_lat", "gauge_lon"],
    )
    return locations.loc[basins]


def draw_background_map(ax, basemap: str, us_states, fallback_polygons: list[list[tuple[float, float]]]) -> None:
    """Draw US states and the optional basemap."""
    ax.set_xlim(-128, -65)
    ax.set_ylim(24, 50)
    background_color = "#b5c7d8" if basemap == "light" else "#d9e8f5"
    ax.set_facecolor(background_color)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    add_basemap(ax, basemap)
    use_transparent_fill = basemap != "none"

    if us_states is not None:
        us_states.plot(
            ax=ax,
            color="none" if use_transparent_fill else "#f4f1e8",
            edgecolor="#111827",
            linewidth=0.6,
            zorder=1.5,
        )
        us_states.boundary.plot(color="#000000", ax=ax, linewidth=0.75, zorder=2)
        return

    for polygon in fallback_polygons:
        ax.add_patch(
            Polygon(
                polygon,
                closed=True,
                facecolor="none" if use_transparent_fill else "#f4f1e8",
                edgecolor="#111827",
                linewidth=0.55,
                zorder=1.5,
            )
        )


def build_output_path(cfg: dict, selection_label: str) -> Path:
    """Resolve output path for the figure."""
    if cfg["output"] is not None:
        return Path(cfg["output"])

    return DEFAULT_OUTPUT_DIR / (
        f"encoded_features_{cfg['model']}_es{cfg['encoded_features']}_{selection_label}_{cfg['basemap']}.png"
    )


def plot_encoded_feature_panels(
    panel_data: list[tuple[str, np.ndarray, object]],
    encoded_features: int,
    selection_label: str,
    basemap: str,
    marker_size: float,
    output_path: Path,
    dpi: int,
) -> None:
    """Plot one US map per latent feature, optionally comparing multiple models."""
    n_models = len(panel_data)
    n_features = panel_data[0][1].shape[1]
    nrows = n_models
    ncols = n_features

    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(5.5 * ncols, 3.9 * nrows),
        squeeze=False,
    )
    us_states = load_us_states(required=False)
    fallback_polygons = [] if us_states is not None else load_state_polygons_local()

    all_feature_values = np.concatenate(
        [feature_values for _, feature_values, _ in panel_data],
        axis=0,
    )
    value_min = float(np.nanmin(all_feature_values))
    value_max = float(np.nanmax(all_feature_values))
    if not np.isfinite(value_min) or not np.isfinite(value_max) or value_max == value_min:
        scaled_panels = [
            (model_name, np.zeros_like(feature_values), locations)
            for model_name, feature_values, locations in panel_data
        ]
    else:
        scaled_panels = [
            (
                model_name,
                (feature_values - value_min) / (value_max - value_min),
                locations,
            )
            for model_name, feature_values, locations in panel_data
        ]
    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    scatter = None
    for model_idx, (model_name, scaled_feature_values, locations) in enumerate(scaled_panels):
        for feature_idx in range(n_features):
            ax = axs[model_idx, feature_idx]
            draw_background_map(ax, basemap, us_states, fallback_polygons)
            scatter = ax.scatter(
                x=locations["gauge_lon"],
                y=locations["gauge_lat"],
                c=scaled_feature_values[:, feature_idx],
                s=marker_size,
                cmap=cm.batlow,
                norm=norm,
                linewidths=0.0,
                alpha=0.95,
                zorder=3,
            )
            if model_idx == 0:
                ax.set_title(f"LF {feature_idx + 1}", fontsize=20)
            if feature_idx == 0:
                ax.text(
                    -0.06,
                    0.5,
                    model_name.upper(),
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=20,
                )

    if n_models > 1:
        fig.subplots_adjust(left=0.06, right=0.90, top=0.90, bottom=0.06, hspace=0.0, wspace=0.06)
        cbar_ax = fig.add_axes([0.92, 0.14, 0.018, 0.72])
    else:
        fig.subplots_adjust(left=0.06, right=0.90, top=0.88, bottom=0.08, hspace=0.08, wspace=0.06)
        cbar_ax = fig.add_axes([0.92, 0.18, 0.018, 0.64])

    colorbar = fig.colorbar(
        scatter,
        cax=cbar_ax,
        orientation="vertical",
    )
    colorbar.set_label("Scaled encoded feature value", fontsize=18)
    colorbar.ax.tick_params(labelsize=12)


    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    cfg = get_args()
    models = ["global", "pub"] if cfg["model"] == "both" else [cfg["model"]]
    panel_data = []
    selection_label = None
    for model in models:
        basins, feature_tensor = load_encoded_feature_tensor(
            model=model,
            encoded_features=cfg["encoded_features"],
        )
        feature_values, model_selection_label = reduce_feature_tensor(
            feature_tensor=feature_tensor,
            run=cfg["run"],
            with_ica=cfg["with_ica"],
            with_pca=cfg["with_pca"],
            model=model,
            basins=basins,
            encoded_features=cfg["encoded_features"],
        )
        if selection_label is None:
            selection_label = model_selection_label
        elif selection_label != model_selection_label:
            raise ValueError(
                "Selection labels differ across models, which should not happen for shared run/ICA/PCA settings."
            )
        locations = load_basin_locations(basins)
        panel_data.append((model, feature_values, locations))

    output_path = build_output_path(cfg, selection_label)

    plot_encoded_feature_panels(
        panel_data=panel_data,
        encoded_features=cfg["encoded_features"],
        selection_label=selection_label,
        basemap=cfg["basemap"],
        marker_size=cfg["marker_size"],
        output_path=output_path,
        dpi=cfg["dpi"],
    )

    print(f"Saved figure to {output_path}")
