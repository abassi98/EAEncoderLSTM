import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm

from scripts.plot_encoded_feature_scatter import (
    TARGET_NAMES,
    load_encoded_feature_tensor,
    load_target_series,
    reduce_feature_tensor,
)
from src.utils import clean_and_capitalize, compute_grid, str2bool

mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

DEFAULT_OUTPUT_DIR = Path("analysis/figures")


def get_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot a 2D projection of the encoded basin space for one model and "
            "color points by one static attribute or streamflow signature."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="global",
        choices=["global", "pub"],
        help="Model family to plot.",
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--target",
        type=str,
        default=None,
        help="One coloring variable chosen among static attributes and streamflow signatures.",
    )
    target_group.add_argument(
        "--targets",
        nargs="+",
        default=None,
        help="One or more coloring variables chosen among static attributes and streamflow signatures.",
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
        default=50.0,
        help="Marker size for basin points.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.9,
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
        help="Optional output path. Defaults to analysis/figures/encoded_space_2d_*.png.",
    )
    return vars(parser.parse_args())


def validate_cfg(cfg: dict) -> dict:
    """Validate target names and fix the latent dimension to 2."""
    targets = [cfg["target"]] if cfg["target"] is not None else cfg["targets"]
    for target in targets:
        if target not in TARGET_NAMES:
            raise ValueError(
                f"Unknown target '{target}'. Choose one of: {', '.join(TARGET_NAMES)}"
            )
    cfg["targets"] = targets
    cfg["encoded_features"] = 2
    return cfg


def build_output_path(cfg: dict, selection_label: str, target_kind: str) -> Path:
    """Resolve output path for the figure."""
    if cfg["output"] is not None:
        return Path(cfg["output"])

    if len(cfg["targets"]) == 1:
        target_label = cfg["targets"][0]
    else:
        target_label = f"{len(cfg['targets'])}targets"

    return DEFAULT_OUTPUT_DIR / (
        f"encoded_space_2d_{cfg['model']}_es{cfg['encoded_features']}_{target_kind}_"
        f"{target_label}_{selection_label}.png"
    )


def get_axis_label(with_pca: bool, dim_index: int) -> str:
    """Build a latent-axis label."""
    prefix = "PC" if with_pca else "LF"
    return f"{prefix} {dim_index + 1}"


def plot_encoded_space_2d(
    feature_values: np.ndarray,
    target_frames: list[tuple[str, str, np.ndarray]],
    model: str,
    encoded_features: int,
    selection_label: str,
    with_pca: bool,
    marker_size: float,
    alpha: float,
    output_path: Path,
    dpi: int,
) -> None:
    """Plot a 2D latent-space scatter colored by one or more variables."""
    x_idx = 0
    y_idx = 1
    x_values = feature_values[:, x_idx]
    y_values = feature_values[:, y_idx]
    n_targets = len(target_frames)
    nrows, ncols = compute_grid(n_targets)

    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(7.2 * ncols, 5.8 * nrows),
        squeeze=False,
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    axes = axs.ravel()

    for target_idx, ax in enumerate(axes):
        if target_idx >= n_targets:
            ax.set_visible(False)
            continue

        target_name, target_kind, target_values = target_frames[target_idx]
        mask = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(target_values)
        x_plot = x_values[mask]
        y_plot = y_values[mask]
        c_plot = target_values[mask]

        scatter = ax.scatter(
            x_plot,
            y_plot,
            c=c_plot,
            s=marker_size,
            cmap=cm.batlow,
            alpha=alpha,
            linewidths=0.0,
        )

        ax.grid(alpha=0.3)
    
        ax.set_title(clean_and_capitalize(target_name).strip(), fontsize=22)

        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.05, pad=0.03)
        #colorbar.set_label(target_kind.capitalize(), fontsize=14)
        colorbar.ax.tick_params(labelsize=15)

    fig.supxlabel(get_axis_label(with_pca, x_idx), fontsize=25)
    fig.supylabel(get_axis_label(with_pca, y_idx), fontsize=25)
    # fig.suptitle(
    #     f"{model.upper()} - {encoded_features}",
    #     fontsize=22,
    # )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    cfg = validate_cfg(get_args())

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
    target_frames = []
    target_kinds = set()
    for target_name in cfg["targets"]:
        target_series, target_kind = load_target_series(basins, target_name)
        target_frames.append((target_name, target_kind, target_series.to_numpy(dtype=float)))
        target_kinds.add(target_kind)

    target_kind_label = "mixed" if len(target_kinds) > 1 else next(iter(target_kinds))
    output_path = build_output_path(cfg, selection_label, target_kind_label)

    plot_encoded_space_2d(
        feature_values=feature_values,
        target_frames=target_frames,
        model=cfg["model"],
        encoded_features=cfg["encoded_features"],
        selection_label=selection_label,
        with_pca=cfg["with_pca"],
        marker_size=cfg["marker_size"],
        alpha=cfg["alpha"],
        output_path=output_path,
        dpi=cfg["dpi"],
    )

    print(f"Saved figure to {output_path}")
