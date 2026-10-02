import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cmcrameri import cm
from scipy.stats import spearmanr

from scripts.plot_encoded_feature_scatter import (
    ATTRIBUTE_NAMES,
    SIGNATURE_NAMES,
    align_feature_axes,
    align_feature_tensor_to_global_reference_signs,
    load_encoded_feature_tensor,
    load_split_file,
    reduce_feature_tensor,
    rotate_feature_tensor_to_reference_ica,
    rotate_feature_tensor_to_reference_pca,
)
from src.datautils import CLIM_ATTRS, GEOL_ATTRS, SOIL_ATTRS, TOPO_ATTRS, VEGE_ATTRS, load_attributes, load_signatures
from src.utils import clean_and_capitalize, str2bool

mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

DEFAULT_OUTPUT_DIR = Path("analysis/figures")
DEFAULT_STATS_DIR = Path("analysis/stats")
BASIN_ATTR_DB = Path("data/attributes.db")

VARIABLE_GROUPS = [
    ("Soil", SOIL_ATTRS),
    ("Climate", CLIM_ATTRS),
    ("Vegetation", VEGE_ATTRS),
    ("Topography", TOPO_ATTRS),
    ("Geology", GEOL_ATTRS),
    ("Hydrological signatures", SIGNATURE_NAMES),
]
VARIABLE_NAMES = [name for _, variables in VARIABLE_GROUPS for name in variables]
VARIABLE_GROUP_LOOKUP = {
    variable_name: group_name
    for group_name, variable_names in VARIABLE_GROUPS
    for variable_name in variable_names
}
LABEL_TO_VARIABLE = {
    clean_and_capitalize(variable_name).strip(): variable_name
    for variable_name in VARIABLE_NAMES
}
VARIABLE_TEXT_COLORS = {
    **{variable_name: "#2166AC" for variable_name in CLIM_ATTRS},
    **{variable_name: "#2166AC" for variable_name in SIGNATURE_NAMES},
    **{variable_name: "#1B7837" for variable_name in VEGE_ATTRS},
    **{variable_name: "#8C510A" for variable_name in SOIL_ATTRS},
    **{variable_name: "#762A83" for variable_name in TOPO_ATTRS},
}


def get_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute the Spearman correlation matrix between latent features and "
            "static attributes plus hydrological signatures for one or more models."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["global", "pub"],
        choices=["global", "pub"],
        help="Model families to analyse.",
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
            "Optional run index to analyse. For global, if omitted, the script aligns "
            "feature axes across runs and uses the mean value per basin. For pub, if "
            "omitted, split-wise correlations are aggregated across runs."
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
        "--eval_period",
        type=str,
        default="test",
        help="Signature file suffix, e.g. test -> analysis/signatures/camels_us_test.csv.",
    )
    parser.add_argument(
        "--annot",
        type=str2bool,
        default=True,
        help="Annotate heatmap cells with the correlation value.",
    )
    parser.add_argument(
        "--absolute",
        type=str2bool,
        default=True,
        help="Plot the absolute value of the Spearman correlation in the heatmap.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output figure DPI.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory for figures.",
    )
    parser.add_argument(
        "--nruns",
        type=int,
        default=4,
        help="Number of PUB runs when using split-aware PUB analysis.",
    )
    parser.add_argument(
        "--nsplits",
        type=int,
        default=12,
        help="Number of PUB splits per run when using split-aware PUB analysis.",
    )
    parser.add_argument(
        "--pub_splitwise",
        type=str2bool,
        default=True,
        help="If true, analyse PUB per split and average across splits. If false, analyse pooled PUB features directly.",
    )
    parser.add_argument(
        "--performance_threshold",
        type=float,
        default=float("-inf"),
        help=(
            "Optional NSE threshold applied using analysis/stats/test/global_esNone.csv. "
            "Default -inf keeps all basins."
        ),
    )
    return vars(parser.parse_args())


def load_attribute_frame(basins: list[str]) -> pd.DataFrame:
    """Load the 26 static attributes for the requested basins."""
    df_attr = load_attributes(BASIN_ATTR_DB, basins, keep_attributes=ATTRIBUTE_NAMES)
    df_attr = df_attr.loc[basins, ATTRIBUTE_NAMES]
    df_attr.index = [str(basin).zfill(8) for basin in df_attr.index]
    return df_attr


def load_signature_frame(basins: list[str], eval_period: str) -> pd.DataFrame:
    """Load CAMELS-US streamflow signatures for the requested basins."""
    signature_file = Path(f"analysis/signatures/camels_us_{eval_period}.csv")
    if not signature_file.is_file():
        raise FileNotFoundError(f"Missing signature file: {signature_file}")

    df_sig = load_signatures(signature_file)
    return df_sig.loc[basins, SIGNATURE_NAMES]


def load_variable_frame(basins: list[str], eval_period: str) -> pd.DataFrame:
    """Load static attributes and signatures for the requested basins."""
    df_attr = load_attribute_frame(basins)
    df_sig = load_signature_frame(basins, eval_period)
    df_var = pd.concat([df_attr, df_sig], axis=1)
    return df_var.loc[basins, VARIABLE_NAMES]


def filter_basins_by_performance(
    basins: list[str],
    feature_tensor: np.ndarray,
    threshold: float,
) -> tuple[list[str], np.ndarray]:
    """Filter basins using the GLOBAL-None NSE threshold."""
    if np.isneginf(threshold):
        return basins, feature_tensor

    stats_path = Path("analysis/stats/test/global_esNone.csv")
    df_stats = pd.read_csv(stats_path, index_col=0)
    df_stats.index = [str(basin).zfill(8) for basin in df_stats.index]
    keep_idx = [
        idx for idx, basin in enumerate(basins)
        if basin in df_stats.index and df_stats.loc[basin, "nse"] > threshold
    ]
    filtered_basins = [basins[idx] for idx in keep_idx]
    return filtered_basins, feature_tensor[keep_idx, ...]


def compute_spearman_matrices(
    feature_values: np.ndarray,
    variable_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute signed Spearman correlation and p-value matrices."""
    n_features = feature_values.shape[1]
    corr_matrix = np.full((len(VARIABLE_NAMES), n_features), np.nan, dtype=float)
    pvalue_matrix = np.full((len(VARIABLE_NAMES), n_features), np.nan, dtype=float)

    for variable_idx, variable_name in enumerate(VARIABLE_NAMES):
        x_values = variable_frame[variable_name].to_numpy(dtype=float)
        for feature_idx in range(n_features):
            y_values = feature_values[:, feature_idx]
            mask = np.isfinite(x_values) & np.isfinite(y_values)
            if np.count_nonzero(mask) < 2:
                continue

            corr, pvalue = spearmanr(x_values[mask], y_values[mask])
            corr_matrix[variable_idx, feature_idx] = corr
            pvalue_matrix[variable_idx, feature_idx] = pvalue

    row_names = [clean_and_capitalize(name).strip() for name in VARIABLE_NAMES]
    col_names = [f"LF {idx + 1}" for idx in range(n_features)]
    df_corr = pd.DataFrame(corr_matrix, index=row_names, columns=col_names)
    df_pvalue = pd.DataFrame(pvalue_matrix, index=row_names, columns=col_names)
    return df_corr, df_pvalue


def summarize_runwise_spearman_matrices(
    feature_tensor: np.ndarray,
    variable_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute mean correlation, standard error, and mean p-value across runs."""
    df_corr_runs = []
    df_pvalue_runs = []

    for run in range(feature_tensor.shape[1]):
        df_corr_run, df_pvalue_run = compute_spearman_matrices(
            feature_values=feature_tensor[:, run, :],
            variable_frame=variable_frame,
        )
        df_corr_runs.append(df_corr_run)
        df_pvalue_runs.append(df_pvalue_run)

    corr_stack = np.stack([df_corr.to_numpy(dtype=float) for df_corr in df_corr_runs], axis=0)
    pvalue_stack = np.stack([df_pvalue.to_numpy(dtype=float) for df_pvalue in df_pvalue_runs], axis=0)

    mean_corr = np.nanmean(corr_stack, axis=0)
    valid_counts = np.sum(np.isfinite(corr_stack), axis=0)
    std_corr = np.nanstd(corr_stack, axis=0, ddof=1)
    stderr_corr = np.full_like(mean_corr, np.nan, dtype=float)
    valid_mask = valid_counts > 1
    stderr_corr[valid_mask] = std_corr[valid_mask] / np.sqrt(valid_counts[valid_mask])

    mean_pvalue = np.nanmean(pvalue_stack, axis=0)

    df_mean = pd.DataFrame(mean_corr, index=df_corr_runs[0].index, columns=df_corr_runs[0].columns)
    df_stderr = pd.DataFrame(stderr_corr, index=df_corr_runs[0].index, columns=df_corr_runs[0].columns)
    df_pvalue = pd.DataFrame(mean_pvalue, index=df_corr_runs[0].index, columns=df_corr_runs[0].columns)
    return df_mean, df_stderr, df_pvalue


def build_mean_selection_label(with_ica: bool, with_pca: bool, pooled: bool = False) -> str:
    """Build the selection label for run-averaged latent features."""
    selection_label = "mean_ica" if with_ica else "mean"
    if with_pca:
        selection_label = f"{selection_label}_pca"
    if pooled:
        selection_label = f"{selection_label}_pooled"
    return selection_label


def prepare_feature_tensor(
    model: str,
    encoded_features: int,
    with_ica: bool,
    with_pca: bool,
) -> tuple[list[str], np.ndarray]:
    """Load one model feature tensor and prepare runs in a common basis."""
    basins, feature_tensor = load_encoded_feature_tensor(
        model=model,
        encoded_features=encoded_features,
    )
    feature_tensor = align_feature_axes(feature_tensor)
    if with_ica:
        feature_tensor = rotate_feature_tensor_to_reference_ica(feature_tensor)
    if with_pca:
        feature_tensor = rotate_feature_tensor_to_reference_pca(feature_tensor)
    if model == "pub":
        feature_tensor = align_feature_tensor_to_global_reference_signs(
            basins=basins,
            feature_tensor=feature_tensor,
            encoded_features=encoded_features,
            with_ica=with_ica,
            with_pca=with_pca,
        )
    return basins, feature_tensor


def compute_pub_split_correlations(
    basins: list[str],
    feature_tensor: np.ndarray,
    variable_frame: pd.DataFrame,
    runs: list[int],
    nsplits: int,
) -> pd.DataFrame:
    """Compute split-wise PUB Spearman correlations from the selected test-split tensor."""
    basin_to_index = {basin: idx for idx, basin in enumerate(basins)}
    rows = []

    for run in runs:
        splits = load_split_file(run)
        for split in range(nsplits):
            test_basins = [str(basin).zfill(8) for basin in splits[split]["test"]]
            selected_test_basins = [basin for basin in test_basins if basin in basin_to_index]
            basin_idx = [basin_to_index[basin] for basin in selected_test_basins]
            if len(basin_idx) == 0:
                continue
            split_features = feature_tensor[basin_idx, run, :]
            split_variables = variable_frame.loc[selected_test_basins]

            for variable_name in VARIABLE_NAMES:
                x_values = split_variables[variable_name].to_numpy(dtype=float)
                for feature_idx in range(split_features.shape[1]):
                    y_values = split_features[:, feature_idx]
                    mask = np.isfinite(x_values) & np.isfinite(y_values)
                    if np.count_nonzero(mask) < 2:
                        corr = np.nan
                        pvalue = np.nan
                    else:
                        corr, pvalue = spearmanr(x_values[mask], y_values[mask])

                    rows.append(
                        {
                            "run": run,
                            "split": split,
                            "variable": variable_name,
                            "group": VARIABLE_GROUP_LOOKUP[variable_name],
                            "feature": feature_idx + 1,
                            "rho": corr,
                            "abs_rho": np.abs(corr) if np.isfinite(corr) else np.nan,
                            "pvalue": pvalue,
                        }
                    )

    return pd.DataFrame(rows)


def summarize_pub_split_correlations(
    df_raw: pd.DataFrame,
    encoded_features: int,
) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    """Aggregate absolute PUB correlations across splits independently for each run."""
    row_names = [clean_and_capitalize(name).strip() for name in VARIABLE_NAMES]
    col_names = [f"LF {idx + 1}" for idx in range(encoded_features)]
    summaries = {}

    for run in sorted(df_raw["run"].dropna().unique()):
        df_run = df_raw[df_raw["run"] == run]
        mean_abs = np.full((len(VARIABLE_NAMES), encoded_features), np.nan, dtype=float)
        std_abs = np.full((len(VARIABLE_NAMES), encoded_features), np.nan, dtype=float)

        for variable_idx, variable_name in enumerate(VARIABLE_NAMES):
            for feature_idx in range(encoded_features):
                subset = df_run[
                    (df_run["variable"] == variable_name) &
                    (df_run["feature"] == feature_idx + 1)
                ]["abs_rho"].to_numpy(dtype=float)
                mean_abs[variable_idx, feature_idx] = np.nanmean(subset)
                std_abs[variable_idx, feature_idx] = np.nanstd(subset)

        df_mean = pd.DataFrame(mean_abs, index=row_names, columns=col_names)
        df_std = pd.DataFrame(std_abs, index=row_names, columns=col_names)
        summaries[int(run)] = (df_mean, df_std)

    return summaries


def summarize_pub_split_correlations_across_runs(
    run_summaries: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate split-wise PUB mean absolute correlations across runs."""
    if not run_summaries:
        raise ValueError("No PUB run summaries available to aggregate across runs.")

    df_means = [df_mean for df_mean, _ in run_summaries.values()]
    mean_stack = np.stack([df_mean.to_numpy(dtype=float) for df_mean in df_means], axis=0)
    aggregated_mean = np.nanmean(mean_stack, axis=0)

    valid_counts = np.sum(np.isfinite(mean_stack), axis=0)
    std_across_runs = np.nanstd(mean_stack, axis=0, ddof=1)
    stderr_across_runs = np.full_like(aggregated_mean, np.nan, dtype=float)
    valid_mask = valid_counts > 1
    stderr_across_runs[valid_mask] = (
        std_across_runs[valid_mask] / np.sqrt(valid_counts[valid_mask])
    )

    template = df_means[0]
    df_mean = pd.DataFrame(aggregated_mean, index=template.index, columns=template.columns)
    df_stderr = pd.DataFrame(stderr_across_runs, index=template.index, columns=template.columns)
    return df_mean, df_stderr


def build_output_paths(
    model: str,
    encoded_features: int,
    selection_label: str,
    eval_period: str,
    output_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    """Resolve output paths for one model."""
    stem = (
        f"encoded_attribute_signature_spearman_{model}_es{encoded_features}_"
        f"{selection_label}_{eval_period}"
    )
    figure_path = output_dir / f"{stem}.png"
    corr_path = DEFAULT_STATS_DIR / f"{stem}.csv"
    stderr_path = DEFAULT_STATS_DIR / f"{stem}_stderr.csv"
    pvalue_path = DEFAULT_STATS_DIR / f"{stem}_pvalues.csv"
    return figure_path, corr_path, stderr_path, pvalue_path


def build_pub_output_stem(cfg: dict) -> str:
    """Resolve the PUB split-aware output stem."""
    selection_label = "split_ica" if cfg["with_ica"] else "split"
    if cfg["with_pca"]:
        selection_label = f"{selection_label}_pca"
    return (
        f"encoded_attribute_signature_spearman_pub_es{cfg['encoded_features']}_"
        f"{selection_label}_{cfg['eval_period']}"
    )


def add_group_sublabels(ax, n_rows: int, with_group_labels: bool = True) -> None:
    """Draw vertical group separators and optional horizontal group labels above the columns."""
    boundary = 0
    for group_name, variable_names in VARIABLE_GROUPS:
        group_size = len(variable_names)
        center = boundary + group_size / 2.0
        if with_group_labels:
            ax.text(
                center,
                1.08,
                group_name,
                transform=ax.get_xaxis_transform(),
                rotation=0,
                ha="center",
                va="bottom",
                fontsize=15,
                fontweight="bold",
            )
        boundary += group_size
        if boundary < len(VARIABLE_NAMES):
            ax.plot(
                [boundary, boundary],
                [0.0, 1.15],
                transform=ax.get_xaxis_transform(),
                color="#111827",
                linewidth=2.0,
                solid_capstyle="butt",
                clip_on=False,
            )
            ax.plot(
                [boundary, boundary],
                [0.0, n_rows],
                color="#111827",
                linewidth=2.0,
                solid_capstyle="butt",
            )


def color_variable_ticklabels(ax) -> None:
    """Apply the repository's sensitivity-table colors to variable tick labels."""
    for tick_label in ax.get_xticklabels():
        variable_name = LABEL_TO_VARIABLE.get(tick_label.get_text().strip())
        color = VARIABLE_TEXT_COLORS.get(variable_name)
        if color is not None:
            tick_label.set_color(color)


def build_annotation_data(
    plot_data: pd.DataFrame,
    df_stderr_t: pd.DataFrame | None,
) -> np.ndarray | None:
    """Build heatmap annotation strings."""
    annot_data = np.empty(plot_data.shape, dtype=object)
    for row_idx in range(plot_data.shape[0]):
        for col_idx in range(plot_data.shape[1]):
            value = plot_data.iloc[row_idx, col_idx]
            stderr_value = None if df_stderr_t is None else df_stderr_t.iloc[row_idx, col_idx]
            if np.isnan(value):
                annot_data[row_idx, col_idx] = ""
            elif stderr_value is None or np.isnan(stderr_value):
                annot_data[row_idx, col_idx] = f"{value:.2f}"
            else:
                annot_data[row_idx, col_idx] = f"{value:.2f} ± {stderr_value:.2f}"
    return annot_data


def build_combined_output_path(
    cfg: dict,
    output_dir: Path,
    global_selection_label: str,
    pub_selection_label: str,
) -> Path:
    """Resolve output path for the combined GLOBAL/PUB figure."""
    stem = (
        f"encoded_attribute_signature_spearman_global_pub_es{cfg['encoded_features']}_"
        f"{global_selection_label}_{pub_selection_label}_{cfg['eval_period']}"
    )
    return output_dir / f"{stem}.png"


def plot_spearman_heatmap(
    df_corr: pd.DataFrame,
    df_stderr: pd.DataFrame | None,
    model: str,
    encoded_features: int,
    selection_label: str,
    eval_period: str,
    annot: bool,
    absolute: bool,
    output_path: Path,
    dpi: int,
) -> None:
    """Plot the grouped Spearman heatmap for attributes and signatures."""
    plot_data = df_corr.abs() if absolute else df_corr
    plot_data = plot_data.T
    fig_width = max(12.2, 0.40 * plot_data.shape[1] + 4.9)
    fig_height = max(5.8, 1.38 * plot_data.shape[0] + 3.2)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    annot_data = None
    if annot:
        annot_data = np.empty(plot_data.shape, dtype=object)
        for row_idx in range(plot_data.shape[0]):
            for col_idx in range(plot_data.shape[1]):
                value = plot_data.iloc[row_idx, col_idx]
                stderr_value = None if df_stderr is None else df_stderr.T.iloc[row_idx, col_idx]
                if np.isnan(value):
                    annot_data[row_idx, col_idx] = ""
                elif stderr_value is None or np.isnan(stderr_value):
                    annot_data[row_idx, col_idx] = f"{value:.2f}"
                else:
                    annot_data[row_idx, col_idx] = f"{value:.2f} ± {stderr_value:.2f}"

    cmap = cm.batlow if absolute else cm.vik
    vmin = 0.0 if absolute else -1.0
    vmax = 1.0
    heatmap_kwargs = {
        "data": plot_data,
        "ax": ax,
        "cmap": cmap,
        "vmin": vmin,
        "vmax": vmax,
        "annot": annot_data,
        "annot_kws": {"fontsize": 14.5, "rotation": 90, "ha": "center", "va": "center"},
        "fmt": "",
        "linewidths": 1,
        "linecolor": "#ffffff",
        "cbar_kws": {
            "label": "Absolute Spearman correlation" if absolute else "Spearman correlation",
            "fraction": 0.075,
            "pad": 0.02,
        },
    }
    if not absolute:
        heatmap_kwargs["center"] = 0.0

    sns.heatmap(**heatmap_kwargs)
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=13)
    colorbar.set_label(
        "Absolute Spearman correlation" if absolute else "Spearman correlation",
        fontsize=16,
    )
    add_group_sublabels(ax=ax, n_rows=plot_data.shape[0])

    title_prefix = "Absolute Spearman correlation" if absolute else "Spearman correlation"
    ax.set_xlabel("")
    ax.set_ylabel("Latent features", fontsize=16)
    fig.suptitle(
        f"{model.upper()}-{encoded_features} latent features vs attributes and signatures",
        fontsize=22,
        y=0.96,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=70, ha="right", fontsize=15)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=13)
    color_variable_ticklabels(ax)

    fig.subplots_adjust(left=0.10, right=0.98, top=0.82, bottom=0.28)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_pub_split_heatmap(
    df_mean: pd.DataFrame,
    df_stderr: pd.DataFrame,
    run: int | None,
    encoded_features: int,
    eval_period: str,
    with_ica: bool,
    with_pca: bool,
    annot: bool,
    output_path: Path,
    dpi: int,
) -> None:
    """Plot the mean absolute split-wise PUB heatmap for one run."""
    plot_data = df_mean.T
    fig_width = max(12.2, 0.40 * plot_data.shape[1] + 4.9)
    fig_height = max(5.8, 1.38 * plot_data.shape[0] + 3.2)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    annot_data = None
    if annot:
        annot_data = np.empty(plot_data.shape, dtype=object)
        for row_idx in range(plot_data.shape[0]):
            for col_idx in range(plot_data.shape[1]):
                mean_value = plot_data.iloc[row_idx, col_idx]
                stderr_value = df_stderr.T.iloc[row_idx, col_idx]
                if np.isnan(mean_value):
                    annot_data[row_idx, col_idx] = ""
                else:
                    annot_data[row_idx, col_idx] = f"{mean_value:.2f} ± {stderr_value:.2f}"

    sns.heatmap(
        plot_data,
        ax=ax,
        cmap=cm.batlow,
        vmin=0.0,
        vmax=1.0,
        annot=annot_data,
        annot_kws={"fontsize": 14.5, "rotation": 90, "ha": "center", "va": "center"},
        fmt="",
        linewidths=0.5,
        linecolor="#ffffff",
        cbar_kws={
            "label": "Absolute Spearman correlation",
            "fraction": 0.075,
            "pad": 0.02,
        },
    )
    colorbar = ax.collections[0].colorbar
    colorbar.ax.tick_params(labelsize=13)
    colorbar.set_label("Absolute Spearman correlation", fontsize=16)
    add_group_sublabels(ax=ax, n_rows=plot_data.shape[0])

    selection_label = "with ICA" if with_ica else "stored latent features"
    if with_pca:
        selection_label = f"{selection_label}, common PCA basis"
    ax.set_xlabel("")
    ax.set_ylabel("Latent features", fontsize=16)
    if run is None:
        title = f"PUB-{encoded_features} latent features vs attributes and signatures"
    else:
        title = f"PUB-{encoded_features} (run {run}) latent features vs attributes and signatures"
    fig.suptitle(title, fontsize=22, y=0.96)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=70, ha="right", fontsize=15)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=13)
    color_variable_ticklabels(ax)

    fig.subplots_adjust(left=0.10, right=0.98, top=0.82, bottom=0.28)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_combined_spearman_heatmaps(
    entries: list[dict],
    encoded_features: int,
    absolute: bool,
    output_path: Path,
    dpi: int,
) -> None:
    """Plot stacked GLOBAL/PUB heatmaps with one common colorbar and one x-label row."""
    plot_frames = []
    for entry in entries:
        plot_data = (entry["df_corr"].abs() if absolute else entry["df_corr"]).T
        plot_frames.append(plot_data)

    fig_width = max(12.2, 0.40 * plot_frames[0].shape[1] + 4.9)
    fig_height = max(9.6, 4.25 * len(entries))
    fig, axs = plt.subplots(
        len(entries),
        1,
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharex=True,
    )
    axes = axs[:, 0]

    cmap = cm.batlow if absolute else cm.vik
    vmin = 0.0 if absolute else -1.0
    vmax = 1.0
    cbar_ax = fig.add_axes([0.92, 0.16, 0.018, 0.68])

    for idx, (ax, entry, plot_data) in enumerate(zip(axes, entries, plot_frames)):
        df_stderr_t = None if entry["df_stderr"] is None else entry["df_stderr"].T
        annot_data = build_annotation_data(plot_data, df_stderr_t) if entry["annot"] else None
        heatmap_kwargs = {
            "data": plot_data,
            "ax": ax,
            "cmap": cmap,
            "vmin": vmin,
            "vmax": vmax,
            "annot": annot_data,
            "annot_kws": {"fontsize": 14.5, "rotation": 90, "ha": "center", "va": "center"},
            "fmt": "",
            "linewidths": 1,
            "linecolor": "#ffffff",
            "cbar": idx == len(entries) - 1,
            "cbar_ax": cbar_ax if idx == len(entries) - 1 else None,
            "cbar_kws": {
                "label": "Absolute Spearman correlation" if absolute else "Spearman correlation"
            } if idx == len(entries) - 1 else None,
        }
        if not absolute:
            heatmap_kwargs["center"] = 0.0

        sns.heatmap(**heatmap_kwargs)
        add_group_sublabels(
            ax=ax,
            n_rows=plot_data.shape[0],
            with_group_labels=(idx == 0),
        )
        ax.set_ylabel(entry["model"].upper(), fontsize=18)
        ax.set_xlabel("")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=15)

        if idx < len(entries) - 1:
            ax.tick_params(axis="x", labelbottom=False, bottom=False)
        else:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=70, ha="right", fontsize=18)
            color_variable_ticklabels(ax)

    colorbar = axes[-1].collections[0].colorbar
    colorbar.ax.tick_params(labelsize=15)
    colorbar.set_label(
        "Absolute Spearman correlation" if absolute else "Spearman correlation",
        fontsize=18,
    )

    # fig.suptitle(
    #     f"GLOBAL-{encoded_features} and PUB-{encoded_features} latent features vs attributes and signatures",
    #     fontsize=24,
    #     y=0.985,
    # )
    #fig.supylabel("Latent features", fontsize=18, x=0.02)
    fig.subplots_adjust(left=0.10, right=0.90, top=0.87, bottom=0.22, hspace=0.10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    cfg = get_args()
    if cfg["output_dir"] is not None:
        output_dir = Path(cfg["output_dir"])
    else:
        output_dir = DEFAULT_OUTPUT_DIR
    compare_models = set(cfg["models"]) == {"global", "pub"} and len(cfg["models"]) == 2
    combined_entries = []

    for model in cfg["models"]:
        if model == "pub":
            if not cfg["pub_splitwise"]:
                basins, feature_tensor = prepare_feature_tensor(
                    model="pub",
                    encoded_features=cfg["encoded_features"],
                    with_ica=cfg["with_ica"],
                    with_pca=cfg["with_pca"],
                )
                basins, feature_tensor = filter_basins_by_performance(
                    basins=basins,
                    feature_tensor=feature_tensor,
                    threshold=cfg["performance_threshold"],
                )
                variable_frame = load_variable_frame(basins, cfg["eval_period"])
                if cfg["run"] is None:
                    feature_values = np.nanmean(feature_tensor, axis=1)
                    df_corr, df_pvalue = compute_spearman_matrices(feature_values, variable_frame)
                    _, df_stderr, _ = summarize_runwise_spearman_matrices(
                        feature_tensor=feature_tensor,
                        variable_frame=variable_frame,
                    )
                    selection_label = build_mean_selection_label(
                        with_ica=cfg["with_ica"],
                        with_pca=cfg["with_pca"],
                        pooled=True,
                    )
                else:
                    feature_values, selection_label = reduce_feature_tensor(
                        feature_tensor=feature_tensor,
                        run=cfg["run"],
                        with_ica=False,
                        with_pca=False,
                        model=model,
                        basins=basins,
                        encoded_features=cfg["encoded_features"],
                    )
                    selection_label = f"{selection_label}_pooled"
                    df_corr, df_pvalue = compute_spearman_matrices(feature_values, variable_frame)
                    df_stderr = None

                figure_path, corr_path, stderr_path, pvalue_path = build_output_paths(
                    model=model,
                    encoded_features=cfg["encoded_features"],
                    selection_label=selection_label,
                    eval_period=cfg["eval_period"],
                    output_dir=output_dir,
                )
                corr_path.parent.mkdir(parents=True, exist_ok=True)
                df_corr.to_csv(corr_path)
                if df_stderr is not None:
                    df_stderr.to_csv(stderr_path)
                df_pvalue.to_csv(pvalue_path)

                if compare_models:
                    combined_entries.append(
                        {
                            "model": model,
                            "df_corr": df_corr,
                            "df_stderr": df_stderr,
                            "selection_label": selection_label,
                            "absolute": cfg["absolute"],
                            "annot": cfg["annot"],
                        }
                    )
                else:
                    plot_spearman_heatmap(
                        df_corr=df_corr,
                        df_stderr=df_stderr,
                        model=model,
                        encoded_features=cfg["encoded_features"],
                        selection_label=selection_label,
                        eval_period=cfg["eval_period"],
                        annot=cfg["annot"],
                        absolute=cfg["absolute"],
                        output_path=figure_path,
                        dpi=cfg["dpi"],
                    )

                print(f"Saved pooled pub correlation matrix to {corr_path}")
                if df_stderr is not None:
                    print(f"Saved pooled pub standard-error matrix to {stderr_path}")
                print(f"Saved pooled pub p-value matrix to {pvalue_path}")
                if not compare_models:
                    print(f"Saved pooled pub figure to {figure_path}")
                continue

            pub_basins, pub_feature_tensor = prepare_feature_tensor(
                model="pub",
                encoded_features=cfg["encoded_features"],
                with_ica=cfg["with_ica"],
                with_pca=cfg["with_pca"],
            )
            pub_basins, pub_feature_tensor = filter_basins_by_performance(
                basins=pub_basins,
                feature_tensor=pub_feature_tensor,
                threshold=cfg["performance_threshold"],
            )
            variable_frame = load_variable_frame(pub_basins, cfg["eval_period"])
            if cfg["run"] is None:
                runs = list(range(cfg["nruns"]))
            else:
                runs = [cfg["run"]]

            df_raw = compute_pub_split_correlations(
                basins=pub_basins,
                feature_tensor=pub_feature_tensor,
                variable_frame=variable_frame,
                runs=runs,
                nsplits=cfg["nsplits"],
            )
            run_summaries = summarize_pub_split_correlations(
                df_raw=df_raw,
                encoded_features=cfg["encoded_features"],
            )

            stem = build_pub_output_stem(cfg)
            raw_path = DEFAULT_STATS_DIR / f"{stem}_raw.csv"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            df_raw.to_csv(raw_path, index=False)
            print(f"Saved pub raw split-wise correlations to {raw_path}")

            if cfg["run"] is None:
                df_mean, df_stderr = summarize_pub_split_correlations_across_runs(run_summaries)
                figure_path = output_dir / f"{stem}.png"
                mean_path = DEFAULT_STATS_DIR / f"{stem}.csv"
                stderr_path = DEFAULT_STATS_DIR / f"{stem}_stderr.csv"
                df_mean.to_csv(mean_path)
                df_stderr.to_csv(stderr_path)

                if compare_models:
                    combined_entries.append(
                        {
                            "model": model,
                            "df_corr": df_mean,
                            "df_stderr": df_stderr,
                            "selection_label": "split",
                            "absolute": True,
                            "annot": cfg["annot"],
                        }
                    )
                else:
                    plot_pub_split_heatmap(
                        df_mean=df_mean,
                        df_stderr=df_stderr,
                        run=None,
                        encoded_features=cfg["encoded_features"],
                        eval_period=cfg["eval_period"],
                        with_ica=cfg["with_ica"],
                        with_pca=cfg["with_pca"],
                        annot=cfg["annot"],
                        output_path=figure_path,
                        dpi=cfg["dpi"],
                    )

                print(f"Saved pub mean absolute matrix to {mean_path}")
                print(f"Saved pub standard-error matrix to {stderr_path}")
                if not compare_models:
                    print(f"Saved pub figure to {figure_path}")
            else:
                run = cfg["run"]
                df_mean, df_std = run_summaries[run]
                figure_path = output_dir / f"{stem}_run{run}.png"
                mean_path = DEFAULT_STATS_DIR / f"{stem}_run{run}_mean_abs.csv"
                std_path = DEFAULT_STATS_DIR / f"{stem}_run{run}_std_abs.csv"
                df_mean.to_csv(mean_path)
                df_std.to_csv(std_path)

                if compare_models:
                    combined_entries.append(
                        {
                            "model": model,
                            "df_corr": df_mean,
                            "df_stderr": df_std,
                            "selection_label": f"split_run{run}",
                            "absolute": True,
                            "annot": cfg["annot"],
                        }
                    )
                else:
                    plot_pub_split_heatmap(
                        df_mean=df_mean,
                        df_stderr=df_std,
                        run=run,
                        encoded_features=cfg["encoded_features"],
                        eval_period=cfg["eval_period"],
                        with_ica=cfg["with_ica"],
                        with_pca=cfg["with_pca"],
                        annot=cfg["annot"],
                        output_path=figure_path,
                        dpi=cfg["dpi"],
                    )

                print(f"Saved pub run {run} mean absolute matrix to {mean_path}")
                print(f"Saved pub run {run} std absolute matrix to {std_path}")
                if not compare_models:
                    print(f"Saved pub run {run} figure to {figure_path}")
            continue

        basins, feature_tensor = prepare_feature_tensor(
            model=model,
            encoded_features=cfg["encoded_features"],
            with_ica=cfg["with_ica"],
            with_pca=cfg["with_pca"],
        )
        basins, feature_tensor = filter_basins_by_performance(
            basins=basins,
            feature_tensor=feature_tensor,
            threshold=cfg["performance_threshold"],
        )
        variable_frame = load_variable_frame(basins, cfg["eval_period"])
        if cfg["run"] is None:
            feature_values = np.nanmean(feature_tensor, axis=1)
            df_corr, df_pvalue = compute_spearman_matrices(feature_values, variable_frame)
            _, df_stderr, _ = summarize_runwise_spearman_matrices(
                feature_tensor=feature_tensor,
                variable_frame=variable_frame,
            )
            selection_label = build_mean_selection_label(
                with_ica=cfg["with_ica"],
                with_pca=cfg["with_pca"],
            )
        else:
            feature_values, selection_label = reduce_feature_tensor(
                feature_tensor=feature_tensor,
                run=cfg["run"],
                with_ica=False,
                with_pca=False,
                model=model,
                basins=basins,
                encoded_features=cfg["encoded_features"],
            )
            df_corr, df_pvalue = compute_spearman_matrices(feature_values, variable_frame)
            df_stderr = None

        figure_path, corr_path, stderr_path, pvalue_path = build_output_paths(
            model=model,
            encoded_features=cfg["encoded_features"],
            selection_label=selection_label,
            eval_period=cfg["eval_period"],
            output_dir=output_dir,
        )
        corr_path.parent.mkdir(parents=True, exist_ok=True)
        df_corr.to_csv(corr_path)
        if df_stderr is not None:
            df_stderr.to_csv(stderr_path)
        df_pvalue.to_csv(pvalue_path)

        if compare_models:
            combined_entries.append(
                {
                    "model": model,
                    "df_corr": df_corr,
                    "df_stderr": df_stderr,
                    "selection_label": selection_label,
                    "absolute": cfg["absolute"],
                    "annot": cfg["annot"],
                }
            )
        else:
            plot_spearman_heatmap(
                df_corr=df_corr,
                df_stderr=df_stderr,
                model=model,
                encoded_features=cfg["encoded_features"],
                selection_label=selection_label,
                eval_period=cfg["eval_period"],
                annot=cfg["annot"],
                absolute=cfg["absolute"],
                output_path=figure_path,
                dpi=cfg["dpi"],
            )

        print(f"Saved {model} correlation matrix to {corr_path}")
        if df_stderr is not None:
            print(f"Saved {model} standard-error matrix to {stderr_path}")
        print(f"Saved {model} p-value matrix to {pvalue_path}")
        if not compare_models:
            print(f"Saved {model} figure to {figure_path}")

    if compare_models:
        entry_by_model = {entry["model"]: entry for entry in combined_entries}
        if set(entry_by_model) == {"global", "pub"}:
            if entry_by_model["global"]["absolute"] != entry_by_model["pub"]["absolute"]:
                raise ValueError(
                    "Cannot build a combined GLOBAL/PUB heatmap when one panel is signed and the other is absolute."
                )
            figure_path = build_combined_output_path(
                cfg=cfg,
                output_dir=output_dir,
                global_selection_label=entry_by_model["global"]["selection_label"],
                pub_selection_label=entry_by_model["pub"]["selection_label"],
            )
            plot_combined_spearman_heatmaps(
                entries=[entry_by_model["global"], entry_by_model["pub"]],
                encoded_features=cfg["encoded_features"],
                absolute=entry_by_model["global"]["absolute"],
                output_path=figure_path,
                dpi=cfg["dpi"],
            )
            print(f"Saved combined GLOBAL/PUB figure to {figure_path}")
