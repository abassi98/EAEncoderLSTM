import argparse
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plot_utils import cmodels

mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
mpl.font_manager.findSystemFonts(fontpaths=None, fontext="ttf")[:10]
mpl.rc("text", usetex=True)
mpl.rc("text.latex", preamble=r"\usepackage{amsmath} \usepackage{amsfonts}")

plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

METRIC_CHOICES = [
    "nse",
    "bias",
    "stdev_rat",
    "abs_nse",
    "sqrt_nse",
    "FHV",
    "FLV",
    "kge",
    "FMM",
    "bias_RR",
    "log_stdev",
    "r_coeff",
    "bias_std",
    "skew_rat",
    "kurt_rat",
]

YLIMS = {
    "nse": (0.0, 1.0),
    "kge": (0.2, 1.0),
    "bias": (-0.5, 0.5),
    "stdev_rat": (0.0, 1.5),
    "abs_nse": (0.0, 1.0),
    "sqrt_nse": (0.0, 1.0),
    "FHV": (-1.0, 1.0),
    "FLV": (-1.0, 1.0),
    "FMM": (-1.0, 1.0),
    "bias_RR": (-1.0, 1.0),
    "log_stdev": (-1.0, 1.0),
    "r_coeff": (0.5, 1.0),
    "bias_std": (-1.0, 1.0),
    "skew_rat": (0.0, 3.0),
    "kurt_rat": (0.0, 5.0),
}

METRIC_LABELS = {
    "r_coeff": "R",
    "stdev_rat": "SD",
}

METRIC_COLUMNS = {
    "stdev_rat": "stdev",
}


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval_period",
        type=str,
        default="test",
    )

    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        required=True,
        choices=METRIC_CHOICES,
    )

    parser.add_argument(
        "--figsize_scale",
        type=float,
        default=5.0,
        help="Scale factor for each subplot cell.",
    )

    return vars(parser.parse_args())


def compute_grid(npanels):
    ncols = math.ceil(math.sqrt(npanels))
    nrows = math.ceil(npanels / ncols)
    return nrows, ncols


def get_model_label(experiment, encoded_features):
    experiment_name = experiment.replace("_", " ").upper().replace("AE", "").strip()
    if encoded_features is None:
        return f"{experiment_name}-A"
    return f"{experiment_name}-{encoded_features}"


def update_metric_df(df, experiment, eval_period, encoded_features, metric):
    stats = pd.read_csv(
        f"analysis/stats/{eval_period}/{experiment}_es{encoded_features}.csv",
        sep=",",
        index_col=0,
    )
    stats.index = [str(s).rjust(8, "0") for s in stats.index]
    metric_column = METRIC_COLUMNS.get(metric, metric)
    df[get_model_label(experiment, encoded_features)] = stats[metric_column]


def get_metric_label(metric):
    return METRIC_LABELS.get(metric, metric.replace("_", " ").upper())


def load_metric_long_df(metric, eval_period):
    df_metric = pd.DataFrame()
    palette = {}
    model_order = []

    for experiment, encoded_features_dict in cmodels.items():
        for encoded_features, color in encoded_features_dict.items():
            update_metric_df(
                df_metric,
                experiment,
                eval_period,
                encoded_features,
                metric=metric,
            )
            model_label = get_model_label(experiment, encoded_features)
            palette[model_label] = color
            model_order.append(model_label)

    return df_metric[model_order], model_order, palette


if __name__ == "__main__":
    cfg = get_args()
    eval_period = cfg["eval_period"]
    metrics = cfg["metrics"]
    nrows, ncols = compute_grid(len(metrics))

    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(cfg["figsize_scale"] * ncols, cfg["figsize_scale"] * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    axs = axs.flatten()

    for idx, metric in enumerate(metrics):
        df_metric, model_order, palette = load_metric_long_df(metric, eval_period)
        ax = axs[idx]
        ax.grid(True, axis="y", alpha=0.3)
        box_data = [df_metric[col].dropna().values for col in model_order]

        bp = ax.boxplot(
            box_data,
            tick_labels=model_order,
            patch_artist=True,
            showfliers=True,
            widths=0.65,
            flierprops={
                "marker": "o",
                "markersize": 3,
                "markerfacecolor": "black",
                "markeredgecolor": "black",
                "alpha": 0.5,
            },
            medianprops={"color": "black", "linewidth": 1.5},
            boxprops={"linewidth": 1.2},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
        )

        for patch, model_label in zip(bp["boxes"], model_order):
            patch.set_facecolor(palette[model_label])
            patch.set_alpha(0.9)

        ax.set_title(get_metric_label(metric), fontsize=18, fontweight="bold")
        if metric in YLIMS:
            ax.set_ylim(*YLIMS[metric])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=45)

    for ax in axs[len(metrics):]:
        ax.axis("off")

    fig.supylabel("Metric value", fontsize=18)
    fig.supxlabel("Model", fontsize=18)
    fig.savefig(f"analysis/figures/boxplots_{eval_period}.png", dpi=300)
