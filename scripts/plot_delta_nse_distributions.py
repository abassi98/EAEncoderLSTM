
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cmcrameri import cm


mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "dejavuserif"

EXPERIMENTS = {
    "global": [0, 1, 2, 4, 26],
    "pub": [0, 1, 2],
}


def get_args():
    parser = argparse.ArgumentParser(
        description="Plot global and pub successive-model delta-NSE distributions."
    )
    parser.add_argument("--eval_period", default="test")
    return parser.parse_args()


def load_delta_nse(experiment, encoded_features, eval_period):
    """Compute per-basin NSE changes between consecutive encoded-space sizes."""
    delta_nse_by_pair = {}
    stats_dir = Path("analysis") / "stats" / eval_period

    for previous, current in zip(encoded_features, encoded_features[1:]):
        current_stats = pd.read_csv(
            stats_dir / f"{experiment}_es{current}.csv", index_col=0
        )
        previous_stats = pd.read_csv(
            stats_dir / f"{experiment}_es{previous}.csv", index_col=0
        )
        current_stats.index = current_stats.index.map(lambda basin: str(basin).rjust(8, "0"))
        previous_stats.index = previous_stats.index.map(lambda basin: str(basin).rjust(8, "0"))
        delta_nse_by_pair[rf"$\Delta$NSE {current}-{previous}"] = (
            current_stats["nse"] - previous_stats["nse"]
        )

    return pd.DataFrame(delta_nse_by_pair)


def to_long(delta_nse_df):
    return (
        delta_nse_df.rename_axis("basin")
        .reset_index()
        .melt(id_vars="basin", var_name="transition", value_name="delta_nse")
        .dropna()
    )


def plot_distributions(ax, delta_nse_long, transitions, title, y_limit):
    palette = [cm.roma_r(value) for value in np.linspace(0.15, 0.85, len(transitions))]
    sns.violinplot(
        data=delta_nse_long,
        x="transition",
        y="delta_nse",
        order=transitions,
        inner=None,
        cut=0,
        linewidth=1.2,
        palette=palette,
        ax=ax,
    )
    sns.boxplot(
        data=delta_nse_long,
        x="transition",
        y="delta_nse",
        order=transitions,
        width=0.22,
        whis=(5, 95),
        showfliers=False,
        boxprops={"facecolor": "white", "zorder": 3},
        medianprops={"color": "black", "linewidth": 2},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
        ax=ax,
    )

    positive_fraction = delta_nse_long.groupby("transition")["delta_nse"].apply(
        lambda values: (values > 0).mean()
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_ylim(-y_limit, y_limit)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(title, fontsize=24)
    
    ax.tick_params(axis="x", rotation=15, labelsize=14)

    y_text = y_limit * 0.92
    for idx, transition in enumerate(transitions):
        ax.text(
            idx,
            y_text,
            f"{positive_fraction[transition]:.0%} > 0",
            ha="center",
            va="top",
            fontsize=12,
        )


def main():
    args = get_args()
    delta_nse = {
        experiment: load_delta_nse(experiment, encoded_features, args.eval_period)
        for experiment, encoded_features in EXPERIMENTS.items()
    }
    delta_nse_long = {experiment: to_long(values) for experiment, values in delta_nse.items()}

    all_values = pd.concat(delta_nse_long.values(), ignore_index=True)["delta_nse"]
    y_limit = max(abs(all_values.quantile(0.01)), abs(all_values.quantile(0.99)))
    if y_limit == 0:
        y_limit = 0.01

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    for ax, (experiment, values) in zip(axes, delta_nse.items()):
        plot_distributions(
            ax,
            delta_nse_long[experiment],
            values.columns.tolist(),
            experiment.upper(),
            y_limit,
        )
    fig.supylabel(r"$\Delta$ NSE", fontsize=20)
    fig.supxlabel("Successive model transition", fontsize=18)
    fig.tight_layout(rect=(0.03, 0, 1, 1))
    output_path = Path("analysis") / "figures" / f"delta_nse_distribution_global_pub_{args.eval_period}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)


if __name__ == "__main__":
    main()
