
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.plot_utils import get_bootstrap_plot_models
from scripts.compute_leave_one_out import load_leave_one_out


mpl.rcParams["xtick.labelsize"] = 20
mpl.rcParams["ytick.labelsize"] = 20
mpl.rc("text", usetex=True)
mpl.rc("text.latex", preamble=r"\usepackage{amsmath} \usepackage{amsfonts}")
mpl.rcParams["font.serif"] = "Times New Roman"
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["mathtext.fontset"] = "dejavuserif"


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_period", choices=["test", "val"], default="test")
    parser.add_argument("--metric", default="nse")
    parser.add_argument("--xmin", type=float, default=0.4)
    parser.add_argument("--xmax", type=float, default=1.0)
    parser.add_argument(
        "--leave_one_out", action="store_true",
        help="Include full + leave-one-run-out sensitivity envelopes.",
    )
    args = parser.parse_args()
    if not args.xmin < args.xmax:
        parser.error("--xmin must be smaller than --xmax")
    return args


def plot_sensitivity(ax, scores, x_grid, color, label, **line_style):
    """Envelope whole cumulative curves, not per-basin minimum/maximum scores."""
    curves = []
    for column in scores.columns:
        sample = scores[column]
        if sample.nunique() == 1:
            # A constant distribution has no KDE bandwidth; use its exact CDF.
            curve = (x_grid >= sample.iloc[0]).astype(float)
        else:
            sns.kdeplot(sample, ax=ax, cumulative=True, bw_adjust=0.4,
                        common_norm=False, alpha=0, legend=False)
            helper = ax.lines[-1]
            curve = np.interp(x_grid, helper.get_xdata(), helper.get_ydata(),
                              left=0, right=1)
            helper.remove()
        curves.append(curve)
    curves = np.stack(curves)
    ax.fill_between(x_grid, curves.min(axis=0), curves.max(axis=0),
                    color=color, alpha=0.2, linewidth=0, zorder=1)
    ax.plot(x_grid, curves[scores.columns.get_loc("full")],
            color=color, label=label, **line_style)


def main():
    args = get_args()
    model_specs = get_bootstrap_plot_models()
    metric_values = {}
    sensitivity = {}
    if args.leave_one_out:
        sensitivity = load_leave_one_out(model_specs, args.eval_period, args.metric)
        metric_values = {label: scores["full"] for label, scores in sensitivity.items()}
    else:
        for spec in model_specs:
            stats_path = Path("analysis/stats") / args.eval_period / (
                f"{spec['file_experiment']}_es{spec['encoded_features']}.csv"
            )
            stats = pd.read_csv(stats_path, dtype={"basin": str}, index_col="basin")
            metric_values[spec["label"]] = stats[args.metric]

    # Compare the same basins for every model, retaining the full NSE range
    # when fitting the distributions (xmin/xmax only control the displayed view).
    values = pd.DataFrame(metric_values).replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 2:
        raise ValueError("At least two common basins with finite metrics are required.")
    print(f"Using {len(values)} common basins for the {args.eval_period} period.")

    fig, axes = plt.subplots(
        1, 2, figsize=(10, 5), constrained_layout=True, sharex=True, sharey=True
    )
    x_grid = np.linspace(args.xmin, args.xmax, 500)
    for ax, family in zip(axes, ("pub", "global")):
        reference = "GLOBAL-A" if family == "pub" else "PUB-A"
        if reference in values and args.leave_one_out:
            plot_sensitivity(ax, sensitivity[reference], x_grid, "0.2", reference,
                             linestyle="--", linewidth=1.5, alpha=0.7, zorder=1)
        elif reference in values:
            sns.kdeplot(
                values[reference],
                ax=ax,
                cumulative=True,
                bw_adjust=0.4,
                common_norm=False,
                color="0.2",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label=reference,
                zorder=1,
            )

        for spec in model_specs:
            if spec["family"] != family:
                continue
            if args.leave_one_out:
                plot_sensitivity(ax, sensitivity[spec["label"]], x_grid,
                                 spec["color"], spec["label"], linewidth=2.0, zorder=2)
                continue
            sns.kdeplot(
                values[spec["label"]],
                ax=ax,
                cumulative=True,
                bw_adjust=0.4,
                common_norm=False,
                color=spec["color"],
                label=spec["label"],
                linewidth=2.0,
                zorder=2,
            )

        ax.grid()
        ax.set_title(f"{family.upper()} models", fontsize=22, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xlim(args.xmin, args.xmax)
        ax.legend()

    period = "Test" if args.eval_period == "test" else "Validation"
    fig.supxlabel(f"{period} {args.metric.upper()}", fontweight="bold", fontsize=25)
    suffix = "_leave_one_out" if args.leave_one_out else ""
    output = Path("analysis/figures") / (
        f"ensemble_{args.metric}_cumulative_{args.eval_period}{suffix}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
