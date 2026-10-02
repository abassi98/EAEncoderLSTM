import argparse
import os
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from src.plot_utils import cmodels

mpl.rcParams["xtick.labelsize"] = 16
mpl.rcParams["ytick.labelsize"] = 16
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

DEFAULT_INPUT = Path("analysis/stats/val_loss.csv")
DEFAULT_OUTPUT = Path("analysis/figures/validation_loss_global2_global26.png")


def get_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot the validation-loss curves stored in analysis/stats/val_loss.csv for the "
            "GLOBAL-2 and GLOBAL-26 models."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Path to the CSV file containing validation-loss histories.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Path where the figure will be saved.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output figure DPI.",
    )
    return parser.parse_args()


def find_loss_column(columns: list[str], model_token: str) -> str:
    """Find one validation-loss column matching an exact model token."""
    matches = [
        col for col in columns
        if col.lower().split(" - ")[0] == model_token.lower()
        and "val_loss_epoch" in col
        and "__" not in col
    ]
    if not matches:
        raise ValueError(f"Could not find a validation-loss column matching '{model_token}'.")
    if len(matches) > 1:
        raise ValueError(
            f"Found multiple validation-loss columns matching '{model_token}': {matches}"
        )
    return matches[0]


def main() -> None:
    args = get_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(f"Missing validation-loss CSV: {input_path}")

    df = pd.read_csv(input_path)
    if "epoch" not in df.columns:
        raise ValueError("Expected an 'epoch' column in the validation-loss CSV.")

    global2_col = find_loss_column(df.columns.tolist(), "GLOBAL-2-run4")
    global26_col = find_loss_column(df.columns.tolist(), "GLOBAL-26-run4")

    curves = [
        {
            "label": "GLOBAL-2",
            "column": global2_col,
            "color": cmodels["global"][2],
        },
        {
            "label": "GLOBAL-26",
            "column": global26_col,
            "color": cmodels["global"][4],
        },
    ]

    fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.8))

    for curve in curves:
        ax.plot(
            df["epoch"],
            1-df[curve["column"]],
            color=curve["color"],
            linewidth=2.6,
            marker="o",
            markersize=4.8,
            label=curve["label"],
        )
    #ax.set_ylim(0.97,1)
    #ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.set_xlabel("Epoch", fontsize=18)
    ax.set_ylabel("Validation NSE", fontsize=18)
    ax.legend(frameon=False, fontsize=14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
