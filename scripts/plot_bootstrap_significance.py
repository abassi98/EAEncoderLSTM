"""Map paired-bootstrap NSE significance for successive models and attributes models."""

import argparse
import json
import os
import shutil
import sqlite3
import tempfile
import warnings
from contextlib import closing
from pathlib import Path

# Keep font caches off the quota-limited cluster home directory.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / f"attention4hydro-matplotlib-{os.getuid()}"))
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.plot_utils import (
    load_state_polygons_local,
    BOOTSTRAP_COMPARISON_FEATURES as EXPERIMENTS,
    get_bootstrap_comparison_pairs as model_pairs,
)

try:
    from cmcrameri import cm
    MODEL_CMAP = cm.roma_r
except ImportError:
    MODEL_CMAP = plt.get_cmap("RdBu_r")

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "dejavuserif"

ROOT = Path(__file__).resolve().parents[1]  # repository root
P_THRESHOLDS = (0.05, 0.10, 0.15)
WIN_COLOR = "#d73027"
LOSS_COLOR = "#4575b4"
UNRESOLVED_COLOR = "#d0d0d0"


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_period", "--eval-period", choices=("test", "val"),
                        help="Defaults to the period saved with the comparisons, or test for older results.")
    parser.add_argument("--comparison-dir", type=Path, default=ROOT / "analysis/bootstrap/comparisons")
    parser.add_argument("--attributes-db", type=Path, default=ROOT / "data/attributes.db")
    parser.add_argument("--output-dir", type=Path,
                        help="Defaults to COMPARISON_DIR/maps, also used by compare_bootstrap.py --plot-maps.")
    parser.add_argument("--alphas", "--alpha", type=float, nargs="+",
                        help="Alphas for win/loss/unresolved maps; defaults to the saved comparison alpha.")
    parser.add_argument("--evidence-maps", action="store_true",
                        help="Also save the old fixed-p-value colour-band maps as bootstrap_evidence_*.png.")
    parser.add_argument("--strict", action="store_true", help="Stop if a requested comparison is missing instead of marking its panel unavailable.")
    return parser.parse_args()


def load_pair(comparisons, model_a, model_b, allow_missing=False):
    """Orient saved differences as A minus B; reversing a pair leaves its p-value unchanged."""
    forward = comparisons[(comparisons["model_a"] == model_a) & (comparisons["model_b"] == model_b)].copy()
    reverse = comparisons[(comparisons["model_a"] == model_b) & (comparisons["model_b"] == model_a)].copy()
    reverse["delta_nse_median"] *= -1
    pair = pd.concat([forward, reverse], ignore_index=True)
    if pair.empty:
        if allow_missing:
            warnings.warn(f"Missing comparison {model_a} / {model_b}; its panel will be marked unavailable.", stacklevel=2)
            return None
        raise ValueError(
            f"Missing comparison {model_a} / {model_b}. Rerun compare_bootstrap.py with --pairs paper "
            "and --extra-global-features 26; the required models need paired-bootstrap files."
        )
    pair["basin"] = pair["basin"].astype(str).str.zfill(8)
    if pair["basin"].duplicated().any():
        raise ValueError(f"Duplicate catchments or reversed duplicate comparisons for {model_a} / {model_b}.")
    return pair.set_index("basin")[["delta_nse_median", "p_adjusted", "status"]]


def significance_grades(pair):
    """Signed evidence at fixed p-value thresholds; zero is p > 0.15 or no direction."""
    delta = pair["delta_nse_median"].to_numpy(dtype=float)
    p = pair["p_adjusted"].to_numpy(dtype=float)
    if np.any(np.isfinite(p) & ((p < 0) | (p > 1))):
        raise ValueError("Adjusted p-values must lie in [0, 1].")
    valid = (pair["status"].to_numpy() == "tested") & np.isfinite(delta) & np.isfinite(p)
    grades = np.full(len(pair), np.nan)
    strength = sum((p <= cutoff).astype(int) for cutoff in P_THRESHOLDS)
    grades[valid] = np.sign(delta[valid]) * strength[valid]
    return grades


def decision_masks(pair, alpha):
    """First-model wins/losses at alpha, using the saved adjusted p-values."""
    if not 0 < alpha < 1:
        raise ValueError("Significance levels must be between 0 and 1.")
    valid = np.isfinite(significance_grades(pair))
    significant = valid & (pair["p_adjusted"].to_numpy(dtype=float) <= alpha)
    delta = pair["delta_nse_median"].to_numpy(dtype=float)
    return significant & (delta > 0), significant & (delta < 0)


def load_coordinates(path, basins):
    # Read the same CAMELS coordinates used by the existing NSE maps.
    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as connection:
        coordinates = pd.read_sql(
            "SELECT gauge_id, gauge_lon, gauge_lat FROM basin_attributes", connection, index_col="gauge_id"
        )
    coordinates.index = coordinates.index.astype(str).str.zfill(8)
    if coordinates.index.duplicated().any():
        raise ValueError("Duplicate catchment coordinates in the attributes database.")
    coordinates = coordinates.reindex(basins)
    missing = ~np.isfinite(coordinates.to_numpy(dtype=float)).all(axis=1)
    if missing.any():
        raise ValueError(f"Missing coordinates for catchments: {coordinates.index[missing].tolist()[:10]}")
    return coordinates


def plot_maps(pairs, coordinates, state_polygons, experiment, kind, config, output_path,
              decisions_only=True, alias_path=None):
    alpha = config["alpha"]
    # Fixed evidence shades, independent of the alpha used for the table and counts.
    colors = [MODEL_CMAP(value) for value in (0.03, 0.17, 0.31, 0.5, 0.69, 0.83, 0.97)]
    colors[3] = "#d0d0d0"
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-3.5, 4), cmap.N)
    nrows = (len(pairs) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(24, 6.5 * nrows + 3), squeeze=False)

    for ax, ((model_a, model_b), pair) in zip(axes.flat, pairs.items()):
        ax.set_title(f"NSE {model_a} minus {model_b}", fontsize=35, pad=20)
        if pair is None:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"Comparison unavailable\n{model_a} / {model_b}\n\n"
                    "Include both models in compare_bootstrap.py.",
                    transform=ax.transAxes, ha="center", va="center", fontsize=17, color="#666666")
            continue
        ax.add_collection(LineCollection(state_polygons, colors="black", linewidths=0.4, zorder=0))
        ax.set_xlim(-128, -65)
        ax.set_ylim(24, 50)
        ax.set_aspect(1 / np.cos(np.deg2rad(37)))
        ax.set_axis_off()
        locations = coordinates.loc[pair.index]
        grades = significance_grades(pair)
        better, worse = decision_masks(pair, alpha)
        if decisions_only:
            unresolved = ~(better | worse)
            for mask, color in ((unresolved, UNRESOLVED_COLOR), (better, WIN_COLOR), (worse, LOSS_COLOR)):
                ax.scatter(locations.loc[mask, "gauge_lon"], locations.loc[mask, "gauge_lat"],
                           c=color, s=100, edgecolors="#555555", linewidths=0.3)
            ax.text(0.5, -0.035,
                    f"At α = {alpha:g}: {better.sum()} first wins | {worse.sum()} second wins | "
                    f"{unresolved.sum()} unresolved",
                    transform=ax.transAxes, ha="center", va="top", fontsize=14)
            continue
        for mask in (grades == 0, np.isfinite(grades) & (grades != 0)):
            ax.scatter(locations.loc[mask, "gauge_lon"], locations.loc[mask, "gauge_lat"],
                       c=grades[mask], cmap=cmap, norm=norm, s=100, edgecolors="#555555", linewidths=0.3)
        missing = ~np.isfinite(grades)
        ax.scatter(locations.loc[missing, "gauge_lon"], locations.loc[missing, "gauge_lat"],
                   marker="x", c="black", s=100, linewidths=1.2)
        no_difference = ~missing & ~better & ~worse
        ax.text(0.5, -0.035,
                f"At α = {alpha:g}, first model: {better.sum()} better | {worse.sum()} worse | "
                f"{no_difference.sum()} no difference detected | {missing.sum()} untested",
                transform=ax.transAxes, ha="center", va="top", fontsize=14)
    for ax in axes.flat[len(pairs):]:
        ax.set_axis_off()

    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.32 if nrows == 1 else 0.22,
                        top=0.93 if nrows == 1 else 0.94, hspace=0.35, wspace=0.05)
    if decisions_only:
        handles = [Line2D([], [], marker="o", linestyle="none", color=color, markersize=12,
                          label=label) for color, label in (
                              (WIN_COLOR, "First model wins"),
                              (LOSS_COLOR, "Second model wins (first model loses)"),
                              (UNRESOLVED_COLOR, "Unresolved"),
                          )]
        level_label = "FDR target" if config["correction"] in ("fdr_by", "fdr_bh") else "Significance level"
        fig.legend(handles=handles, loc="lower center", ncol=3,
                   bbox_to_anchor=(0.5, 0.10 if nrows == 1 else 0.05), fontsize=20,
                   title=(f"{level_label} α = {alpha:g}\n"
                          f"Wins require adjusted p ≤ α ({config['correction']})"), title_fontsize=22)
    else:
        cax = fig.add_axes([0.10, 0.18 if nrows == 1 else 0.10, 0.80, 0.03 if nrows == 1 else 0.02])
        colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                               orientation="horizontal", ticks=np.arange(-3, 4))
        low, middle, high = P_THRESHOLDS
        intervals = [f"p ≤ {low:.2f}", f"{low:.2f} < p ≤ {middle:.2f}", f"{middle:.2f} < p ≤ {high:.2f}"]
        colorbar.set_ticklabels([f"Worse\n{interval}" for interval in intervals]
                               + [f"No detected\ndifference\np > {high:.2f}"]
                               + [f"Better\n{interval}" for interval in reversed(intervals)])
        colorbar.ax.tick_params(labelsize=20)
        colorbar.set_label(r"Significance of $\Delta$ NSE (adjusted $p$)", fontsize=25, labelpad=12)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if alias_path is not None:
        shutil.copyfile(output_path, alias_path)


def generate_maps(comparison_dir, output_dir, eval_period=None, attributes_db=ROOT / "data/attributes.db",
                  allow_missing=True, alphas=None, evidence_maps=False):
    comparison_dir, output_dir = Path(comparison_dir), Path(output_dir)
    for filename in ("comparison_config.json", "catchment_comparisons.csv"):
        if not (comparison_dir / filename).is_file():
            raise FileNotFoundError(
                f"Missing {comparison_dir.resolve() / filename}. Set --comparison-dir to the --output-dir "
                "used for compare_bootstrap.py (for example, analysis/bootstrap/comparisons_alpha010), "
                "or run compare_bootstrap.py --alpha 0.10 --plot-maps first."
            )
    config = json.loads((comparison_dir / "comparison_config.json").read_text())
    print(f"Reading comparison CSV: {(comparison_dir / 'catchment_comparisons.csv').resolve()}", flush=True)
    if config.get("models"):
        print("Models in this saved comparison: " + ", ".join(config["models"]), flush=True)
    saved_period = config.get("eval_period")
    if saved_period and eval_period and saved_period != eval_period:
        raise ValueError(f"These comparisons are for {saved_period}, but --eval_period is {eval_period}.")
    eval_period = eval_period or saved_period or "test"
    if not 0 < config["alpha"] < 1:
        raise ValueError("The saved significance level must be between 0 and 1.")
    decision_alphas = list(dict.fromkeys([config["alpha"]] if alphas is None else alphas))
    if not decision_alphas or any(not 0 < alpha < 1 for alpha in decision_alphas):
        raise ValueError("Significance levels must be between 0 and 1.")
    comparisons = pd.read_csv(comparison_dir / "catchment_comparisons.csv", dtype={"basin": str})
    required = {"model_a", "model_b", "basin", "delta_nse_median", "p_adjusted", "status"}
    if not required.issubset(comparisons.columns):
        raise ValueError(f"Comparison CSV is missing columns: {sorted(required - set(comparisons.columns))}")
    # A missing model should not prevent the available comparisons from being plotted.
    figures = {
        (experiment, kind): {pair: load_pair(comparisons, *pair, allow_missing=allow_missing)
                             for pair in model_pairs(experiment, kind)}
        for experiment in EXPERIMENTS for kind in ("previous", "attr")
    }
    basins = sorted({basin for pairs in figures.values() for pair in pairs.values()
                     if pair is not None for basin in pair.index})
    if not basins:
        raise ValueError("None of the requested model pairs are present. Use the comparison output with PUB/GLOBAL model labels.")
    coordinates = load_coordinates(Path(attributes_db), basins)
    state_path = ROOT / "data/usa-states-census-2014.shp"
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing state outlines: {state_path}")
    state_polygons = load_state_polygons_local(state_path)
    if not state_polygons:
        raise ValueError(f"No polygons read from {state_path}. Update src/plot_utils.py to include PolygonZ support.")
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Generating win/loss/unresolved maps at alpha=" +
          ", ".join(f"{alpha:g}" for alpha in decision_alphas) +
          f"; saved alpha={config['alpha']:g}, correction={config['correction']}.", flush=True)
    print(f"The standard bootstrap_significance filenames use alpha={decision_alphas[0]:g}.", flush=True)
    print(f"Map output directory: {output_dir.resolve()}", flush=True)
    decision_counts = []
    for (experiment, kind), pairs in figures.items():
        if evidence_maps:
            evidence_path = output_dir / f"bootstrap_evidence_{kind}_{experiment}_{eval_period}.png"
            plot_maps(pairs, coordinates, state_polygons, experiment, kind, config, evidence_path,
                      decisions_only=False)
            print(f"Saved {evidence_path.resolve()}", flush=True)
        previous_decisions = {}
        for alpha_index, alpha in enumerate(decision_alphas):
            for (a, b), pair in pairs.items():
                if pair is None:
                    continue
                wins, losses = decision_masks(pair, alpha)
                unresolved = len(pair) - int(wins.sum()) - int(losses.sum())
                decision_counts.append(dict(experiment=experiment, kind=kind, model_a=a, model_b=b,
                                            alpha=alpha, first_wins=int(wins.sum()),
                                            second_wins=int(losses.sum()), unresolved=unresolved))
                message = (f"{a} vs {b}, alpha={alpha:g}: {wins.sum()} first wins, "
                           f"{losses.sum()} second wins, {unresolved} unresolved.")
                if (a, b) in previous_decisions:
                    previous_alpha, previous_wins, previous_losses = previous_decisions[a, b]
                    if np.array_equal(wins, previous_wins) and np.array_equal(losses, previous_losses):
                        message += (f" Same catchments as alpha={previous_alpha:g}; "
                                    "no decision threshold was crossed.")
                print(message, flush=True)
                previous_decisions[a, b] = alpha, wins, losses
            tag = str(float(alpha)).replace(".", "p")
            decision_path = output_dir / f"bootstrap_decisions_{kind}_{experiment}_{eval_period}_alpha{tag}.png"
            alias_path = (output_dir / f"bootstrap_significance_{kind}_{experiment}_{eval_period}.png"
                          if alpha_index == 0 else None)
            plot_maps(pairs, coordinates, state_polygons, experiment, kind, {**config, "alpha": alpha},
                      decision_path, decisions_only=True, alias_path=alias_path)
            print(f"Saved {decision_path.resolve()}", flush=True)
            if alias_path is not None:
                print(f"Updated {alias_path.resolve()}", flush=True)
    counts_path = output_dir / "map_decision_counts.csv"
    pd.DataFrame(decision_counts).to_csv(counts_path, index=False)
    print(f"Saved {counts_path.resolve()}", flush=True)
    missing = [f"{a} / {b}" for pairs in figures.values() for (a, b), pair in pairs.items() if pair is None]
    if missing:
        print("Unavailable panels: " + ", ".join(missing), flush=True)
        missing_labels = {label for pairs in figures.values() for pair, values in pairs.items()
                          if values is None for label in pair if label not in config.get("models", {})}
        skipped = {model["label"]: model for model in config.get("skipped_models", [])}
        for label in sorted(missing_labels):
            if label in skipped:
                print(f"  {label} was skipped by the saved comparison: {skipped[label]['reason']}", flush=True)
                for path in skipped[label].get("expected_files", []):
                    print(f"    Expected: {path}", flush=True)
                for path in skipped[label].get("related_files", []):
                    print(f"    Other filename found: {path}", flush=True)
            elif config.get("models"):
                print(f"  {label} was not selected in this saved comparison run.", flush=True)
        if not missing_labels:
            print(f"  The models are selected, but their pair is absent. Saved pair selection: "
                  f"pairs={config.get('pairs', 'unknown')}, reference={config.get('reference')}.", flush=True)
        print("To fill them, include the missing models when rerunning compare_bootstrap.py; "
              "GLOBAL-26 needs --extra-global-features 26 (or --plot-maps).", flush=True)


def main():
    args = get_args()
    try:
        output_dir = args.output_dir if args.output_dir is not None else args.comparison_dir / "maps"
        generate_maps(args.comparison_dir, output_dir, args.eval_period, args.attributes_db,
                      allow_missing=not args.strict, alphas=args.alphas, evidence_maps=args.evidence_maps)
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
