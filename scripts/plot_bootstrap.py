
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.plot_utils import (
    get_bootstrap_plot_models,
    get_bootstrap_random_feature_model_ids,
)
from pathlib import PosixPath
import pickle
import matplotlib as mpl
from scripts.plot_ensemble import plot_sensitivity
from scripts.compute_leave_one_out import load_leave_one_out

mpl.rcParams['xtick.labelsize'] = 20 
mpl.rcParams['ytick.labelsize'] = 20 
#matplotlib.font_manager.findfont("Symbol")
mpl.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')[:10]
mpl.rc('text', usetex=True)
mpl.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsfonts}')

# Say, "the default sans-serif font is COMIC SANS"
plt.rcParams['font.serif'] = "Times New Roman"
# Then, "ALWAYS use sans-serif fonts"
plt.rcParams['font.family'] = "serif"
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
#plt.rcParams['font.weight'] = 'bold'


def get_args():
    """Parse input argumentsexit#

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()

   
    parser.add_argument(
        "--eval_period",
        type=str,
        default="test",
    )
    
    parser.add_argument(
        "--metric",
        type=str,
        default="nse",
        choices=["nse","bias","stdev","obs5","sim5","obs95","sim95","obs0","sim0","obsL","simL","obsH","simH","obsFDC","simFDC","obsBF","simBF","abs_nse","sqrt_nse","FHV","FLV","kge","FMM","bias_RR","log_stdev","r_coeff","bias_std","skew_rat","kurt_rat"],
    )

    parser.add_argument(
        "--xmin",
        type=float,
        default=0.4,
    )   
    parser.add_argument(
        "--xmax",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--random_features_models",
        nargs="*",
        default=(),
        choices=get_bootstrap_random_feature_model_ids(),
        help="Random-feature bootstrap models to include (default: none).",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--restart_envelope",
        action="store_true",
        help=(
            "Plot the cumulative-distribution envelope across the four restart "
            "seeds stored in analysis/stats, instead of bootstrap curves."
        ),
    )
    modes.add_argument(
        "--leave_one_out", "--leave-one-out",
        action="store_true",
        help=("Read leave_one_out CSVs from compute_leave_one_out.py. Plot the "
              "full-ensemble curve and shade the range of the full and omission "
              "curves (sensitivity envelope, not a confidence interval)."),
    )
    cfg = vars(parser.parse_args())
    if not np.isfinite([cfg["xmin"], cfg["xmax"]]).all() or cfg["xmin"] >= cfg["xmax"]:
        parser.error("--xmin and --xmax must be finite, with xmin < xmax")
    return cfg


if __name__ == '__main__':
    ##########################################################
    # Load encoded features of chosen LSTM-AE model
    ##########################################################
    # Load encoded features
    cfg = get_args()
    eval_period = cfg["eval_period"]
    metric = cfg["metric"]
    xmax= cfg["xmax"]
    xmin = cfg["xmin"]
    
    
    # Statistics distributions
    palette = {}
    model_specs = get_bootstrap_plot_models(
        random_feature_models=tuple(cfg["random_features_models"])
    )

    restart_inputs = {}
    if cfg["leave_one_out"]:
        sensitivity = load_leave_one_out(model_specs, eval_period, metric)
        palette = {spec["label"]: spec["color"] for spec in model_specs}
    elif cfg["restart_envelope"]:
        seed_prefix = f"{metric}_seed"
        for spec in model_specs:
            stats_path = PosixPath(
                f"analysis/stats/{eval_period}/{spec['file_experiment']}_es"
                f"{spec['encoded_features']}.csv"
            )
            if not stats_path.is_file():
                warnings.warn(f"Skipping model without restart statistics: {stats_path}")
                continue

            stats = pd.read_csv(stats_path, index_col=0)
            seed_columns = [col for col in stats.columns if col.startswith(seed_prefix)]
            if len(seed_columns) != 4:
                warnings.warn(
                    f"Skipping {spec['label']}: expected four {seed_prefix} columns in "
                    f"{stats_path}, found {len(seed_columns)}."
                )
                continue

            restart_inputs[spec["label"]] = {
                "spec": spec,
                "data": stats[seed_columns],
            }
            palette[spec["label"]] = spec["color"]

        if not restart_inputs:
            raise ValueError(
                f"No four-restart {metric.upper()} statistics found for {eval_period}."
            )
    else:
        basin_file = PosixPath(__file__).resolve().parents[1] / "data/basin_list.txt"
        basins = basin_file.read_text().splitlines()
        bootstrap_inputs = {}
        for spec in model_specs:
            experiment = spec["file_experiment"]
            ef = spec["encoded_features"]
            output_suffix = "_random_features" if spec["random_features"] else ""
            fname = (
                f"analysis/bootstrap/restarts_{metric}_{experiment}_{ef}_{eval_period}"
                f"{output_suffix}.pkl"
            )

            try:
                with open(fname, "rb") as f:
                    bootstrap_inputs[spec["label"]] = {
                        "spec": spec,
                        "data": pickle.load(f),
                    }
            except FileNotFoundError:
                if spec["random_features"]:
                    warnings.warn(f"Skipping missing random-features bootstrap file: {fname}")
                    continue
                raise

        common_basins = set(basins)
        for bootstrap_entry in bootstrap_inputs.values():
            common_basins &= set(bootstrap_entry["data"].keys())
        common_basins = [basin for basin in basins if basin in common_basins]

        if len(common_basins) == 0:
            raise ValueError(
                f"No shared basins found across bootstrap files for metric={metric} and eval_period={eval_period}."
            )

        print(
            f"Using {len(common_basins)} common basins for bootstrap plotting in the "
            f"{eval_period} period."
        )

        df_q05 = pd.DataFrame(index=common_basins)
        df_q5 = pd.DataFrame(index=common_basins)
        df_q95 = pd.DataFrame(index=common_basins)

        for model_name, bootstrap_entry in bootstrap_inputs.items():
            c = bootstrap_entry["spec"]["color"]
            dict_metric = bootstrap_entry["data"]

            q05 = []
            q5 = []
            q95 = []
            for basin in common_basins:
                bootstrapped_nse = dict_metric[basin]
                q05.append(np.quantile(bootstrapped_nse, 0.25))
                q5.append(np.quantile(bootstrapped_nse, 0.5))
                q95.append(np.quantile(bootstrapped_nse, 0.75))

            df_q05[model_name] = q05
            df_q5[model_name] = q5
            df_q95[model_name] = q95
            palette[model_name] = c
             

    
    # ======================================================
    # Plot: pUB models left, global models right
    # cumulative distributions only, preserving colors
    # ======================================================

    bootstrap_mode = not (cfg["restart_envelope"] or cfg["leave_one_out"])
    available_labels = sensitivity if cfg["leave_one_out"] else (
        restart_inputs if cfg["restart_envelope"] else df_q5.columns
    )
    pub_cols = [
        spec["label"]
        for spec in model_specs
        if spec["label"] in available_labels and spec["family"] == "pub"
    ]
    
    global_cols = [
        spec["label"]
        for spec in model_specs
        if spec["label"] in available_labels and spec["family"] == "global"
    ]

    if bootstrap_mode:
        pub_reference = df_q5["PUB-A"]
        global_reference = df_q95["GLOBAL-A"]

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True, sharex=True, sharey=True)

    plot_groups = [
        (ax[0], pub_cols, r"PUB models"),
        (ax[1], global_cols, r"GLOBAL models"),
    ]

    x_grid = np.linspace(xmin, xmax, 500)

    def plot_restart_envelope(axi, model_name, color, alpha, label, zorder):
        """Plot the pointwise cumulative-KDE range across the four restarts."""
        seed_cdfs = []
        for _, seed_values in restart_inputs[model_name]["data"].items():
            sns.kdeplot(
                seed_values.dropna(),
                bw_adjust=0.4,
                cumulative=True,
                common_norm=False,
                ax=axi,
                alpha=0,
                legend=False,
            )
            seed_curve = axi.lines[-1]
            seed_cdfs.append(
                np.interp(x_grid, seed_curve.get_xdata(), seed_curve.get_ydata())
            )
            seed_curve.remove()

        seed_cdfs = np.vstack(seed_cdfs)
        axi.fill_between(
            x_grid,
            seed_cdfs.min(axis=0),
            seed_cdfs.max(axis=0),
            color=color,
            alpha=alpha,
            linewidth=0,
            label=label,
            zorder=zorder,
        )

    for axi, cols, title in plot_groups:

        axi.grid()

        if cfg["leave_one_out"]:
            reference_model = "GLOBAL-A" if "PUB" in title else "PUB-A"
            if reference_model in sensitivity:
                plot_sensitivity(axi, sensitivity[reference_model], x_grid, "0.2", reference_model,
                                 linestyle="--", linewidth=1.5, alpha=0.7, zorder=1)
            for col in cols:
                plot_sensitivity(axi, sensitivity[col], x_grid, palette[col], col,
                                 linewidth=2.0, zorder=2)
            axi.legend()
        elif cfg["restart_envelope"]:
            reference_model = "GLOBAL-A" if "PUB" in title else "PUB-A"
            if reference_model in restart_inputs:
                plot_restart_envelope(
                    axi,
                    reference_model,
                    color="0.2",
                    alpha=0.30,
                    label=reference_model,
                    zorder=1,
                )

            for col in cols:
                plot_restart_envelope(
                    axi,
                    col,
                    color=palette[col],
                    alpha=0.55,
                    label=col,
                    zorder=2,
                )
            axi.legend()
        else:
            sns.kdeplot(
                df_q5[cols],
                ax=axi,
                bw_adjust=0.4,
                cumulative=True,
                common_norm=False,
                legend=True,
                palette={c: palette[c] for c in cols},
            )

        for col in cols if bootstrap_mode else []:

            

            color = palette[col]

            sns.kdeplot(
                df_q05[col].dropna(),
                bw_adjust=0.4,
                cumulative=True,
                common_norm=False,
                ax=axi,
                color=color,
                alpha=0,
                legend=False,
            )

            sns.kdeplot(
                df_q95[col].dropna(),
                bw_adjust=0.4,
                cumulative=True,
                common_norm=False,
                ax=axi,
                color=color,
                alpha=0,
                legend=False,
            )

            x05 = axi.lines[-2].get_xdata()
            y05 = axi.lines[-2].get_ydata()

            x95 = axi.lines[-1].get_xdata()
            y95 = axi.lines[-1].get_ydata()

            y05_interp = np.interp(x_grid, x05, y05)
            y95_interp = np.interp(x_grid, x95, y95)

            axi.fill_between(
                x_grid,
                y05_interp,
                y95_interp,
                color=color,
                alpha=0.2,
                linewidth=0,
            )

            # remove invisible helper lines
            axi.lines[-1].remove()
            axi.lines[-1].remove()

        axi.set_title(title, fontsize=22, fontweight="bold")
        axi.set_ylabel("")
        axi.set_xlabel("")
        axi.set_xlim([xmin, xmax])

    
        if bootstrap_mode:
            # Reference distribution from opposite family.
            if "PUB" in title:
                sns.kdeplot(
                    global_reference,
                    ax=axi,
                    cumulative=True,
                    bw_adjust=0.4,
                    color="0.2",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    label="Global ensemble",
                    zorder=1,
                )
            else:
                sns.kdeplot(
                    pub_reference,
                    ax=axi,
                    cumulative=True,
                    bw_adjust=0.4,
                    color="0.2",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.7,
                    label="PUB ensemble",
                    zorder=1,
                )

            sns.kdeplot(
                df_q5[cols],
                ax=axi,
                bw_adjust=0.4,
                cumulative=True,
                common_norm=False,
                palette={c: palette[c] for c in cols},
                linewidth=2.0,
                zorder=2,
            )

    if cfg["eval_period"] == "test":
        title = f"Test {cfg['metric'].upper()}"
    elif cfg["eval_period"] == "train":
        title = f"Train {cfg['metric'].upper()}"
    else:
        title = f"Validation {cfg['metric'].upper()}"

    fig.supxlabel(
        title,
        rotation=0,
        fontweight="bold",
        fontsize=25,
    )

    output_suffix = "_restart_envelope" if cfg["restart_envelope"] else ""
    if cfg["leave_one_out"]:
        output_suffix = "_leave_one_out"
    if cfg["random_features_models"]:
        output_suffix += "_with_random_features"
    output_path = PosixPath(
        f"analysis/figures/bootstrap_{metric}_cumulative_{cfg['eval_period']}{output_suffix}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved {output_path}")
    # # Create formatted dataframe
    # table_df = pd.DataFrame(index=mean_vals.index)

    # table_df["Metric"] = [
    #     f"${m:.3f}\\,(\\pm\\,{s:.3f})$"
    #     for m, s in zip(mean_vals, std_vals)
    # ]

    # # transpose so models are rows
    # latex_df = table_df.reset_index()
    # latex_df.columns = ["Model", metric.upper()]

    # latex_table = latex_df.to_latex(
    #     index=False,
    #     escape=False,
    #     column_format="lc",
    #     caption=f"{metric.upper()} performance summary.",
    #     label=f"tab:{metric}_summary"
    # )

    # print(latex_table)
    
  
