import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.datautils import load_attributes, load_signatures, LANDSCAPE_ATTRS, CLIM_ATTRS, HYDRO_ATTRS, SOIL_ATTRS, VEGE_ATTRS, TOPO_ATTRS, GEOL_ATTRS
from src.utils import get_basin_list, clean_and_capitalize
from pathlib import PosixPath
from scipy.stats import pearsonr
from cmcrameri import cm
from src.plot_utils import cmodels, load_us_states

import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 20 
mpl.rcParams['ytick.labelsize'] = 20 
#matplotlib.font_manager.findfont("Symbol")
mpl.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')[:10]
mpl.rc('text', usetex=True)
mpl.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsfonts} \usepackage{xcolor}')

# Say, "the default sans-serif font is COMIC SANS"
plt.rcParams['font.serif'] = "Times New Roman"
# Then, "ALWAYS use sans-serif fonts"
plt.rcParams['font.family'] = "serif"
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
#plt.rcParams['font.weight'] = 'bold'

LABEL_TO_ATTR = {
    clean_and_capitalize(attr): attr
    for attr in CLIM_ATTRS + SOIL_ATTRS + VEGE_ATTRS + TOPO_ATTRS + GEOL_ATTRS + HYDRO_ATTRS
}

ATTR_TEXT_COLORS = {
    **{attr: "2166AC" for attr in CLIM_ATTRS},
    **{attr: "1B7837" for attr in VEGE_ATTRS},
    **{attr: "8C510A" for attr in SOIL_ATTRS},
    **{attr: "762A83" for attr in TOPO_ATTRS},
}


def get_attr_text_color(label):
    attr = LABEL_TO_ATTR.get(label)
    return f"#{ATTR_TEXT_COLORS[attr]}" if attr in ATTR_TEXT_COLORS else None


def save_and_plot_delta_nse_distributions(delta_nse_by_pair, experiment, eval_period):
    """Persist successive delta-NSE values and plot their distributions."""
    delta_nse_df = pd.DataFrame(delta_nse_by_pair)
    os.makedirs("analysis/delta_nse", exist_ok=True)
    delta_nse_df.to_csv(f"analysis/delta_nse/delta_nse_{experiment}.csv")

    delta_nse_long = (
        delta_nse_df.rename_axis("basin")
        .reset_index()
        .melt(id_vars="basin", var_name="transition", value_name="delta_nse")
        .dropna()
    )

    if delta_nse_long.empty:
        raise ValueError(
            f"No successive delta-NSE values available for {experiment} during {eval_period}."
        )

    fig_width = max(8, 3.5 * len(delta_nse_by_pair))
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 7))
    palette = [cm.roma_r(value) for value in np.linspace(0.15, 0.85, len(delta_nse_by_pair))]

    sns.violinplot(
        data=delta_nse_long,
        x="transition",
        y="delta_nse",
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
        width=0.22,
        whis=(5, 95),
        showfliers=False,
        boxprops={"facecolor": "white", "zorder": 3},
        medianprops={"color": "black", "linewidth": 2},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
        ax=ax,
    )

    positive_fraction = (
        delta_nse_long.groupby("transition")["delta_nse"]
        .apply(lambda values: (values > 0).mean())
        .reindex(delta_nse_df.columns)
    )

    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlabel("Successive model transition", fontsize=25)
    ax.set_ylabel(r"$\Delta$ NSE", fontsize=25)
    ax.tick_params(axis="x", rotation=15, labelsize=18)

    # Focus the figure on the central distribution; the CSV retains all outliers.
    y_limit = max(abs(delta_nse_long["delta_nse"].quantile(0.01)),
                  abs(delta_nse_long["delta_nse"].quantile(0.99)))
    ax.set_ylim(-y_limit, y_limit)

    ymin, ymax = ax.get_ylim()
    y_text = ymax - 0.04 * (ymax - ymin)
    for idx, transition in enumerate(delta_nse_df.columns):
        ax.text(
            idx,
            y_text,
            f"{positive_fraction[transition]:.0%} > 0",
            ha="center",
            va="top",
            fontsize=14,
        )

    fig.tight_layout()
    fig.savefig(
        f"analysis/figures/delta_nse_distribution_{experiment}_{eval_period}.png",
        dpi=300,
    )


def get_args():
    """Parse input arguments

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
        "--top_improvement_fraction",
        type=float,
        default=0.30,
        help="Fraction of the largest positive delta NSE values retained for each feature transition.",
    )
    cfg = vars(parser.parse_args())
    return cfg


if __name__ == '__main__':
    ##########################################################
    # Load encoded features of chosen LSTM-AE model
    ##########################################################
    # Load encoded features
    cfg = get_args()
    if not 0 < cfg["top_improvement_fraction"] <= 1:
        raise ValueError("--top_improvement_fraction must be in the interval (0, 1].")
    
    us_states = load_us_states()
    experiment = "pub"
    eval_period = cfg["eval_period"]

    # load caam and basins
    stat_global_a = pd.read_csv(f"analysis/stats/{eval_period}/{experiment}_esNone.csv", sep=",", index_col=0)
    stat_global_a.index = [str(s).rjust(8,"0") for s in stat_global_a.index]
    basins = get_basin_list()
    num_basins = len(basins)

    vmin = -0.2
    vmax = 0.2
    # load lat and lon
    camels_root = PosixPath("data/basin_dataset_public_v1p2/")
    KEEP_ATTRS = SOIL_ATTRS + CLIM_ATTRS + VEGE_ATTRS + TOPO_ATTRS + GEOL_ATTRS + ["gauge_lat", "gauge_lon"]
    df_S = load_attributes("data/attributes.db", basins, keep_attributes=KEEP_ATTRS)

    # insert lat and lon in encoded attributes
    lon = df_S["gauge_lon"]
    lat = df_S["gauge_lat"]
    df_S = df_S.drop(['gauge_lat', 'gauge_lon'], axis=1)
    df_S = (df_S - df_S.mean())/df_S.std()
    attrs_names = df_S.columns
    signatures = load_signatures(f"analysis/signatures/camels_us_{eval_period}.csv")
    signatures = signatures.reindex(df_S.index)
    signatures = (signatures - signatures.mean()) / signatures.std()
    explanatory_variables = pd.concat([df_S, signatures], axis=1)
    explanatory_names = explanatory_variables.columns
    delta_nse_by_pair = {}
    selection_by_pair = {}

    markers = ["^", "o", "v", "<", ">", "s"]
    i=0
    line = np.linspace(0,1,1000)
    

    ### plot nse current N versus previous in encoded_features_vec
    encoded_features_vec = [0,1,2]
    # plot scatter delta nse vs attributes
    names_to_keep = ["baseflow_index", "aridity", "high_q_freq", "q95", "carbonate_rocks_frac", "elev_mean",
                         "frac_snow", "frac_forest", "soil_conductivity" ]
    
    fig,ax = plt.subplots(1,4, figsize=(30,5), squeeze=False)

    for ii in range(1,len(encoded_features_vec)): # i = 1,2,3,4 column = 0, 2, 0, 2
        row, column = (ii-1) // 2, 2 * ((ii-1 ) % 2 )
        ax[row, column].set_xlim(-128, -65)
        ax[row, column].set_ylim(24, 50)
        ax[row, column].set_axis_off()
        us_states.boundary.plot(color="black", ax=ax[row, column], linewidth=0.5)

        # retrieve stats
        stats = pd.read_csv(f"analysis/stats/{eval_period}/{experiment}_es{encoded_features_vec[ii]}.csv", sep=",", index_col=0)
        stats.index = [str(s).rjust(8,"0") for s in stats.index]
        nse = stats["nse"]

        # retrieve previous stats
        stats_prev = pd.read_csv(f"analysis/stats/{eval_period}/{experiment}_es{encoded_features_vec[ii-1]}.csv", sep=",", index_col=0)
        stats_prev.index = [str(s).rjust(8,"0") for s in stats_prev.index]
        nse_prev = stats_prev["nse"]

        delta_nse = nse - nse_prev
        color = np.clip(delta_nse, -1,1)
        pair_name = rf"$\Delta$NSE {encoded_features_vec[ii]}-{encoded_features_vec[ii-1]}"
        delta_nse_by_pair[pair_name] = delta_nse

        # Retain the largest positive performance improvements after adding latent features.
        candidates = pd.concat(
            [delta_nse.rename("delta_nse"), nse_prev.rename("previous_nse")], axis=1, join="inner"
        ).dropna()
        candidates = candidates[candidates["delta_nse"] > 0]
        if candidates.empty:
            raise ValueError(f"No improved basins available for {pair_name}.")
        cutoff = candidates["delta_nse"].quantile(1 - cfg["top_improvement_fraction"])
        selection_by_pair[pair_name] = candidates.loc[
            candidates["delta_nse"] >= cutoff, ["delta_nse", "previous_nse"]
        ]
        
        im = ax[row, column].scatter(x=lon, y=lat,c=color, s=100, cmap=cm.roma_r, vmin=vmin,vmax=vmax )
        ax[row, column].set_title(f"NSE {experiment.upper()}-{encoded_features_vec[ii]} minus {experiment.upper()}-{encoded_features_vec[ii-1]}", fontsize=35, y=-0.2)
        ax[row, column+1].set_ylabel(f"NSE {experiment.upper()}-{encoded_features_vec[ii-1]}", fontsize=35)
        ax[row, column+1].set_xlabel(f"NSE {experiment.upper()}-{encoded_features_vec[ii]}", fontsize=35)
        ax[row, column+1].plot(line, line, c="black",ls="--")
        sc = ax[row, column+1].scatter(np.clip(nse, 0,1), np.clip(nse_prev,0,1), c=color, s=100, cmap=cm.roma_r, vmin=vmin,vmax=vmax )
        cb = fig.colorbar(sc)
        cb.set_label(f"Delta NSE", fontsize=25) 
        #ax[row, column+1].text(-0.8,0.8, f"r = {np.round(pearsonr(nse, nse_prev)[0], 2)}", fontsize=30)
        ax[row, column+1].grid()
        
   
    fig.tight_layout()
    fig.savefig(f"analysis/figures/delta_nse_previous_{experiment}_{eval_period}.png", dpi = 300)
    save_and_plot_delta_nse_distributions(delta_nse_by_pair, experiment, eval_period)


    fig, axs = plt.subplots(1,1, figsize=(26,6))
    # Correlate each transition only on basins with the largest positive improvements.
    corr = pd.DataFrame(index=delta_nse_by_pair, columns=explanatory_names, dtype=float)
    selected_delta = pd.DataFrame(index=explanatory_variables.index)
    selected_basins = []
    for pair_name, delta_nse in delta_nse_by_pair.items():
        selection = selection_by_pair[pair_name]
        selected_delta[pair_name] = delta_nse.reindex(explanatory_variables.index)
        selected_delta.loc[~selected_delta.index.isin(selection.index), pair_name] = np.nan
        selected_basins.append(selection.assign(transition=pair_name))
        frame = pd.concat(
            [explanatory_variables.loc[selection.index], delta_nse.loc[selection.index].rename(pair_name)], axis=1
        )
        corr.loc[pair_name] = frame.corr("spearman").loc[pair_name, explanatory_names]
    g = sns.heatmap(np.abs(corr), fmt='', cmap="viridis", vmin=0, vmax=1, ax=axs)
    selected_delta.to_csv(f"analysis/delta_nse/delta_nse_{experiment}_top_improvement.csv")
    pd.concat(selected_basins).to_csv(f"analysis/delta_nse/delta_nse_{experiment}_selection.csv")
    xlabels = [clean_and_capitalize(attr) for attr in explanatory_names]
    g.set_xlabel("Catchment attributes and hydrologic signatures", fontsize=25)
    g.set_xticks(np.arange(len(xlabels)) + 0.5)
    g.set_xticklabels(xlabels, rotation = 80, fontsize=15)
    for tick_label, label in zip(g.get_xticklabels(), xlabels):
        color = get_attr_text_color(label)
        if color is not None:
            tick_label.set_color(color)
    g.set_yticks(np.arange(len(corr.index)) + 0.5)
    g.set_yticklabels(corr.index, rotation = 00, fontsize=15)
    fig.tight_layout()
    fig.savefig(f"analysis/figures/Spearman_delta_nse_{experiment}_top_improvement.png", dpi=300)
    
    
    ### plot nse -N vs -A
    encoded_features_vec = [0, 1, 2]

    fig = plt.figure(figsize=(30, 10))

    outer = fig.add_gridspec(
        2, 3,
        width_ratios=[1, 3, 1],  # center column contains the plots
        height_ratios=[1, 1]
    )

    # First row
    top = outer[0, :].subgridspec(
        1, 4,
        width_ratios=[2, 1, 2, 1]
    )

    # Second row (centered pair)
    bottom = outer[1, 1].subgridspec(
        1, 2,
        width_ratios=[2, 1]
    )

    axes_pairs = [
        (fig.add_subplot(top[0, 0]), fig.add_subplot(top[0, 1])),
        (fig.add_subplot(top[0, 2]), fig.add_subplot(top[0, 3])),
        (fig.add_subplot(bottom[0, 0]), fig.add_subplot(bottom[0, 1]))
    ]

    for ii, (ax_map, ax_scatter) in enumerate(axes_pairs):
        stats = pd.read_csv(
            f"analysis/stats/{eval_period}/{experiment}_es{encoded_features_vec[ii]}.csv",
            sep=",",
            index_col=0
        )
        stats.index = [str(s).rjust(8, "0") for s in stats.index]
        nse = stats["nse"]

        # plot US map
        ax_map.set_xlim(-128, -65)
        ax_map.set_ylim(24, 50)
        ax_map.set_axis_off()
        us_states.boundary.plot(color="black", ax=ax_map, linewidth=0.5)

        color = np.clip(stat_global_a["nse"] - nse, -1, 1)

        ax_map.scatter(
            x=lon,
            y=lat,
            c=color,
            s=100,
            cmap=cm.roma_r,
            vmin=vmin,
            vmax=vmax
        )

        ax_map.set_title(
            f"NSE {experiment.upper()}-A minus {experiment.upper()}-{encoded_features_vec[ii]}",
            fontsize=35,
            y=-0.2
        )

        # plot scatter
        ax_scatter.set_xlabel(f"NSE {experiment.upper()}-A", fontsize=35)
        ax_scatter.set_ylabel(f"NSE {experiment.upper()}-{encoded_features_vec[ii]}", fontsize=35)
        ax_scatter.plot(line, line, c="black", ls="--")

        sc = ax_scatter.scatter(
            np.clip(stat_global_a["nse"], 0, 1),
            np.clip(nse, 0, 1),
            c=color,
            s=100,
            cmap=cm.roma_r,
            vmin=vmin,
            vmax=vmax
        )

        cb = fig.colorbar(sc, ax=ax_scatter)
        cb.set_label(r"$\Delta$ NSE", fontsize=25)

        ax_scatter.grid()

    fig.tight_layout()
    fig.savefig(
        f"analysis/figures/delta_nse_attr_{experiment}_{eval_period}.png",
        dpi=300
    )
