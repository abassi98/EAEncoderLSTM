import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from src.plot_utils import load_us_states
from matplotlib.colors import ListedColormap, BoundaryNorm
import pickle
import argparse
from src.utils import str2bool,  get_basin_list, clean_and_capitalize, attribute_draw_style, compute_grid
from src.datautils import LANDSCAPE_ATTRS, CLIM_ATTRS, HYDRO_ATTRS, SOIL_ATTRS, VEGE_ATTRS, TOPO_ATTRS, GEOL_ATTRS
from pathlib import PosixPath
from src.datautils import load_attributes
from matplotlib.lines import Line2D

import matplotlib as mpl
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
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--encoded_features',
        type=int,
        default=2,
        help="Encoded space dimension"
    )
    

    parser.add_argument(
        "--experiment",
        type=str,
        default="global"
    )

   
    cfg = vars(parser.parse_args())
    return cfg

def augment_with_noise(data, target_size=2000, noise_std=0.01):
    original_size = data.size(0)
    assert target_size >= original_size
    new_data = []
    for ii in range(target_size-original_size):
        ind = np.random.choice(np.arange(original_size))
        new_sample = data[ind, :] + torch.randn_like(data[ind, :]) * noise_std
        new_data.append(new_sample)

    new_data = torch.stack(new_data, dim=0)
    augmented_data = torch.cat((data, new_data), dim=0)
    return augmented_data

if __name__ == '__main__':
    ##########################################################
    # Load encoded features of chosen LSTM-AE model
    ##########################################################
  
    # Load encoded features
    cfg = get_args()
    encoded_features = cfg["encoded_features"]
    experiment = cfg["experiment"]

    us_states = load_us_states()
    eval_period = "test"

    savefile = f"analysis/gradients/{experiment}_es{encoded_features}_{eval_period}.pkl"
    with open(savefile, "rb") as f:
        grads = pickle.load(f)
    
    
    ### load static attrs
    basins =  get_basin_list()

    # load attributes
    camels_root = PosixPath("../Datasets/CAMELS-US/")
    #add_camels_attributes(camels_root)
    KEEP_ATTRS = SOIL_ATTRS + CLIM_ATTRS + VEGE_ATTRS + TOPO_ATTRS + GEOL_ATTRS + ["gauge_lat", "gauge_lon"]
    df_S = load_attributes("data/attributes.db", basins, keep_attributes=KEEP_ATTRS)
    # insert lat and lon in encoded attributes
    lon = df_S["gauge_lon"]
    lat = df_S["gauge_lat"]
    df_S = df_S.drop(['gauge_lat', 'gauge_lon'], axis=1)
    attrs_name = [clean_and_capitalize(a) for a in df_S.columns]

    nruns = 4
    nrows, ncols = compute_grid(nruns)
    fig, axs = plt.subplots(nrows, ncols, figsize=(30,20))
    axs = axs.flatten()
    ### plot on map
    for run in range(nruns):
        ax = axs[run]
        ax.set_title(f"Run {run}", fontsize=25)
        ax.set_xlim(-128, -65)
        ax.set_ylim(24, 50)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        us_states.boundary.plot(color="black", ax=ax, linewidth=0.5)

        features = []
        for basin in basins:
            # plot on map
            grads_run_jj = grads[basin][run,:]
            grads_run_jj =  (grads_run_jj - grads_run_jj.min()) / (grads_run_jj.max() - grads_run_jj.min())
            features.append(grads_run_jj)
            best_feature_index = np.argmax(grads_run_jj)
            attr_name = df_S.columns[best_feature_index]
            marker = attribute_draw_style[attr_name]["marker"]
            color = attribute_draw_style[attr_name]["color"]
            ax.scatter(x=lon.loc[basin], y=lat.loc[basin], c=color,  s=200, marker=marker, edgecolor='black', alpha=0.8)
        
        df_sensistivity = pd.DataFrame(np.array(features).mean(0), index=attrs_name)
        print(df_sensistivity.sort_values(0, ascending=False))


    legend_items = {}

    for attr_name, style in attribute_draw_style.items():

        label = clean_and_capitalize(attr_name)

        # avoid duplicates
        if label not in legend_items:

            legend_items[label] = Line2D(
                [0],
                [0],
                marker=style["marker"],
                color='w',
                label=label,
                markerfacecolor=style["color"],
                markeredgecolor='black',
                markersize=20,
                linewidth=0
            )
    

    # create legend
    fig.legend(
        handles=list(legend_items.values()),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.0),
        ncol=3,
        fontsize=20
    )

    # reserve space at bottom
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    fig.savefig(f"analysis/figures/morris_{experiment}_es{encoded_features}.png", dpi=300)
        
