import numpy as np
import matplotlib.pyplot as plt
from src.plot_utils import load_us_states
import pickle
import argparse
from src.utils import str2bool,  get_basin_list, clean_and_capitalize, compute_grid, attribute_draw_style
from src.datautils import LANDSCAPE_ATTRS, CLIM_ATTRS, HYDRO_ATTRS, SOIL_ATTRS, VEGE_ATTRS, TOPO_ATTRS, GEOL_ATTRS
from pathlib import PosixPath
from src.datautils import load_attributes
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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
        "--with_pca",
        type=str2bool,
        default=False,
        help="Analyse PC of data"
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="global"
    )

  
    parser.add_argument(
        "--attributes",
        type=str,
        nargs='+',
        default="all",  
        choices=CLIM_ATTRS + HYDRO_ATTRS  +  LANDSCAPE_ATTRS + ["all"])
    
    parser.add_argument(
        "--nn",
        type=int,
        default=1)
 
    parser.add_argument(    "--run",
        type=int,
        default=0,
        help="Run number to plot (0-3)"
    )
    parser.add_argument(
        '--proximity',
        type=str,
        default="attrs",
        choices=["attrs", "geo"],
        help='Method to determine proximity of basins (attributes or geographic)'
    )

    parser.add_argument(
        "--fig_scale",
        type=float,
        default=15.0,
    )

    parser.add_argument(
        "--with_ica",
        type=str2bool,
        default=False,
        help="Use ICA for dimensionality reduction"
    )

    parser.add_argument(    "--rank",
        type=int,
        default=0,
        help="Importance of the features plotted"
    )   
    cfg = vars(parser.parse_args())

    if "all" in cfg["attributes"]:
        cfg["attributes"] = CLIM_ATTRS +  LANDSCAPE_ATTRS 

    return cfg

def haversine(lat1, lon1, lat2, lon2, radius=6371.0):
    """
    Vectorized haversine distance.

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : pandas Series or numpy arrays
        Coordinates in degrees.
    radius : float
        Earth radius (6371 km or 3959 miles)

    Returns
    -------
    pandas Series
        Distance along Earth's surface.
    """

    # convert degrees to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return radius * c


from typing import Dict, Tuple

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon


def ecdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate empirical cummulative density function
    
    Parameters
    ----------
    x : np.ndarray
        Array containing the data
    
    Returns
    -------
    x : np.ndarray
        Array containing the sorted metric values
    y : np.ndarray]
        Array containing the sorted cdf values
    """
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    return xs, ys


def get_shape_collections(data: Dict):
    shapes = []

    for points in data.values():
        shapes.append(Polygon(np.array([points['lons'], points['lats']]).T, closed=True))

    collection = PatchCollection(shapes)
    collection.set_facecolor('#eeeeee')
    collection.set_edgecolor('black')
    collection.set_linewidth(0.2)

    return collection



def bin_cosine_stats(vectors):
    
    # edge cases: 0 or 1 vector
    if len(vectors) < 2:
        return pd.Series({"cosine_mean": np.nan, "cosine_std": np.nan})

    # cosine similarity matrix
    sim = cosine_similarity(vectors)

    # extract upper triangle (exclude diagonal)
    sims = np.triu(sim, k=1).flatten()
    sims = sims[sims!=0]


    return pd.Series({
        "cosine_mean": sims.mean(),
        "cosine_std": sims.std()
    })


if __name__ == '__main__':
    ##########################################################
    # Load encoded features of chosen LSTM-AE model
    ##########################################################
    
    # Load encoded features
    cfg = get_args()
    encoded_features = cfg["encoded_features"]
    with_pca = cfg["with_pca"]
    experiment = cfg["experiment"]
    nn = cfg["nn"]
    rank = cfg["rank"]
    proximity = cfg["proximity"]
    fig_scale = cfg["fig_scale"]
    with_ica = cfg["with_ica"]
    nruns = 4
    us_states = load_us_states()
    savefile = f"analysis/gradients/grads_encoder_{experiment}.pkl"
    attributes = cfg["attributes"]
    nattr = len(attributes)
    nrows, ncols = compute_grid(nattr)

    with open(f"analysis/gradients/encoder_{experiment}_es{encoded_features}_ica{with_ica}.pkl", "rb") as f:
        grads_dict = pickle.load(f)
    

    ### load static attrs
    basins =  get_basin_list()
    num_basins = len(basins)
    # transform dict
    grads = np.zeros((num_basins, nruns, encoded_features, nattr))
    for ii, basin in enumerate(basins):
        grads[ii, :, :, :] = grads_dict[basin]

    
    # load attributes
    camels_root = PosixPath("../Datasets/CAMELS-US/")
    #add_camels_attributes(camels_root)
    KEEP_ATTRS = SOIL_ATTRS + CLIM_ATTRS + VEGE_ATTRS + TOPO_ATTRS + GEOL_ATTRS + ["gauge_lat", "gauge_lon"]
    df_S = load_attributes("data/attributes.db", basins, keep_attributes=KEEP_ATTRS)

    # insert lat and lon in encoded attributes
    lon = df_S["gauge_lon"]
    lat = df_S["gauge_lat"]
    df_S = df_S.drop(['gauge_lat', 'gauge_lon'], axis=1)
    df_attributes = df_S[cfg["attributes"]]
   
    
    # scale attributes
    scaler = StandardScaler()
    attributes_scaled = scaler.fit_transform(df_attributes)  
    df_attributes_scaled = pd.DataFrame(attributes_scaled, index=basins, columns=df_attributes.columns)

    fig, axs = plt.subplots(1, encoded_features,figsize=(fig_scale*encoded_features,fig_scale), squeeze=False)
    run = cfg["run"]
    for ii_es in range(encoded_features):
        df = pd.DataFrame(index=basins, data=grads[:, run, ii_es, :])
        mean_sim = []
        std_sim = []
        ax = axs[0,ii_es]
        ax.set_xlim(-128, -65)
        ax.set_ylim(24, 50)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        us_states.boundary.plot(color="black", ax=ax, linewidth=0.5)
        
        sensitivity = []
        
        for basin in basins:
            
            if cfg["proximity"] == "geo":
                # compute geographic distance
                dist = haversine(lat[basin], lon[basin], lat, lon)
            else:
                # compute attribute distance
                dist = np.sqrt(((df_attributes_scaled.loc[basin,:]- df_attributes_scaled)**2).sum(axis=1))

            #dist.drop(index=[basin], inplace=True)
            dist.sort_values(inplace=True)
            ball = dist.index[:nn]
            
            df_loc = df.loc[ball,:]
            
            similarities = bin_cosine_stats(df_loc.values)
            mean_sim.append(similarities["cosine_mean"])
            std_sim.append(similarities["cosine_std"])
            results = df_loc.mean(axis=0)
            results = np.abs(results) 
            results = (results - results.min()) / (results.max() - results.min())

            sensitivity.append(results)
            
            order = np.argsort(results)
            best_feature_index = order[25-rank]
            attr_name = df_attributes.columns[best_feature_index]
            
            marker = attribute_draw_style[attr_name]["marker"]
            color = attribute_draw_style[attr_name]["color"]

            ax.scatter(x=lon.loc[basin], y=lat.loc[basin], c=color,  s=200, marker=marker, edgecolor='black', alpha=0.8)

        sensitivity = pd.DataFrame(np.array(sensitivity), index=basins, columns=df_attributes.columns)
        #print(sensitivity.sum(axis=0))
        mean_sim = np.array(mean_sim).mean()
        std_sim = np.array(std_sim).mean()
        print(f"Run {run+1}: Encoded feature {ii_es+1}: mean cosine similarity = {mean_sim:.4f}, std = {std_sim:.4f}")
        
    
        print(sensitivity.mean(axis=0).sort_values(ascending=False))
        if run == 0:
            ax.set_title(f"Encoded feature {ii_es+1}", fontsize=30)
        
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
    

    fig.savefig(f"analysis/figures/grad_sim_{cfg['proximity']}_nn{cfg['nn']}_{experiment}_run{run}_es{encoded_features}_rank{rank}.png", dpi=300)
    
