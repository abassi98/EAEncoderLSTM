import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from src.plot_utils import add_basemap, load_us_states
from scripts.plot_encoded_feature_scatter import (
    build_global_reference_values,
    load_encoded_feature_tensor,
)
from src.utils import str2bool,  get_basin_list, clean_and_capitalize, compute_grid, attribute_draw_style
from src.datautils import LANDSCAPE_ATTRS, CLIM_ATTRS, HYDRO_ATTRS, SOIL_ATTRS, VEGE_ATTRS, TOPO_ATTRS, GEOL_ATTRS
from pathlib import PosixPath
from src.datautils import load_attributes
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.linalg import orthogonal_procrustes
from matplotlib.lines import Line2D

import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 20 
mpl.rcParams['ytick.labelsize'] = 20 
#matplotlib.font_manager.findfont("Symbol")
mpl.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')[:10]
mpl.rc('text', usetex=True)
mpl.rc(
    'text.latex',
    preamble=r'\usepackage{amsmath} \usepackage{amsfonts} \usepackage{xcolor}'
)

# Say, "the default sans-serif font is COMIC SANS"
plt.rcParams['font.serif'] = "Times New Roman"
# Then, "ALWAYS use sans-serif fonts"
plt.rcParams['font.family'] = "serif"
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
#plt.rcParams['font.weight'] = 'bold'
mpl.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsfonts} \usepackage{xcolor}')

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


def latex_attr_with_color(label):
    attr = LABEL_TO_ATTR.get(label)
    color = ATTR_TEXT_COLORS.get(attr)

    if color is None:
        return label

    return rf"\textcolor[HTML]{{{color}}}{{{label}}}"


def get_args():
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()


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

    parser.add_argument(
        "--encoded_features",
        type=int,
        default=2,
        help="Number of latent features to analyze"
    )
 
    parser.add_argument("--run",
        type=int,
        default=None,
        help="Optional run number to plot (0-3). If omitted, align runs and average sensitivities across runs."
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

    parser.add_argument(
        "--basemap",
        type=str,
        default="satellite",
        choices=["none", "light", "terrain", "satellite"],
        help="Background basemap style"
    )

    parser.add_argument(    "--rank",
        type=int,
        default=0,
        help="Importance of the features plotted"
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
    cfg = vars(parser.parse_args())

    if cfg["encoded_features"] < 1:
        raise ValueError("--encoded_features must be greater than or equal to 1")

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


def compute_run_alignment_operators(
    feature_tensor: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Mirror the run-wise Procrustes alignment used by the encoded-feature plots."""
    if feature_tensor.shape[1] <= 1:
        n_features = feature_tensor.shape[2]
        return feature_tensor.copy(), [np.eye(n_features)], np.ones(n_features, dtype=float)

    run_means = np.nanmean(feature_tensor, axis=0, keepdims=True)
    centered_tensor = feature_tensor - run_means
    reference = centered_tensor[:, 0, :].copy()
    aligned_centered = centered_tensor.copy()
    rotations = [np.eye(feature_tensor.shape[2]) for _ in range(feature_tensor.shape[1])]

    for _ in range(32):
        previous_reference = reference.copy()
        for run in range(centered_tensor.shape[1]):
            rotation, _ = orthogonal_procrustes(centered_tensor[:, run, :], reference)
            rotations[run] = rotation
            aligned_centered[:, run, :] = centered_tensor[:, run, :] @ rotation

        reference = np.nanmean(aligned_centered, axis=1)
        denominator = np.linalg.norm(previous_reference) + 1e-12
        if np.linalg.norm(reference - previous_reference) / denominator < 1e-6:
            break

    signs = np.ones(aligned_centered.shape[2], dtype=float)
    reference_run = centered_tensor[:, 0, :]
    for feature_idx in range(aligned_centered.shape[2]):
        ref_feature = reference_run[:, feature_idx]
        aligned_feature = reference[:, feature_idx]
        if np.std(ref_feature) == 0.0 or np.std(aligned_feature) == 0.0:
            continue

        corr = np.corrcoef(ref_feature, aligned_feature)[0, 1]
        if np.isfinite(corr) and corr < 0.0:
            signs[feature_idx] = -1.0

    aligned_centered *= signs.reshape(1, 1, -1)
    consensus_mean = np.nanmean(run_means, axis=1, keepdims=True)
    return aligned_centered + consensus_mean, rotations, signs


def compute_consensus_ica_operator(
    aligned_feature_tensor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one ICA basis on the consensus mean space and return its gradient operator."""
    from sklearn.decomposition import FastICA

    reference_values = np.nanmean(aligned_feature_tensor, axis=1)
    ica = FastICA(
        n_components=aligned_feature_tensor.shape[2],
        whiten=False,
        random_state=0,
        max_iter=1000,
    )
    reference_transformed = ica.fit_transform(reference_values)
    signs = np.ones(reference_transformed.shape[1], dtype=float)
    for component_idx in range(reference_transformed.shape[1]):
        if np.nanmean(reference_transformed[:, component_idx]) < 0.0:
            signs[component_idx] = -1.0
    return ica.components_.copy(), signs


def compute_global_sign_operator(
    basins: list[str],
    feature_tensor: np.ndarray,
    encoded_features: int,
    with_ica: bool,
) -> np.ndarray:
    """Match PUB feature signs to the GLOBAL reference used by the encoded-feature plots."""
    reference_basins, reference_values = build_global_reference_values(
        encoded_features=encoded_features,
        with_ica=with_ica,
        with_pca=False,
    )
    reference_index = {basin: idx for idx, basin in enumerate(reference_basins)}
    basin_pairs = [
        (basin_idx, ref_idx)
        for basin_idx, basin in enumerate(basins)
        if (ref_idx := reference_index.get(basin)) is not None
    ]
    if not basin_pairs:
        return np.ones(feature_tensor.shape[2], dtype=float)

    basin_idx = np.array([pair[0] for pair in basin_pairs], dtype=int)
    ref_idx = np.array([pair[1] for pair in basin_pairs], dtype=int)
    summary_values = np.nanmean(feature_tensor, axis=1)[basin_idx, :]
    reference_subset = reference_values[ref_idx, :]
    signs = np.ones(feature_tensor.shape[2], dtype=float)

    for feature_idx in range(summary_values.shape[1]):
        target_feature = summary_values[:, feature_idx]
        reference_feature = reference_subset[:, feature_idx]
        mask = np.isfinite(target_feature) & np.isfinite(reference_feature)
        if np.count_nonzero(mask) < 2:
            continue
        if np.nanstd(target_feature[mask]) == 0.0 or np.nanstd(reference_feature[mask]) == 0.0:
            continue

        corr = np.corrcoef(target_feature[mask], reference_feature[mask])[0, 1]
        if np.isfinite(corr) and corr < 0.0:
            signs[feature_idx] = -1.0

    return signs


def apply_left_operator(
    gradient_tensor: np.ndarray,
    operator: np.ndarray,
) -> np.ndarray:
    """Apply one feature-space linear operator to [n_basins, n_features, n_attrs] gradients."""
    return np.einsum("ij,bjk->bik", operator, gradient_tensor)


def transform_features_and_gradients(
    basins: list[str],
    feature_tensor: np.ndarray,
    gradient_tensor: np.ndarray,
    model: str,
    encoded_features: int,
    with_ica: bool,
    align_runs: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same latent-space transforms as plot_encoded_features.py to gradients."""
    transformed_features = feature_tensor.copy()
    transformed_gradients = gradient_tensor.copy()

    if align_runs:
        transformed_features, rotations, procrustes_signs = compute_run_alignment_operators(
            transformed_features
        )
        procrustes_operator = np.diag(procrustes_signs)
        for run in range(transformed_gradients.shape[1]):
            left_operator = procrustes_operator @ rotations[run].T
            transformed_gradients[:, run, :, :] = apply_left_operator(
                transformed_gradients[:, run, :, :],
                left_operator,
            )

    if with_ica:
        ica_matrix, ica_signs = compute_consensus_ica_operator(transformed_features)
        ica_operator = np.diag(ica_signs) @ ica_matrix
        for run in range(transformed_gradients.shape[1]):
            transformed_gradients[:, run, :, :] = apply_left_operator(
                transformed_gradients[:, run, :, :],
                ica_operator,
            )
        for run in range(transformed_features.shape[1]):
            transformed_features[:, run, :] = transformed_features[:, run, :] @ ica_matrix.T
            transformed_features[:, run, :] *= ica_signs.reshape(1, -1)

    if model == "pub":
        global_signs = compute_global_sign_operator(
            basins=basins,
            feature_tensor=transformed_features,
            encoded_features=encoded_features,
            with_ica=with_ica,
        )
        transformed_features *= global_signs.reshape(1, 1, -1)
        transformed_gradients *= global_signs.reshape(1, 1, -1, 1)

    return transformed_features, transformed_gradients


def compute_sensitivity_frame(
    df: pd.DataFrame,
    basins: list[str],
    lat: pd.Series,
    lon: pd.Series,
    df_S_scaled: pd.DataFrame,
    proximity: str,
    nn: int,
) -> tuple[pd.DataFrame, float, float]:
    """Compute local sensitivity vectors for one run and one latent feature."""
    sensitivity = []
    mean_sim = []
    std_sim = []

    for basin in basins:
        if proximity == "geo":
            dist = haversine(lat[basin], lon[basin], lat, lon)
        else:
            dist = np.sqrt(((df_S_scaled.loc[basin, :] - df_S_scaled) ** 2).sum(axis=1))

        dist.sort_values(inplace=True)
        ball = dist.index[:nn]

        df_loc = df.loc[ball, :]
        similarities = bin_cosine_stats(df_loc.values)
        mean_sim.append(similarities["cosine_mean"])
        std_sim.append(similarities["cosine_std"])

        results = df_loc.mean(axis=0)
        results = np.abs(results)
        denominator = results.max() - results.min()
        if denominator > 0.0:
            results = (results - results.min()) / denominator
        else:
            results = np.zeros_like(results, dtype=float)

        sensitivity.append(results)

    sensitivity = pd.DataFrame(np.array(sensitivity), index=basins, columns=df.columns)
    return sensitivity, np.nanmean(mean_sim), np.nanmean(std_sim)


if __name__ == '__main__':
    ##########################################################
    # Load encoded features of chosen LSTM-AE model
    ##########################################################
    
    # Load encoded features
    cfg = get_args()
    experiments = ["global", "pub"]
    encoded_features = cfg["encoded_features"]
    nn = cfg["nn"]
    rank = cfg["rank"]
    proximity = cfg["proximity"]
    fig_scale = cfg["fig_scale"]
    with_ica = cfg["with_ica"]
    basemap = cfg["basemap"]
    if encoded_features == 1:
        with_ica = False
    nruns = 4
    us_states = load_us_states()

    

    # load raw global and pub grads; latent-space rotations are applied below to
    # match the encoded-feature plotting workflow.
    with open(f"analysis/gradients/encoder_global_es{encoded_features}_icaFalse.pkl", "rb") as f:
        grads_global_dict = pickle.load(f)
    
    with open(f"analysis/gradients/encoder_pub_es{encoded_features}_icaFalse.pkl", "rb") as f:
        grads_pub_dict = pickle.load(f)

    global_feature_basins, global_feature_tensor = load_encoded_feature_tensor(
        model="global",
        encoded_features=encoded_features,
    )
    pub_feature_basins, pub_feature_tensor = load_encoded_feature_tensor(
        model="pub",
        encoded_features=encoded_features,
    )
    global_gradient_tensor = np.stack(
        [np.asarray(grads_global_dict[basin], dtype=float) for basin in global_feature_basins],
        axis=0,
    )
    pub_gradient_tensor = np.stack(
        [np.asarray(grads_pub_dict[basin], dtype=float) for basin in pub_feature_basins],
        axis=0,
    )
    _, global_gradient_tensor = transform_features_and_gradients(
        basins=global_feature_basins,
        feature_tensor=global_feature_tensor,
        gradient_tensor=global_gradient_tensor,
        model="global",
        encoded_features=encoded_features,
        with_ica=with_ica,
        align_runs=(cfg["run"] is None) or with_ica,
    )
    _, pub_gradient_tensor = transform_features_and_gradients(
        basins=pub_feature_basins,
        feature_tensor=pub_feature_tensor,
        gradient_tensor=pub_gradient_tensor,
        model="pub",
        encoded_features=encoded_features,
        with_ica=with_ica,
        align_runs=(cfg["run"] is None) or with_ica,
    )
    grads_global_dict = {
        basin: global_gradient_tensor[ii, :, :, :]
        for ii, basin in enumerate(global_feature_basins)
    }
    grads_pub_dict = {
        basin: pub_gradient_tensor[ii, :, :, :]
        for ii, basin in enumerate(pub_feature_basins)
    }

    ### load basins
    all_basins = get_basin_list()
    basins = all_basins.copy()
    global_none_stats = pd.read_csv("analysis/stats/test/global_esNone.csv", sep=",", index_col=0)
    global_none_stats.index = [str(s).rjust(8, "0") for s in global_none_stats.index]
    threshold = cfg["performance_threshold"]
    basins = [basin for basin in basins if basin in global_none_stats.index and global_none_stats.loc[basin, "nse"] > threshold]
    below_threshold_basins = [basin for basin in all_basins if basin not in basins]
    num_basins = len(basins)
    

    # transform dict
    grads_global = np.zeros((num_basins, nruns, encoded_features, 26))
    grads_pub = np.zeros((num_basins, nruns, encoded_features, 26))
    for ii, basin in enumerate(basins):
        grads_global[ii, :, :, :] = grads_global_dict[basin]
        grads_pub[ii, :, :, :] = grads_pub_dict[basin]

    

    
    # load attributes
    camels_root = PosixPath("../Datasets/CAMELS-US/")
    #add_camels_attributes(camels_root)
    KEEP_ATTRS = SOIL_ATTRS + CLIM_ATTRS + VEGE_ATTRS + TOPO_ATTRS + GEOL_ATTRS + ["gauge_lat", "gauge_lon"]
    df_S = load_attributes("data/attributes.db", basins, keep_attributes=KEEP_ATTRS)
    df_all_locations = load_attributes(
        "data/attributes.db",
        all_basins,
        keep_attributes=["gauge_lat", "gauge_lon"],
    )
    
    # insert lat and lon in encoded attributes
    lon = df_S["gauge_lon"]
    lat = df_S["gauge_lat"]
    df_S = df_S.drop(['gauge_lat', 'gauge_lon'], axis=1)
    df_S = df_S[cfg["attributes"]]
    nattr = df_S.shape[1]
    nrows, ncols = compute_grid(nattr)

    
    # scale attributes
    scaler = StandardScaler()
    attributes_scaled = scaler.fit_transform(df_S)  
    df_S_scaled = (df_S - df_S.mean())/df_S.std()

    if encoded_features == 1:
        fig, axs = plt.subplots(
            1,
            len(experiments),
            figsize=(fig_scale * len(experiments), fig_scale * 0.75),
            squeeze=False,
        )
    else:
        fig, axs = plt.subplots(
            2,
            encoded_features,
            figsize=(fig_scale * encoded_features, fig_scale * 1.3),
            squeeze=False,
        )

    run = cfg["run"]
    results_dict = {}
    for ii_es in range(encoded_features):
        for ii_exp, exp in enumerate(experiments):
            if encoded_features == 1:
                ax = axs[0, ii_exp]
            else:
                ax = axs[ii_exp, ii_es]
            if exp == "global":
                grads_tensor = grads_global
            elif exp == "pub":
                grads_tensor = grads_pub
            else:
                raise ValueError(f"Unknown experiment: {exp}")

            if run is None:
                sensitivity_runs = []
                mean_sim_runs = []
                std_sim_runs = []
                for run_idx in range(nruns):
                    df_run = pd.DataFrame(index=basins, data=grads_tensor[:, run_idx, ii_es, :])
                    sensitivity_run, mean_sim_run, std_sim_run = compute_sensitivity_frame(
                        df=df_run,
                        basins=basins,
                        lat=lat,
                        lon=lon,
                        df_S_scaled=df_S_scaled,
                        proximity=cfg["proximity"],
                        nn=nn,
                    )
                    sensitivity_runs.append(sensitivity_run)
                    mean_sim_runs.append(mean_sim_run)
                    std_sim_runs.append(std_sim_run)

                sensitivity_stack = np.stack(
                    [sensitivity_run.to_numpy(dtype=float) for sensitivity_run in sensitivity_runs],
                    axis=0,
                )
                sensitivity = pd.DataFrame(
                    np.nanmean(sensitivity_stack, axis=0),
                    index=basins,
                    columns=df_S.columns,
                )
                mean_sim = float(np.nanmean(mean_sim_runs))
                std_sim = float(np.nanmean(std_sim_runs))
            else:
                df = pd.DataFrame(index=basins, data=grads_tensor[:, run, ii_es, :])
                sensitivity, mean_sim, std_sim = compute_sensitivity_frame(
                    df=df,
                    basins=basins,
                    lat=lat,
                    lon=lon,
                    df_S_scaled=df_S_scaled,
                    proximity=cfg["proximity"],
                    nn=nn,
                )
            
            ax.set_xlim(-128, -65)
            ax.set_ylim(24, 50)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            add_basemap(ax, basemap)
            us_states.boundary.plot(color="black", ax=ax, linewidth=0.5, zorder=2)
            ax.scatter(
                x=df_all_locations.loc[below_threshold_basins, "gauge_lon"],
                y=df_all_locations.loc[below_threshold_basins, "gauge_lat"],
                s=100,
                facecolors="none",
                edgecolors="#000000",
                linewidth=2.0,
                alpha=1.0,
                zorder=2.5,
            )

            for basin in basins:
                results = sensitivity.loc[basin, :].to_numpy(dtype=float)
                order = np.argsort(results)
                best_feature_index = order[len(results) - 1 - rank]
                attr_name = df_S.columns[best_feature_index]
                
                marker = attribute_draw_style[attr_name]["marker"]
                color = attribute_draw_style[attr_name]["color"]

                ax.scatter(
                    x=lon.loc[basin],
                    y=lat.loc[basin],
                    c=color,
                    s=400,
                    marker=marker,
                    edgecolor='black',
                    alpha=0.8,
                    zorder=3,
                )
            mean_sensitivity = sensitivity.mean(axis=0).sort_values(ascending=False)
            mean_sensitivity.index = [clean_and_capitalize(ind) for ind in mean_sensitivity.index]

            results_dict[f"{exp.upper()}-{ii_es+1}"] = mean_sensitivity

            if encoded_features == 1:
                if run is None:
                    ax.set_title(f"{exp.upper()}-{ii_es+1} mean", fontsize=40)
                else:
                    ax.set_title(f"{exp.upper()}-{ii_es+1}", fontsize=40)
            elif ii_exp == 0:
                ax.set_title(f"Latent feature {ii_es+1}", fontsize=40)
            if ii_es == 0 and encoded_features != 1:
                if run is None:
                    ax.set_ylabel(f"{exp.upper()}-{encoded_features} mean", fontsize=40)
                else:
                    ax.set_ylabel(f"{exp.upper()}-{encoded_features}", fontsize=40)
    
  
    table_dict = {}

    for key, series in results_dict.items():

        formatted = [
            rf"{latex_attr_with_color(idx)} ({val:.2f})"
            for idx, val in zip(series.index, series.values)
        ]

        table_dict[key] = formatted

    ordered_columns = []
    multiindex_columns = []
    for model_name in ["PUB", "GLOBAL"]:
        for lf_idx in range(1, encoded_features + 1):
            key = f"{model_name}-{lf_idx}"
            if key not in table_dict:
                continue
            ordered_columns.append(key)
            multiindex_columns.append((model_name, f"LF{lf_idx}"))

    latex_df = pd.DataFrame({key: table_dict[key] for key in ordered_columns})
    latex_df.columns = pd.MultiIndex.from_tuples(multiindex_columns)

    print(latex_df.to_latex(index=False, escape=False))
   
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
    
    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.93,
        bottom=0.28,
        hspace=0.02,
        wspace=0.02
    )
    # create legend
    fig.legend(
        handles=list(legend_items.values()),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=5,
        fontsize=30
    )
    

    run_label = "mean" if run is None else f"run{run}"
    fig.savefig(
        f"analysis/figures/grad_sim_{cfg['proximity']}_nn{cfg['nn']}_{run_label}_es{encoded_features}_ica{with_ica}_rank{rank}.png",
        dpi=300,
    )
    
