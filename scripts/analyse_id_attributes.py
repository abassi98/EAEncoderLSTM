
import numpy as np
import matplotlib.pyplot as plt
from dadapy import data

from src.utils import get_basin_list
from src.datautils import load_attributes, CLIM_ATTRS, TOPO_ATTRS, GEOL_ATTRS, SOIL_ATTRS, VEGE_ATTRS


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


def plot():
    basins = get_basin_list()

    keep_attrs = (
        SOIL_ATTRS + CLIM_ATTRS + VEGE_ATTRS +
        TOPO_ATTRS + GEOL_ATTRS + ["gauge_lat", "gauge_lon"]
    )

    df_S = load_attributes(
        "data/attributes.db",
        basins,
        keep_attributes=keep_attrs
    )

    lat = df_S["gauge_lat"]
    lon = df_S["gauge_lon"]

    df = df_S.drop(columns=["gauge_lat", "gauge_lon"])

    # clean numeric matrix
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=1, how="all")
    df = df.fillna(df.mean(numeric_only=True))

    # z-score standardization
    df = (df - df.mean()) / df.std()
    df = df.dropna(axis=1)  # removes zero-variance attributes

    X = df.to_numpy(dtype=float)

    # DADApy GRIDE intrinsic dimension
    d = data.Data(coordinates=X, maxk=min(100, X.shape[0] - 1))
    ids_gride, ids_err_gride, rs_gride = d.return_id_scaling_gride(
        range_max=X.shape[0]
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        rs_gride,
        ids_gride,
        yerr=ids_err_gride,
        marker="o",
        capsize=4,
        linewidth=2,
        c="black",
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"Neighbourhood average distance", fontsize=22)
    ax.set_ylabel(r"Intrinsic dimension (ID)", fontsize=22)
   
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    plt.savefig("analysis/figures/gride_known_attr.png", dpi=300, bbox_inches="tight")
    

    return ids_gride, ids_err_gride, rs_gride

if __name__=="__main__":
    plot()