
import numpy as np
import matplotlib.pyplot as plt
from dadapy import data

from src.utils import get_basin_list
from src.datautils import load_attributes, CLIM_ATTRS, TOPO_ATTRS, GEOL_ATTRS, SOIL_ATTRS, VEGE_ATTRS


from src.plot_style import configure_plot_style, LABEL_FONTSIZE

configure_plot_style()



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
    ax.set_xlabel(r"Neighbourhood average distance", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(r"Intrinsic dimension (ID)", fontsize=LABEL_FONTSIZE)
   
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()

    plt.savefig("analysis/figures/gride_known_attr.pdf", dpi=300, bbox_inches="tight", format="pdf")
    

    return ids_gride, ids_err_gride, rs_gride

if __name__=="__main__":
    plot()