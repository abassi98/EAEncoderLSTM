
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import clean_and_capitalize, get_basin_list
from src.datautils import HYDRO_ATTRS, load_attributes, CLIM_ATTRS, TOPO_ATTRS, GEOL_ATTRS, SOIL_ATTRS, VEGE_ATTRS
from cmcrameri import cm
from src.plot_style import configure_plot_style, LABEL_FONTSIZE, TICK_FONTSIZE

configure_plot_style()



def plot():
    basins = get_basin_list()
    keep = CLIM_ATTRS + HYDRO_ATTRS + TOPO_ATTRS + GEOL_ATTRS + SOIL_ATTRS + VEGE_ATTRS
    df_S = load_attributes("data/attributes.db", basins, keep_attributes=keep)
    
    ### plot correalation matrix ES Spearman 
    fig, axs = plt.subplots(1,1, figsize=(25,20))
    columns = df_S.columns
    names = []
    for c in columns:
        names.append(clean_and_capitalize(c))
    corr = np.array(np.abs(df_S.corr("spearman")))

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    fig.subplots_adjust(left=.5, right=1.0) 
    g = sns.heatmap(corr, mask=mask, annot=True,fmt=".2f", xticklabels=names, yticklabels=names,cmap=cm.batlow,vmin=0, vmax=1, ax=axs , rasterized=True)#cbar_kws={'label': 'Absolute Spearman Correlation'})

    lines = [0, 8, 20, 23, 25, 33, 38]
    for l in lines:
        axs.hlines(l, xmin=-10, xmax=l, color='black',clip_on = False)
        axs.vlines(l, ymin=l+100, ymax=l, color='black',clip_on = False)

    
    axs.figure.axes[-1].yaxis.label.set_size(LABEL_FONTSIZE)
    axs.figure.axes[-1].tick_params(labelsize=TICK_FONTSIZE)

    g.set_xticklabels(g.get_xticklabels(), rotation = 90, fontsize=TICK_FONTSIZE)
    g.set_yticklabels(g.get_yticklabels(), rotation = 0, fontsize=TICK_FONTSIZE)

    
    # labels YAXIS
    fig.text(0.002, 0.85, "Climate", fontsize=LABEL_FONTSIZE, rotation="vertical")
    fig.text(0.002, 0.60, "Hydro. Signatures", fontsize=LABEL_FONTSIZE, rotation="vertical")
    fig.text(0.002, 0.5, "Topo.", fontsize=LABEL_FONTSIZE, rotation="vertical")
    fig.text(0.0002, 0.45, "Geo.", fontsize=LABEL_FONTSIZE, rotation="vertical")
    fig.text(0.002, 0.33, "Soil", fontsize=LABEL_FONTSIZE, rotation="vertical")
    fig.text(0.002, 0.2, "Vege.", fontsize=LABEL_FONTSIZE, rotation="vertical")

    # labels XAXIS
    fig.text( 0.2, 0.002, "Climate", fontsize=LABEL_FONTSIZE, rotation="horizontal")
    fig.text(0.33, 0.002, "Hydro. Signatures", fontsize=LABEL_FONTSIZE, rotation="horizontal", fontweight="bold")
    fig.text(0.51, 0.002, "Topo.", fontsize=LABEL_FONTSIZE, rotation="horizontal")
    fig.text(0.56,0.002, "Geo.", fontsize=LABEL_FONTSIZE, rotation="horizontal")
    fig.text(0.65, 0.002, "Soil", fontsize=LABEL_FONTSIZE, rotation="horizontal")
    fig.text(0.75, 0.002, "Vege.", fontsize=LABEL_FONTSIZE, rotation="horizontal")
   
    fig.tight_layout()
    fig.savefig("analysis/figures/KnownAttributes_spearman.pdf", dpi=300, bbox_inches="tight", format="pdf")

if __name__=="__main__":
    plot()