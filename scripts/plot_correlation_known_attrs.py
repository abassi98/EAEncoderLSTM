
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import clean_and_capitalize, get_basin_list
from src.datautils import HYDRO_ATTRS, load_attributes, CLIM_ATTRS, TOPO_ATTRS, GEOL_ATTRS, SOIL_ATTRS, VEGE_ATTRS
from cmcrameri import cm
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
    g = sns.heatmap(corr, mask=mask, annot=True,fmt=".2f", xticklabels=names, yticklabels=names,cmap=cm.batlow,vmin=0, vmax=1, ax=axs )#cbar_kws={'label': 'Absolute Spearman Correlation'})

    lines = [0, 8, 20, 23, 25, 33, 38]
    for l in lines:
        axs.hlines(l, xmin=-10, xmax=l, color='black',clip_on = False)
        axs.vlines(l, ymin=l+100, ymax=l, color='black',clip_on = False)

    
    axs.figure.axes[-1].yaxis.label.set_size(15)
    axs.figure.axes[-1].tick_params(labelsize=15)

    g.set_xticklabels(g.get_xticklabels(), rotation = 90, fontsize=20)
    g.set_yticklabels(g.get_yticklabels(), rotation = 0, fontsize=20)

    # labels YAXIS
    fig.text(0.002, 0.85, "Climate", fontsize=22, rotation="vertical")
    fig.text(0.002, 0.60, "Hydro. Signatures", fontsize=22, rotation="vertical")
    fig.text(0.002, 0.5, "Topo.", fontsize=22, rotation="vertical")
    fig.text(0.0002, 0.45, "Geo.", fontsize=22, rotation="vertical")
    fig.text(0.002, 0.33, "Soil", fontsize=22, rotation="vertical")
    fig.text(0.002, 0.2, "Vege.", fontsize=22, rotation="vertical")

    # labels XAXIS
    fig.text( 0.2, 0.002, "Climate", fontsize=22, rotation="horizontal")
    fig.text(0.33, 0.002, "Hydro. Signatures", fontsize=22, rotation="horizontal", fontweight="bold")
    fig.text(0.51, 0.002, "Topo.", fontsize=22, rotation="horizontal")
    fig.text(0.56,0.002, "Geo.", fontsize=22, rotation="horizontal")
    fig.text(0.65, 0.002, "Soil", fontsize=22, rotation="horizontal")
    fig.text(0.75, 0.002, "Vege.", fontsize=22, rotation="horizontal")
   
    fig.tight_layout()
    fig.savefig("analysis/figures/KnownAttributes_spearman.png", dpi=300)

if __name__=="__main__":
    plot()