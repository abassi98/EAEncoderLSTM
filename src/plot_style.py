"""Common typography and export defaults for all figures."""

import matplotlib as mpl

LABEL_FONTSIZE = 18
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 20


def configure_plot_style(*, euler_math=False):
    preamble = [
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[osf,sc]{mathpazo}",
    ]
    if euler_math:
        preamble.append(r"\usepackage[euler-digits]{eulervm}")

    mpl.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": "\n".join(preamble),
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": LABEL_FONTSIZE,
        "axes.labelsize": LABEL_FONTSIZE,
        "axes.titlesize": LABEL_FONTSIZE,
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
        "legend.fontsize": LEGEND_FONTSIZE,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.3,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "savefig.format": "pdf",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
