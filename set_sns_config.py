#!/usr/bin/env python
 
"""
set_sns_config.py

Default dictionaries for updating seaborn and matplotlib default 
configurations (AXES_STYLE, PLOTTING_CONTEXT), or for saving matplotlib 
figures (SAVE_CONFIG).
"""

import matplotlib as mpl
import seaborn as sns


# from seaborn: sns.axes_style()
AXES_STYLE = {
    "axes.facecolor"       : "white",
    "axes.edgecolor"       : "black",
    "axes.grid"            : False,
    "axes.axisbelow"       : "line",
    "axes.labelcolor"      : "black",
    "figure.facecolor"     : "white",
    "grid.color"           : "#b0b0b0",
    "grid.linestyle"       : "-",
    "text.color"           : "black",
    "xtick.color"          : "black",
    "ytick.color"          : "black",
    "xtick.direction"      : "out",
    "ytick.direction"      : "out",
    "lines.solid_capstyle" : mpl._enums.CapStyle("projecting"),
    "patch.edgecolor"      : "black",
    "patch.force_edgecolor": False,
    "image.cmap"           : "viridis",
    "font.family"          : ["sans-serif"],
    "font.sans-serif"      : ["DejaVu Sans",
                            "Bitstream Vera Sans",
                            "Computer Modern Sans Serif",
                            "Lucida Grande",
                            "Verdana",
                            "Geneva",
                            "Lucid",
                            "Arial",
                            "Helvetica",
                            "Avant Garde",
                            "sans-serif"],
    "xtick.bottom"         : True,
    "xtick.top"            : False,
    "ytick.left"           : True,
    "ytick.right"          : False,
    "axes.spines.left"     : True,
    "axes.spines.bottom"   : True,
    "axes.spines.right"    : True,
    "axes.spines.top"      : True,
    }


# from seaborn: sns.plotting_context()
PLOTTING_CONTEXT = {
    "font.size"            : 10.0,
    "axes.labelsize"       : "medium",
    "axes.titlesize"       : "large",
    "xtick.labelsize"      : "medium",
    "ytick.labelsize"      : "medium",
    "legend.fontsize"      : "medium",
    "axes.linewidth"       : 0.8,
    "grid.linewidth"       : 0.8,
    "lines.linewidth"      : 1.5,
    "lines.markersize"     : 6.0,
    "patch.linewidth"      : 1.0,
    "xtick.major.width"    : 0.8,
    "ytick.major.width"    : 0.8,
    "xtick.minor.width"    : 0.6,
    "ytick.minor.width"    : 0.6,
    "xtick.major.size"     : 3.5,
    "ytick.major.size"     : 3.5,
    "xtick.minor.size"     : 2.0,
    "ytick.minor.size"     : 2.0,
    "legend.title_fontsize": None,
    }


# Default matplotlib save parameters
SAVE_CONFIG = {
    "dpi"           : None,
    "facecolor"     : "w",
    "edgecolor"     : "w",
    "orientation"   : "portrait",
    "papertype"     : None, 
    "format"        : None,
    "transparent"   : False, 
    "bbox_inches"   : None, 
    "pad_inches"    : 0.1,
    "frameon"       : None,
    "metadata"      : None,
    "backend"       : None,
    }


# Seaborn palette
# Examples: "paired", "cubehelix", "pastel", ...
PALETTE = None 


#############################################
def set_sns_config():

    # Set style, context and palette for Seaborn
    sns.set_style(rc=AXES_STYLE)
    sns.set_context(rc=PLOTTING_CONTEXT)
    sns.set_palette(PALETTE)

    # Update matplotlib as well
    for key, val in AXES_STYLE.items():
        mpl.rcParams[key] = val
    for key, val in PLOTTING_CONTEXT.items():
        mpl.rcParams[key] = val


#############################################
if __name__ == "__main__":
    set_sns_config()
    
