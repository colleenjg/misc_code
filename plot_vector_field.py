#!/usr/bin/env python

"""
plot_vector_field.py

This script plots a vector field relative to the gradient of a surface

See Richards et al. 2019 Nature Neuroscience, Fig. 3: 
https://www.nature.com/articles/s41593-019-0520-2/figures/3

NOTE: Only implemented for a unit 2D Gaussian surface.

Authors: Colleen Gillon and Blake Richards
"""

from mpl_toolkits.mplot3d import Axes3D # to set 3D axes

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

PURPLE = "#612c85"
CMAP_COLORS = ["#4242ff", "#ffffff", "#ed1c24"] # blue, white, red
GAUSS_VIEW = (35, -65)
TOP_VIEW = (90, 0)


#############################################
def stretch_x_axis(ax, factor=1.4):
    """
    stretch_x_axis(ax)

    Stretches the x axis of a 3D projected subplot.

    Required args:
        - ax (plt subplot): subplot

    Optional args:
        - factor (float): stretch factor for the x axis
                          default: 1.4
    """
    
    stretch_matrix = np.diag([1, factor, 1, 1])
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), stretch_matrix)
    return


#############################################
def get_cmap(nbins=100):
    """
    get_cmap()

    Returns a colormap.

    Optional args:
        - nbins (int): number of bins for the colormap
                       default: 100
        
    Returns:
        - cmap (plt colormap): colormap
    """
    rgb_colors = [mpl.colors.to_rgb(col) for col in CMAP_COLORS]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "gradient", rgb_colors, N=nbins
        )
    return cmap


#############################################
def add_title_savename(ax, arrow_dir="gradient", top_view=False, 
                       surface_type="gaussian"):
    """
    add_title_savename(ax)

    Add titles to a subplot, and returns a name under which the plot can be 
    saved.

    Required args:
        - ax (plt subplot): subplot

    Optional args:
        - arrow_dir (str)   : arrow directions with respect to the gradient, 
                              i.e. "gradient", "curled", "mixed"
                              default: "gradient"
        - top_view (bool)   : if True, plot is shown from the top
                              default: False
        - surface_type (str): type of surface plotted
                              default: "gaussian"

    Returns:
        - savename (str): name under which to save plot
    """

    if top_view:
        view_str_pr = " (from top)"
        view_str = "_top"
        title_y = 1.0
    else:
        view_str_pr, view_str = "", ""
        title_y = 0.9

    title = f"{arrow_dir.capitalize()} {surface_type} vector field{view_str_pr}"
    ax.set_title(title, y=title_y)
    
    savename = f"{arrow_dir}_{surface_type}{view_str}"

    return savename


#############################################
def get_arrow_variables(arrow_dir="gradient", top_view=False):
    """
    get_arrow_variables()

    Add titles to a subplot, and returns a name under which the plot can be 
    saved.

    Optional args:
        - arrow_dir (str): arrow directions with respect to the gradient, 
                           i.e. "gradient", "curled", "mixed"
                           default: "gradient"
        - top_view (bool): if True, plot is shown from the top
                           default: False

    Returns:
        - arrow_variables (dict): dictionary with arrow variables
            ["arrow_length"]      : arrow length
            ["arrow_length_ratio"]: arrow head to quiver ratio
            ["grad_prop"]         : arrow direction as a proportion of the 
                                    gradient
            ["interval"]          : plot interval

    """

    # get arrow start coordinates
    if arrow_dir == "gradient":
        grad_prop = 1.0
        interval = 0.35
    elif arrow_dir == "mixed":
        grad_prop = 0.5
        interval = 0.25
    elif arrow_dir == "curled":
        grad_prop = 0
        interval = 0.25
    else:
        raise ValueError(
            "arrow_dir must be in ['gradient', 'mixed', 'curled'], "
            f"but found {arrow_dir}."
            )

    if top_view:
        interval = 0.4
        arrow_length_ratio = 0
        arrow_length = 0.2
    else:
        arrow_length_ratio = 0.3
        arrow_length = 0.3
    
    arrow_variables = {
        "arrow_length"      : arrow_length,
        "arrow_length_ratio": arrow_length_ratio,
        "grad_prop"         : grad_prop,
        "interval"          : interval,
    }

    return arrow_variables


#############################################
def format_axis(ax, top_view=False):
    """
    format_axis(ax)

    Formats subplot by removing the grid lines, axis spines, and axis panes, 
    adjusting the view angle, and adding a small basis vector lines, if 
    top_view is False. 

    Required args:
        - ax (plt subplot): subplot

    Optional args:
        - top_view (bool): if True, plot is shown from the top
                           default: False
    """

    # adjust plot formatting
    ax.grid(False)

    # Transparent spines and panes
    transparent = (1.0, 1.0, 1.0, 0.0)
    for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
        axis.line.set_color(transparent)
        axis.set_pane_color(transparent)

    if top_view:
        ax.view_init(*TOP_VIEW)
    else:
        ax.view_init(*GAUSS_VIEW)
        
        # add a line for each axis
        prop = 0.3
        lims = [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]
        spine_pos, spine_edges = [], []
        for lim, idx in zip(lims, [0, 1, 0]):
            spine_pos.append([lim[idx], lim[idx]])
            if idx == 0:
                spine_edges.append([lim[0], lim[0] + prop * (lim[1] - lim[0])])
            else:
                spine_edges.append([lim[1] + prop * (lim[0] - lim[1]), lim[1]])

        ax.plot(spine_edges[0], spine_pos[1], spine_pos[2], color="k") # x line
        ax.plot(spine_pos[0], spine_edges[1], spine_pos[2], color="k") # y line
        ax.plot(spine_pos[0], spine_pos[1], spine_edges[2], color="k") # z line

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


#############################################
def calculate_gaussian_z(x, y):
    """
    calculate_gaussian_z(x, y)

    Returns the z values for a unit 2D Gaussian.

    Required args:
        - x (nd array): x axis values
        - y (nd array): y axis values, same shape as x

    Returns:
        - z (nd array): z axis Gaussian values
    """

    z = np.exp(-(x ** 2 + y ** 2))
    return z


#############################################
def get_XYZ(edges, interval=0.05, surface_type="gaussian"):
    """
    get_XYZ(edges)

    Returns XYZ 2D mesh values for the surface type.

    Required args:
        - edges (list or tuple): edge values (start, end) for x/y

    Optional args:
        - interval (float)  : interval between x values, and between y values
                              default: 0.05
        - surface_type (str): type of surface plotted
                              default: "gaussian"

    Returns:
        - X (2D array): mesh values in X
        - Y (2D array): mesh values in Y
        - Z (2D array): mesh values in Z
    """

    X, Y = np.meshgrid(
        np.arange(*edges, interval), 
        np.arange(*edges, interval), 
        )

    if surface_type == "gaussian":
        Z = calculate_gaussian_z(X, Y)
    else:
        raise NotImplementedError(
            "Only 'gaussian' surface_type is implemented."
            )

    return X, Y, Z


#############################################
def plot_surface(ax, edges, interval=0.05, arrow_dir="gradient", 
                 top_view=False, surface_type="gaussian"):
    """
    plot_surface(ax, edges)

    Plots the specified surface on the subplot.

    Required args:
        - ax (plt subplot)     : subplot
        - edges (list or tuple): edge values (start, end) for x/y

    Optional args:
        - interval (float)  : interval between x values, and between y values
                              default: 0.05
        - arrow_dir (str)   : arrow directions with respect to the gradient, 
                              i.e. "gradient", "curled", "mixed"
                              default: "gradient"
        - top_view (bool)   : if True, plot is shown from the top
                              default: False
        - surface_type (str): type of surface plotted
                              default: "gaussian"
    """

    if top_view:
        surf_alpha = 0.7
    else:
        surf_alpha = 0.5

    X_surf, Y_surf, Z_surf = get_XYZ(edges, interval, surface_type=surface_type)

    if arrow_dir == "curled": # hackey adjustment, works for Gaussian
        Z_surf *= 1.1

    surf_data = [X_surf, Y_surf, Z_surf]
    
    ax.plot_surface(
        *surf_data,
        rstride=1, 
        cstride=1, 
        cmap=get_cmap(), 
        linewidth=0, 
        antialiased=True, 
        alpha=surf_alpha,
        )


#############################################
def get_UVW(X, Y, Z, grad_prop=0.5, surface_type="gaussian"):
    """
    get_UVW(X, Y, Z)

    Returns arrow directions for each X, Y, Z combination, relative to the 
    gradient of the surface.

    Required args:
        - X (2D array): mesh values in X
        - Y (2D array): mesh values in Y
        - Z (2D array): mesh values in Z

    Optional args:
        - grad_prop (float) : arrow direction as a proportion of the gradient
                              default: 0.5
        - surface_type (str): type of surface plotted
                              default: "gaussian"

    Returns:
        - U (2D array): arrow direction in the X axis
        - W (2D array): arrow direction in the Y axis 
        - Z (2D array): arrow direction in the Z axis
    """

    if surface_type != "gaussian":
        raise NotImplementedError(
            "Only 'gaussian' surface_type is implemented."
            )

    # get arrow directions with UVW mapping
    U_grad = -2.0 * X * Z
    V_grad = -2.0 * Y * Z

    # rotate each U and V
    half_pi = np.pi / 2.0
    R = np.array([
        [np.cos(half_pi), -np.sin(half_pi)],
        [np.sin(half_pi), np.cos(half_pi)]
        ])

    U_orth = np.zeros_like(U_grad)
    V_orth = np.zeros_like(V_grad)
    for i in range(U_grad.shape[0]):
        for j in range(U_grad.shape[1]):
            vector = np.asarray([U_grad[i, j], V_grad[i, j]])
            rotation  = R.dot(vector.reshape(2, 1))
            U_orth[i, j] = rotation[0] 
            V_orth[i, j] = rotation[1] 

    # get arrow directions
    U = grad_prop * U_grad + (1.0 - grad_prop) * U_orth
    V = grad_prop * V_grad + (1.0 - grad_prop) * V_orth
    W = calculate_gaussian_z(X + U, Y + V) - Z

    if grad_prop == 0: # hackey
        W = np.zeros_like(W)

    return U, V, W


#############################################
def get_arrow_alphas_indices(X, Y, edges, arrow_dir="gradient", 
                             top_view=False, surface_type="gaussian"):
    """
    get_arrow_alphas_indices(X, Y, edges)

    Returns a list of alpha values for complementary parts of the surface, and 
    the mesh indices for which each alpha value should be used.

    This allows arrows situated behind the surface parts, based on the 
    view point, to be plotted with a higher transparency. It is expected that 
    the global view variables are used (e.g., GAUSS_VIEW, TOP_VIEW).

    Required args:
        - X (2D array)         : mesh values in X
        - Y (2D array)         : mesh values in Y
        - edges (list or tuple): edge values (start, end) for x/y

    Optional args:
        - arrow_dir (str)   : arrow directions with respect to the gradient, 
                              i.e. "gradient", "curled", "mixed"
                              default: "gradient"
        - top_view (bool)   : if True, plot is shown from the top
                              default: False
        - surface_type (str): type of surface plotted
                              default: "gaussian"

    Returns:
        - alphas (list): alpha values to use
        - idxs (list)  : list of 2D boolean arrays for each alpha value 
    """

    if surface_type != "gaussian":
        raise NotImplementedError(
            "Only 'gaussian' surface_type is implemented."
            )

    # split quivers into high and low alphas
    if top_view:
        alphas = [1.0]
        idxs = [np.ones_like(X).astype(bool)]
    else:
        alphas = [1.0, 0.4]

        # these values are set for the Gaussian surface, plotting from 
        # the GAUSS_VIEW perspective
        if arrow_dir == "gradient":
            x_props = [0.1, 0.38]
            y_props = [0.5, 0]
        elif arrow_dir in ["mixed", "curled"]:
            x_props = [0.08, 0.45]
            y_props = [1 / 2.1, 0]
        else:
            raise ValueError(
                "arrow_dir must be in ['gradient', 'mixed', 'curled'], "
                f"but found {arrow_dir}."
                )

        if len(edges) != 2:
            raise ValueError("edges must have length 2.")
        st, end = edges
        edge_ran = end - st

        x_low_alpha_edges = [
            st + edge_ran * x_props[0], 
            end - edge_ran * x_props[1]
            ]

        y_low_alpha_edges = [
            st + edge_ran * y_props[0], 
            end - edge_ran * y_props[1]
            ]

        low_alpha_idx = (
            (X >= x_low_alpha_edges[0]) * 
            (X < x_low_alpha_edges[1]) * 
            (Y > y_low_alpha_edges[0]) * 
            (Y < y_low_alpha_edges[1])
            )        
        idxs = [~low_alpha_idx, low_alpha_idx]

    return alphas, idxs


#############################################
def plot_arrows(ax, edges, arrow_dir="gradient", top_view=False, 
                surface_type="gaussian"):
    """
    plot_arrows(ax, edges)

    Plots arrows relative to the surface gradient.

    Required args:
        - ax (plt subplot)     : subplot
        - edges (list or tuple): edge values (start, end) for x/y

    Optional args:
        - arrow_dir (str)   : arrow directions with respect to the gradient, 
                              i.e. "gradient", "curled", "mixed"
                              default: "gradient"
        - top_view (bool)   : if True, plot is shown from the top
                              default: False
        - surface_type (str): type of surface plotted
                              default: "gaussian"
    """

    arrow_vars = get_arrow_variables(arrow_dir, top_view)

    X, Y, Z = get_XYZ(edges, arrow_vars["interval"], surface_type=surface_type)
    U, V, W = get_UVW(
        X, Y, Z, grad_prop=arrow_vars["grad_prop"], surface_type=surface_type
        )

    # plot quivers with high and low alphas
    alphas, idxs = get_arrow_alphas_indices(
        X, Y, edges, arrow_dir=arrow_dir, top_view=top_view, 
        surface_type=surface_type)


    for alpha, idx in zip(alphas, idxs):
        start_pts = [val[idx] for val in [X, Y, Z]]
        directions = [val[idx] for val in [U, V, W]]

        ax.quiver(
            *start_pts,
            *directions,
            length=arrow_vars["arrow_length"], 
            normalize=True, 
            arrow_length_ratio=arrow_vars["arrow_length_ratio"], 
            color=PURPLE,
            alpha=alpha
            )


#############################################
def plot_vector_field(arrow_dir="gradient", top_view=False, 
                      surface_type="gaussian", output_dir="vector_fields"):
    """
    plot_vector_field()

    Plots a vector field relative to the gradient of a surface.

    Optional args:
        - arrow_dir (str)   : arrow directions with respect to the gradient, 
                              i.e. "gradient", "curled", "mixed"
                              default: "gradient"
        - top_view (bool)   : if True, plot is shown from the top
                              default: False
        - surface_type (str): type of surface plotted
                              default: "gaussian"
        - output_dir (Path) : directory to save output files to
                              default: "vector_fields"
    """

    fig, ax = plt.subplots(
        subplot_kw = {"projection": "3d"}
        )
    
    if top_view:
        stretch_x_axis(ax)

    if surface_type == "gaussian":
        edges = [-2.25, 2.25]
    else:
        raise NotImplementedError(
            "Only 'gaussian' surface_type is implemented."
            )
    
    kwargs = {
        "arrow_dir"   : arrow_dir,
        "top_view"    : top_view,
        "surface_type": surface_type,
    }

    plot_surface(ax, edges, **kwargs)
    plot_arrows(ax, edges, **kwargs)

    format_axis(ax, top_view=top_view)

    savename = add_title_savename(ax, **kwargs)
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    for ext in ["svg", "png"]:
        fig.savefig(Path(output_dir, f"{savename}.{ext}"), bbox_inches="tight")


#############################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--arrow_dir", default="all", 
        help=("arrow direction: 'gradient', 'curled' or 'mixed'. "
            "'all' runs through all 3 sequentially")
        )
    parser.add_argument("--top_view", action="store_true")
    parser.add_argument(
        "--surface_type", default="gaussian", 
        help="type of surface to plot (only 'gaussian' is implemented)"
        )
    parser.add_argument("--output_dir", default="vector_fields", type=Path)

    args = parser.parse_args()

    if args.arrow_dir == "all":
        arrow_dirs = ["gradient", "curled", "mixed"]
    else:
        arrow_dirs = [args.arrow_dirs]

    for arrow_dir in arrow_dirs:
        plot_vector_field(
            arrow_dir, args.top_view, args.surface_type, 
            output_dir=args.output_dir
            )

    plt.close("all")

    