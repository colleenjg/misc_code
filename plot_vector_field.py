#!/usr/bin/env python

"""
plot_vector_field.py

This script plots a vector field relative to the gradient of a surface

See Richards et al. 2019 Nature Neuroscience, Fig. 3: 
https://www.nature.com/articles/s41593-019-0520-2/figures/3

NOTE: Only implemented for a unit 2D Gaussian surface.

Authors: Colleen Gillon and Blake Richards
"""

import argparse
from pathlib import Path

from mpl_toolkits.mplot3d import Axes3D # import needed to set 3D axes

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np


def get_cmap(nbins=100):
    # full blue: "#2e3092"
    colors = ["#4242ff", "#ffffff", "#ed1c24"] # lighter blue, white, red
    # convert to RGB
    rgb_col = [[] for _ in range(len(colors))]
    for co, col in enumerate(colors):
        ch_vals = mpl.colors.to_rgb(col)
        for ch_val in ch_vals:
            rgb_col[co].append(ch_val)

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "gradient", rgb_col, N=nbins
        )

    return cmap


def plot_quiver_gaussian(arrow_dir="gradient", top_view=False, nbins=100, 
                         output_dir="vector_fields"):

    if arrow_dir == "gradient":
        grad = 1.0
        x_ed = [-1.8, 0.9]
        y_ed = [0, 2]
        interv = 0.35
    elif arrow_dir == "mixed":
        grad = 0.5
        x_ed = [-1.99, 0.25]
        y_ed = [0, 2]
        interv = 0.25
    elif arrow_dir == "curled":
        grad = 0.0
        x_ed = [-1.99, 0.25]
        y_ed = [0, 2]
        interv = 0.25
    else:
        raise ValueError(
            "arrow_dir must be in ['gradient', 'mixed', 'curled'], but "
            f"found {arrow_dir}."
            )

    fig, ax = plt.subplots(subplot_kw = {"projection": "3d"})
    purple = "#612c85"
    cmap = get_cmap(nbins)

    X_surf = np.arange(-2.05, 2.10, 0.05)
    Y_surf = np.arange(-2.15, 2.15, 0.05)
    X_surf, Y_surf = np.meshgrid(X_surf, Y_surf)
    Z_surf = np.exp(-(X_surf**2 + Y_surf**2))

    if top_view:
        interv = 0.4
        arl = 0
        low_alph = 1.0
    else:
        arl = 0.3
        low_alph = 0.4
    X = np.arange(-2.00, 2.25, interv) 
    Y = np.arange(-2.25, 2.25, interv) 
    X, Y = np.meshgrid(X, Y)
    Z = np.exp(-(X**2 + Y**2))

    U = -2.0*X*np.exp(-(X**2 + Y**2))
    V = -2.0*Y*np.exp(-(X**2 + Y**2))

    Urot = np.zeros(np.shape(U)) 
    Vrot = np.zeros(np.shape(V))
    vec  = np.zeros((2,1)) 
    rot  = np.zeros((2,1)) 
    R    = np.array([
        [np.cos(np.pi/2.0), -np.sin(np.pi/2.0)],
        [np.sin(np.pi/2.0), np.cos(np.pi/2.0)]
        ])
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            vec[0] = U[i,j]
            vec[1] = V[i,j]
            rot    = R.dot(vec)
            Urot[i,j] = rot[0] 
            Vrot[i,j] = rot[1] 

    Umix = grad*U + (1.0 - grad)*Urot
    Vmix = grad*V + (1.0 - grad)*Vrot
    W = np.exp(-((Umix + X)**2 + (Vmix + Y)**2)) - Z
    if arrow_dir == "curled": # hackey fix
        W = np.zeros_like(W)
        Z_surf = 1.1*Z_surf

    x_ran = [
        [-2.25, x_ed[0]], [x_ed[0], x_ed[1]], 
        [x_ed[0], x_ed[1]], [x_ed[1], 2.25]
        ]
    y_ran = [
        [-2.25, 2.25], [-2.25, y_ed[0]], [y_ed[0], y_ed[1]], [-2.25, 2.25]
        ]
    alphas = [1.0, 1.0, low_alph, 1.0]
    comps = []

    for i, alph in enumerate(alphas):
        idx = ((X >= x_ran[i][0]) * (X < x_ran[i][1]) * 
            (Y > y_ran[i][0]) * (Y < y_ran[i][1]))
        comp = ax.quiver(X[idx], Y[idx], Z[idx], Umix[idx], Vmix[idx], W[idx], 
                        length=0.3, normalize=True, arrow_length_ratio=arl, 
                        color=purple, alpha=alph)
        comps.append(comp)

    ax.plot_surface(
        X_surf, Y_surf, Z_surf, rstride=1, cstride=1, cmap=cmap, 
        linewidth=0, antialiased=True, alpha=1.0
        )

    ax.grid(False)
    if top_view:
        ax.view_init(90, 0)
        add = "_top"
    else:
        ax.view_init(35, -65)
        add = ""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    Path(output_dir).mkdir(exist_ok=True, parents=True)
    fig.savefig(Path(output_dir, "{}{}.svg".format(arrow_dir, add)))
    fig.savefig(Path(output_dir, "{}{}.png".format(arrow_dir, add)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--arrow_dir", default="all", 
        help="arrow directions: all, gradient, curled or mixed"
        )
    parser.add_argument(
        "--surface", default="gaussian", help="vector field surface"
        )
    parser.add_argument(
        "--top_view", action="store_true", 
        help="whether to show view from the top"
        )
    parser.add_argument(
        "--output_dir", default="vector_fields", 
        help="directory in which to store plots"
        )
    args = parser.parse_args()

    if args.arrow_dir == "all":
        arrow_dirs = ["gradient", "curled", "mixed"]
    else:
        arrow_dirs = [args.arrow_dir]

    for arrow_dir in arrow_dirs:
        if args.surface == "gaussian":
            plot_quiver_gaussian(
                arrow_dir, args.top_view, output_dir=args.output_dir
                )
        else:
            raise NotImplementedError("Only a gaussian surface is implemented.")

