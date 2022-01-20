#!/usr/bin/env python

"""
process_movie_data.py

This script processing movie data, performing a variety of different 
pre-processing steps, for comparison, as well 
Non-Negative Matrix Factorization (NMF).
"""

import argparse
import copy
import logging
from pathlib import Path
import sys
import warnings

import cv2
import h5py
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition, exceptions


logger = logging.getLogger(__name__)

# Set limit for array size warning 
LIM_E6_SIZE = 350


#############################################
def set_logger():
    """
    set_logger()

    Sets the logger to stream to the console, if no streams exist, and sets the 
    level to logging.INFO.
    """
    
    if len(logger.handlers) == 0:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)


#############################################
def seq_blur_downsample(arr, rounds=1):
    """
    seq_blur_downsample(arr)

    Returns array after the specified number of rounds of bluring and 
    downsampling.

    bluring is done with a Gaussian kernel, and downsampling by rejecting even 
    rows and columns.

    Required args:
        - arr (2D array): array to blur and downsample
    
    Optional args:
        - rounds (int): number of times to perform bluring and downsampling, 
                        sequentially
                        default: 1

    Returns:
        - arr (2D arr): blurred and downsampled array.
    """

    arr = copy.deepcopy(arr)
    for _ in range(rounds):
        arr = cv2.pyrDown(arr)
    return arr


#############################################
def scale_clip_8bit(arr, max_intensity=None):
    """
    scale_clip_8bit(arr)

    Returns scaled and thresholded array, converted to unsigned 8 bit int.

    Required args:
        - arr (nd array): array to scale and clip

    Optional args:
        - max_intensity (float): max intensity at which to clip array values
                                 default: None

    Returns:
        - clipped (nd array): scaled and clipped array, converted to unsigned 
                              8 bit int
    """

    if max_intensity is None:
        max_intensity = arr.max() + 1

    if arr.min() < 0:
        raise ValueError("arr should not contain any negative values.")

    rescaled = 256 * arr.astype(float) / max_intensity
    clipped = np.clip(rescaled, 0, 255).astype(np.uint8)

    return clipped


#############################################
def scale_concatenate(arrs, max_intensity=None):
    """
    scale_concatenate(arrs)

    Returns movies, scaled and clipped individually, and concatenated along the 
    width dimension.

    Required args:
        - arrs (list): list of movie arrays of the same shape, each with 
                       dimensions frames x height x width

    Optional args:
        - max_intensity (float): max intensity at which to clip array values
                                 default: None

    Returns:
        - scaled_concat_arr (3D array): scaled, clipped and concatenated arrays, 
                                        structured as frames x height x width
    """

    if not isinstance(arrs, list):
        arrs = [arrs]
    
    scaled_concat_arr = np.concatenate(
        [scale_clip_8bit(arr, max_intensity=max_intensity) for arr in arrs], 
        axis=2
    ) 

    return scaled_concat_arr


#############################################
def get_tiled_images(images, space_prop=0.02, invert=False, fill_val=None):
    """
    get_tiled_images(images)

    Returns images, tiled vertically from left to right, and bottom to top.

    Required args:
        - images (nd array): images to tile, structured as 
                             image x height x width (x additional dims)

    Optional args:
        - space_prop (float): proportion of the longest image side to use as 
                              space between images
                              default: 0.02
        - invert (bool)     : if True, the component images are inverted
                              default: False
        - fill_val (num)    : fill value to use in spaces between components. 
                              If None: if invert is False, the max value from 
                              the images is used, and otherwise half of the max 
                              value is used.
                              default: None

    Returns:
        - tiled_images (nd array): tiled images, structured as 
                                   height x width (x additional dims)
    """

    max_val = images.max()
    if invert:
        if fill_val is None:
            fill_val = max_val / 2
        images = max_val - images
    else:
        if fill_val is None:
            fill_val = max_val

    n, hei, wid = images.shape[:3]
    additional = []
    if len(images.shape) > 3:
        additional = images.shape[3:]

    space = int(max(hei, wid) * space_prop)
    n_rows = int(np.ceil(np.sqrt(n)))
    n_cols = int(np.ceil(n / n_rows))
    
    tiled_images_wid = int(wid * n_cols + space * (n_cols + 1))
    tiled_images_hei = int(hei * n_rows + space * (n_rows + 1))
    tiled_images = np.full(
        (tiled_images_hei, tiled_images_wid, *additional), fill_val
        )
    
    # tile from left to right, bottom to top
    for i in range(n):        
        c = i % n_cols
        r = i // n_cols
        x = c * wid + (c + 1) * space
        y = r * hei + (r + 1) * space
        tiled_images[y : y + hei, x : x + wid] = images[i]

    return tiled_images


#############################################
def get_tiled_comp_movie(W, components, warn_size_e6=LIM_E6_SIZE):
    """
    get_tiled_comp_movie(W, components)

    Generated movies for each component, and returns them tiled vertically into 
    one movie from left to right, and bottom to top.

    Required args:
        - W (2D array)         : component traces, structured as 
                                 frames x components
        - components (3D array): component images, structured as 
                                 components x height x width

    Optional args:
        - warn_size_e6 (int): limit (when multiplied by 1e6) of calculated full 
                              component video size, after which a warning of 
                              potential excess memory use is raised
                              default: LIM_E6_SIZE

    Returns:
        - comps_arr (3D array): tiled component videos, at the same 
                                size as original movie, structured as 
                                frames x height x width
    """

    frames, n_comps = W.shape 
    n_comps, hei, wid = components.shape

    if n_comps * hei * wid * frames > warn_size_e6 * 1e6:
        warnings.warn(
            "The size of the component videos that will be generated will "
            f"exceed the warning size limit ({int(warn_size_e6)} * 10^6). "
            "This may cause excess memory use problems."
            )

    # get tiling size, for early approximate resizing
    dummy_images = np.empty((n_comps, hei, wid))
    tiled_hei, tiled_wid = get_tiled_images(dummy_images).shape

    resize_prop = min(hei / tiled_hei, wid / tiled_wid)
    comps_arr = []
    for comp_traces, comp in zip(W.T, components):
        # calculate individual comp video, and immediately resize, to save 
        # memory
        comp_arr = cv2.resize(
            np.dot(
                comp.reshape(-1, 1), comp_traces.reshape(1, -1)
                ).reshape(hei, wid, -1),
            (int(wid * resize_prop), int(hei * resize_prop)) # pass (wid, hei)
        )
        # hei x wid x frames
        comps_arr.append(comp_arr)
    
    # then tile, and transpose to frames x height x width
    comps_arr = np.transpose(
        get_tiled_images(
            np.asarray(comps_arr), 
            invert=False, 
            fill_val=np.max(comps_arr) / 2
            ), 
        (2, 0, 1)
    )
    _, curr_hei, curr_wid = comps_arr.shape

    # trim/pad edges to match target (input) height and width
    pad_hei_1 = max(0, int(np.floor((hei - curr_hei) / 2)))
    pad_wid_1 = max(0, int(np.floor((wid - curr_wid) / 2)))

    pad_hei_2 = max(0, int(hei - curr_hei - pad_hei_1))
    pad_wid_2 = max(0, int(hei - curr_wid - pad_wid_1))

    comps_arr = np.pad(
        comps_arr, 
        pad_width=((0, 0), (pad_hei_1, pad_hei_2), (pad_wid_1, pad_wid_2)),
        mode="constant",
        constant_values=0
        )[:, : hei, : wid]

    return comps_arr


#############################################
def plot_component_traces(sub_ax, comp_traces):
    """
    plot_component_traces(sub_ax, comp_traces)

    Plots component traces.

    Required args:
        - sub_ax (plt subplot)  : subplot
        - comp_traces (2D array): component traces, 
                                  structured as frames x components
    """

    n_comps = comp_traces.shape[1]

    mins = comp_traces.min(axis=0)
    maxes = comp_traces.max(axis=0)

    ranges_use = (maxes - mins) * 1.3
    comps_scaled = (comp_traces - mins.reshape(1, -1)) / ranges_use
    comps_scaled_shifted = comps_scaled + np.arange(n_comps).reshape(1, -1)

    sub_ax.plot(comps_scaled_shifted, color="k", alpha=0.8)


#############################################
def remove_axis_marks(sub_ax):
    """
    remove_axis_marks(sub_ax)

    Removes all axis marks (ticks, tick labels, spines).

    Required args:
        - sub_ax (plt Axis subplot): subplot    
    """

    sub_ax.tick_params(axis="x", which="both", bottom=False, top=False) 
    sub_ax.tick_params(axis="y", which="both", left=False, right=False) 

    sub_ax.set_xticks([])
    sub_ax.set_yticks([])

    for spine in ["right", "left", "top", "bottom"]:
        sub_ax.spines[spine].set_visible(False)


#############################################
def save_projections(full_data, start_frame=0, num_frames=100, 
                     save_path="projections.png"):
    """
    save_projections(full_data)

    Plots and saves max and mean projections.

    Required args:
        - full_data (3D array): full_video_data, 
                                structured as frames x height x width

    Optional args:
        - start_frame (int)      : start frame
                                   default: 0
        - num_frames (int)       : number of frames
                                   default: 100
        - save_path (Path or str): path under which to save projections
                                   default: "projections.png"
    """
    
    arr = full_data[start_frame : start_frame + num_frames]

    fig, ax = plt.subplots(1, 2)

    ax[0].set_title("Max projection")
    ax[0].imshow(arr.max(axis=0), cmap="gray")

    ax[1].set_title("Mean projection")
    ax[1].imshow(arr.mean(axis=0), cmap="gray")

    for sub_ax in ax.reshape(-1):
        remove_axis_marks(sub_ax)

    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)


#############################################
def save_movie(arr, fps=31, save_path="video.avi"):
    """
    save_movie(arr)

    Saves arrays as a movie.

    Required args:
        - arr (3D array): video data, structured as frames x height x width

    Optional args:
        - fps (num)              : frames per second for video encoding
                                   default: 31
        - save_path (Path or str): path under which to save video
                                   default: "video.avi"
    """
    
    _, height, width = arr.shape

    arr = scale_clip_8bit(arr)

    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    writer = cv2.VideoWriter(
        str(Path(save_path)), # filename
        cv2.VideoWriter_fourcc(*"FFV1"), # fourcc: lossless codec
        fps, # fps
        (width, height), # frameSize
        isColor=False,
    )

    for frame in arr:
        writer.write(frame)

    writer.release()


#############################################
def raw_movie(full_data, start_frame=0, num_frames=100, fps=31, gen_path=None):
    """
    raw_movie(full_data)

    Returns, and optionally saved, raw movie.

    Required args:
        - full_data (3D array): full_video_data, 
                                structured as frames x height x width

    Optional args:
        - start_frame (int)     : start frame
                                  default: 0
        - num_frames (int)      : number of frames
                                  default: 100
        - fps (num)             : frames per second for video encoding
                                  default: 31
        - gen_path (Path or str): general path to save movie under, without 
                                  extension. If None, movie is not saved.
                                  default: None

    Returns:
        - raw_arr (3D array): raw video, 
                              structured as frames x height x width
    """
    
    raw_arr = full_data[start_frame : start_frame + num_frames]

    if gen_path is not None:
        logger.info(f"Saving raw movie ({num_frames} frames)...")
        save_path = f"{gen_path}_raw.avi"
        save_movie(raw_arr, fps=fps, save_path=save_path)
    
    return raw_arr


#############################################
def smooth_movie(full_data, start_frame=0, num_frames=100, fps=31, 
                 window=3, gen_path=None):
    """
    smooth_movie(full_data)

    Returns, and optionally saved, smoothed movie.

    Required args:
        - full_data (3D array): full_video_data, 
                                structured as frames x height x width

    Optional args:
        - start_frame (int)     : start frame
                                  default: 0
        - num_frames (int)      : number of frames
                                  default: 100
        - fps (num)             : frames per second for video encoding
                                  default: 31
        - window (int)          : smoothing window size
                                  default: 3
        - gen_path (Path or str): general path to save movie under, without 
                                  extension. If None, movie is not saved.
                                  default: None
    
    Returns:
        - smoothed_arr (3D array): smoothed video, 
                                   structured as frames x height x width
    """
    
    min_frame = max(0, start_frame - window)
    max_frame = start_frame + num_frames + window

    smoothed_arr = scipy.ndimage.uniform_filter(
        full_data[min_frame : max_frame],
        size=(window, 1, 1),
        mode="reflect",
        )[start_frame - min_frame : start_frame - min_frame + num_frames]

    if gen_path is not None:
        logger.info(
            f"Saving movie with cross-frame averaging ({num_frames} frames)..."
            )

        save_path = f"{gen_path}_smoothed_{window}.avi"
        save_movie(smoothed_arr, fps=fps / window, save_path=save_path)
    
    return smoothed_arr


#############################################
def gauss_blur_movie(full_data, start_frame=0, num_frames=100, fps=31, 
                     kernel_size=5, kernel_std=1, gen_path=None):
    """
    gauss_blur_movie(full_data)

    Returns, and optionally saved, Gaussian blurred movie.

    Required args:
        - full_data (3D array): full video data, 
                                structured as frames x height x width

    Optional args:
        - start_frame (int)     : start frame
                                  default: 0
        - num_frames (int)      : number of frames
                                  default: 100
        - fps (num)             : frames per second for video encoding
                                  default: 31
        - kernel_size (int)     : Gaussian smoothing kernel size (odd number)
                                  default: 3
        - kernel_std (int)      : Gaussian kernel standard deviation
                                  default: 1
        - gen_path (Path or str): general path to save movie under, without 
                                  extension. If None, movie is not saved.
                                  default: None
    
    Returns:
        - gauss_blurred_arr (3D array): Gaussian blurred video, 
                                        structured as frames x height x width
    """
    
    # place frames last (treated as independent channels)
    gauss_blurred_arr = np.transpose(
        cv2.GaussianBlur(
            np.transpose(
                full_data[start_frame : start_frame + num_frames], (1, 2, 0)
                ),
            ksize=(kernel_size, kernel_size),
            sigmaX=kernel_std,
            sigmaY=kernel_std,
            ), (2, 0, 1)
        )

    if gen_path is not None:
        logger.info(
            f"Saving Gaussian blurred movie ({num_frames} frames)..."
            )

        save_path = f"{gen_path}_gaussian_blurred_{kernel_size}.avi"
        save_movie(gauss_blurred_arr, fps=fps, save_path=save_path)

    return gauss_blurred_arr


#############################################
def downsample_movie(full_data, start_frame=0, num_frames=100, fps=31, 
                     rounds=1, gen_path=None):
    """
    downsample_movie(full_data)

    Returns, and optionally saved, downsampled movie.

    Required args:
        - full_data (3D array): full_video_data, 
                                structured as frames x height x width

    Optional args:
        - start_frame (int)     : start frame
                                  default: 0
        - num_frames (int)      : number of frames
                                  default: 100
        - fps (num)             : frames per second for video encoding
                                  default: 31
        - rounds (int)          : rounds of consecutive downsampling to perform
                                  default: 3
        - gen_path (Path or str): general path to save movie under, without 
                                  extension. If None, movie is not saved.
                                  default: None
    
    Returns:
        - downsampled_arr (3D array): downsampled video, 
                                      structured as frames x height x width
    """

    downsampled_arr = []
    for i in range(start_frame, start_frame + num_frames):
        downsampled_arr.append(
            seq_blur_downsample(full_data[i], rounds=rounds)
        )
    downsampled_arr = np.asarray(downsampled_arr)

    if gen_path is not None:
        logger.info(
            f"Saving downsampled movie ({num_frames} frames)..."
            )

        save_path = f"{gen_path}_downsampled.avi"
        save_movie(downsampled_arr, fps=fps, save_path=save_path)
    
    return downsampled_arr


#############################################
def save_nmf_movie(full_data, start_frame=0, num_frames=100, fps=31, 
                   n_comps=6, randst=None, gen_path=None):
    """
    save_nmf_movie(full_data)

    Returns, and optionally saved, NMF components, and reconstructed movie.

    Required args:
        - full_data (3D array): full_video_data, 
                                structured as frames x height x width

    Optional args:
        - start_frame (int)     : start frame
                                  default: 0
        - num_frames (int)      : number of frames
                                  default: 100
        - fps (num)             : frames per second for video encoding
                                  default: 31
        - n_comps (int)         : number of components to decompose movie into 
                                  with NMF
                                  default: 6
        - randst (int)          : random state or seed
                                  default: None
        - gen_path (Path or str): general path to save movie under, without 
                                  extension. If None, movie is not saved.
                                  default: None
    
    Returns:
        - W (2D array)         : component intensities, 
                                 structured as frames x components
        - components (3D array): component images, 
                                 structured as components x height x width
        - raw_recon_split_arr_scaled (3D array): 
                                 raw and NMF recontructed videos, concatenated 
                                 along the width dimension, structured as 
                                 frames x height x width
    """

    logger.info(
        f"Generating and saving NMF movie ({num_frames} frames)..."
        )

    raw_arr = full_data[start_frame : start_frame + num_frames]
    hei, wid = raw_arr.shape[1:]

    # get sparse components
    estimator = decomposition.NMF(
        n_comps, init=None, alpha=1500, l1_ratio=0.95, random_state=randst
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=exceptions.ConvergenceWarning)
        W = estimator.fit_transform(raw_arr.reshape(len(raw_arr), -1))

    # plot component images
    components = estimator.components_.reshape(-1, hei, wid)

    if gen_path is not None:
        fig, ax = plt.subplots(1, 3, figsize=(13, 5))
        fig.suptitle("NMF components")

        plot_component_traces(ax[0], W)
        ax[0].spines["right"].set_visible(False)
        ax[0].spines["top"].set_visible(False)
        ax[0].set_xlabel("Frames")
        ax[0].set_ylabel("Components (scaled)")

        for i, invert in enumerate([True, False]):
            invert_str = " (inverted)" if invert else ""
            ax[i + 1].imshow(
                get_tiled_images(components, invert=invert), 
                cmap="gray"
                )
            remove_axis_marks(ax[i + 1])
            ax[i + 1].set_xlabel(f"Components{invert_str}")

        save_path = f"{gen_path}_nmf.png"
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(save_path, dpi=300)

    # reconstruct movie
    recon_arr = estimator.inverse_transform(W).reshape(raw_arr.shape)
    tiled_comp_arr = get_tiled_comp_movie(W, components) # may be memory intensive
    raw_recon_split_arr_scaled = scale_concatenate(
        [raw_arr, recon_arr, tiled_comp_arr]
        )
    if gen_path is not None:
        save_path = f"{gen_path}_raw_nmf-recon_split.avi"
        save_movie(raw_recon_split_arr_scaled, fps=fps, save_path=save_path)

    return W, components, raw_recon_split_arr_scaled


#############################################
def main(dataset, num_frames=200, start_frame=0, fps=31, seed=None, 
         output_dir="processed_videos", output_name="unnamed"):
    """
    main(args)

    Produces pre-processed videos, projections and NMF decomposed videos from 
    input video data.

    Required args:
        - dataset (Path): h5 dataset to read from, with data under the "data" 
                          key of shape: frames x height x width

    Optional args:
        - num_frames (int) : number of frames to extract from the dataset
                             (-1 for all)
                             default: 200
        - start_frame (int): frame number from which to start extracting frames
                             in the dataset
                             default: 0
        - fps (num)        : number of frames per second for rendering movies
                             default: 31
        - seed (int)       : random state seed
                             default: None
        - output_dir (Path): directory to save output files to
                             default: "processed_videos"
        - output_name (str): base name for the output files
                             default: "unnamed"
    """

    set_logger()

    dataset = Path(dataset)
    if not dataset.is_file():
        raise OSError(f"{dataset} not found.")
    elif dataset.suffix != ".h5":
        raise ValueError("Expected data_path to be an h5 file.")

    with h5py.File(dataset, "r") as f:
        if "data" not in f.keys():
            raise RuntimeError(
                "Expected data_path to contain a 'data' dataset."
                )
        dset_handle = f["data"]
        total_frames = dset_handle.shape[0]

        # a few checks
        if start_frame >= total_frames:
            raise ValueError(
                f"Start frame ({start_frame}) is too high for the total "
                f"number of frames ({total_frames})"
                )
        if num_frames == -1:
            num_frames = total_frames - start_frame
        elif start_frame + num_frames > total_frames:
            num_frames = total_frames - start_frame

        kwargs = {
            "full_data"  : dset_handle,
            "start_frame": start_frame,
            "num_frames" : num_frames,
        }

        # prepare the general path save name
        stop_frame = start_frame + num_frames
        gen_path = Path(
            output_dir, f"{output_name}_fr{start_frame}-{stop_frame}"
            )

        # create projections
        logger.info("Generating and saving movie projections...")
        save_projections(save_path=f"{gen_path}_projections.png", **kwargs)

        # concatenate movies with different pre-processing
        logger.info(
            "Generating and saving pre-processed movies "
            f"({num_frames} frames)..."
            )
        raw_arr = raw_movie(**kwargs)
        smoothed_arr = smooth_movie(**kwargs)
        gauss_blurred_arr = gauss_blur_movie(**kwargs)


        all_arr_scaled = scale_concatenate(
            [raw_arr, smoothed_arr, gauss_blurred_arr]
            )

        save_movie(
            all_arr_scaled, 
            fps=fps, 
            save_path=f"{gen_path}_raw_smoothed_gauss-blurred.avi"
            )

        # additional movies
        downsample_movie(gen_path=gen_path, **kwargs)
        save_nmf_movie(gen_path=gen_path, randst=seed, **kwargs)


#############################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset", default="concat_31Hz_0.h5", type=Path, 
        help=("h5 dataset to read from, with data stored under the 'data' key "
            "of shape frames x height x width")
        )
    parser.add_argument(
        "--num_frames", default=200, type=int, 
        help="number of frames to use (-1 for all frames after the start frame)"
        )
    parser.add_argument(
        "--start_frame", default=0, type=int, help="frame to start from"
        )
    parser.add_argument("--fps", default=31, type=int, help="frames per second")
    parser.add_argument("--seed", default=None, help="random state seed")
    parser.add_argument("--output_dir", type=Path, default="processed_videos")
    parser.add_argument("--output_name", type=Path, default="unnamed")

    args = parser.parse_args()
    if args.seed is not None:
        args.seed = int(args.seed)

    main(**args.__dict__)

    plt.close("all")

    