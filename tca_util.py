#!/usr/bin/env python

"""
tca_util.py

This script contains functions to run Tensor Component Analysis 
(i.e., Canonical Polyadic Decomposition (CPD) or 
parallel factor analysis (PARAFAC)) to extract rank-one components from 
multi-dimensional source data.

NOTE: The types of tensor decomposition algorithms used in this script are not 
deterministic, and components are not ordered, as in PCA. Thus, different runs 
can produce different component orders, as well as different components 
altogether.

The components are more compactly expressed as lists of factor matrices for 
each dimension, sharing the same rank. 

Key resources include:
https://tensortools-docs.readthedocs.io/en/latest/
http://tensorly.org/stable/index.html
https://medium.com/@mohammadbashiri93/tensor-decomposition-in-python-f1aa2f9adbf4
https://iopscience.iop.org/article/10.1088/2632-2153/ab8240/pdf
"""

import argparse
import logging
from pathlib import Path
import sys
import warnings

import h5py
import numpy as np
import matplotlib.pyplot as plt

from tensorly import decomposition as tensorly_decomposition
try:
    import tensortools
except ModuleNotFoundError as err:
    if "tensortools" in str(err):
        raise ModuleNotFoundError(
            f"{err}. Can be installed by running `pip install "
            "git+https://github.com/ahwillia/tensortools`"
        )
    else:
        raise err

logger = logging.getLogger(__name__)


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
def rank_one_tensor(*vectors):
    """
    rank_one_tensor(*vectors)

    Returns a rank 1 tensor, given multiple vectors.

    Required args:
        - *vectors (1D arrays): vector to use in generating a rank 1 tensor

    Returns:
        - R1 (nd array): rank one array whose dimensions are sequentially the 
                         lengths of the vectors, 
                         e.g. len(vector1) x len(vector2) x ...
    """

    R1 = vectors[0]
    for vector in vectors[1:]:
        R1 = np.tensordot(R1, vector, axes=-1)

    return R1


#############################################
def get_check_rank(factors, rank=None):
    """
    get_check_rank(factors)

    Returns the rank of a set of components expressed as a list of factor 
    matrices, or checks that the specified rank is appropriate. 

    Raises an error if the factor matrices don't have the same rank.
    
    Required args:
        - factors (list): components, expressed as a list of factor matrices, 
                          structured as dims x [dim_length x rank]
        
    Optional args:
        - rank (int): target rank value (must be smaller or equal to the rank 
                      of each factor, i.e. the number of components
                      default: None

    Returns:
        - rank (int): rank value passed or, if none was passed, the rank of the 
                      factor matrices.
    """

    ranks = list(set([factor_mat.shape[1] for factor_mat in factors]))

    if len(ranks) != 1:
        raise ValueError("All factor matrices should have the same rank.")
    
    if rank is None:
        rank = ranks[0]

    elif rank > ranks[0]:
        raise ValueError(
            f"The rank requested ({rank}) is higher than the number of "
            f"components ({ranks[0]})."
            )
    
    return rank


#############################################
def scale_data(x, axis=0, scale_type="min_max"):
    """
    scale_data(x)

    Returns input data array, scaled as (x - sub) / div.

    Required args:
        - x (nd array): data to scale 

    Optional args:
        - axis (int or tuple): axis or axes along which to calculate scaling 
                               factors
                               default: 0
        - scale_type (str)   : type of scaling to use, e.g. 
                               "min_max": (x - x_min) / (x_max - x_min)
                               default: "min_max"

    Returns:
        - scaled_x (nd array): scaled x array
        - sub (nd array)     : scaling subtraction factor, with same number of 
                               dimensions as x
        - div (nd array)     : scaling division factor, with the same number of 
                               dimensions as x
    """

    if scale_type == "min_max":
        sub = x.min(axis=axis, keepdims=True)
        div = x.max(axis=axis, keepdims=True) - sub
    else:
        raise NotImplementedError("Only 'min_max' scale_type is implemented.")

    scaled_x = (x - sub) / div

    return scaled_x, sub, div


#############################################
def unscale_data(x, sub=0, div=1):
    """
    unscale_data(x)

    Returns input data array, unscaled as x * div + sub.

    Required args:
        - x (nd array): data to unscale 

    Optional args:
        - sub (num or nd array): scaling subtraction factor, with same number 
                                 of dimensions as x, if in array form
        - div (num or nd array): scaling division factor, with same number 
                                 of dimensions as x, if in array form

    Returns:
        - unscaled_x (nd array): unscaled x array

    """

    unscaled_x = x * div + sub

    return unscaled_x



#############################################
def aggregate_correlations(corr_mats, axis=0, aggreg_stat="mean"):
    """
    aggregate_correlations(corr_mats)

    Returns correlations aggregated across the specified axis, using the 
    specified statistic.

    Required args:
        - corr_mats (nd array): array of correlation data, 
                                e.g., organized as dims x factors1 x factors2

    Optional args:
        - axis (int or tuple): axis or axes along which to take statistic
                               default: 0
        - aggreg_stat (str)  : how to aggregate cross-correlations across 
                               component factors ("mean" or "product")  
                               default: "mean"
    
    Returns:
        - aggreg_corr_mat (nd array): array of aggregated correlation data, 
                                      e.g., organized as factors1 x factors2
    """
    
    if aggreg_stat == "mean":
        aggreg_corr_mat = np.mean(corr_mats, axis=axis)
    elif aggreg_stat == "product":
        aggreg_corr_mat = np.product(corr_mats, axis=axis)
    else:
        raise ValueError("aggreg_stat must be 'mean' or 'product'.")

    return aggreg_corr_mat


#############################################
def simple_align_comps(factors1, factors2, allow_inverting=True, 
                       aggreg_stat="mean"):
    """
    simple_align_comps(factors1, factors2)

    Returns two sets of components, expressed as lists of factor matrices, 
    aligned to each other from the strongest match to the weakest. 

    Components are aligned between sets by calculating the correlations between 
    each of their factors, and averaging or multiplying these correlations 
    across factors for each component. 

    Each component from one set is then paired with the best matching remaining  
    component of the other set, in order of strongest to weakest match.

    If allow_inverting is True, and a match reflects negative correlations, 
    the negatively correlated factors can be inverted, up to an even number per 
    component to ensure component sign is preserved. 

    Inversions, if applicable, are always applied to factors2 factors during 
    alignment.

    Required args:
        - factors1 (list): first set of components, expressed as a list of 
                           factor matrices, structured as 
                           dims x [dim_length x rank]
        - factors2 (list): second set of components, expressed as a list of 
                           factor matrices, structured as 
                           dims x [dim_length x rank] 
                           (same length and shapes as factors2)

    Optional args:
        - allow_inverting (bool): if True, inverting factors can be used for 
                                  alignment, but only an even number per 
                                  component, to preserve overall component sign
                                  default: True
        - aggreg_stat (str)     : how to aggregate cross-correlations across 
                                  component factors ("mean" or "product")  
                                  default: "mean"

    Returns:
        - aligned_factors1 (list): first set of factor matrices, organized as 
                                   dims x [dim_length x rank], and aligned with 
                                   aligned_factors2, in order of strongest to 
                                   weakest match
        - aligned_factors2 (list): second set of factor matrices, structured as 
                                   dims x [dim_length x rank], and aligned with 
                                   aligned_factors1, in order of strongest to 
                                   weakest match

    """

    if len(factors1) != len(factors2):
        raise ValueError("factors1 and factors2 must have the same length.")

    rank = get_check_rank(factors1)
    
    corr_mats = []
    for factor_mat1, factor_mat2 in zip(factors1, factors2):
        if factor_mat1.shape != factor_mat2.shape:
            raise ValueError(
                "Corresponding factor matrices of factors1 and factors2 must "
                "have the same shape."
                )
        corr_mats.append(
            np.corrcoef(factor_mat1.T, factor_mat2.T)[:rank, rank:]
        )

    corr_mats = np.asarray(corr_mats)
    
    # for each point, establish whether a flip would improve correlations
    if allow_inverting:
        flip_mat = np.sign(corr_mats)

        # ensure even number of flips per component
        uneven_flips = (np.prod(flip_mat, axis=0) == -1)
        if np.sum(uneven_flips):
            sort_mat = np.argsort(
                np.argsort(np.absolute(corr_mats), axis=0), axis=0
                )
            flip_mat[uneven_flips * (sort_mat == 0)] *= -1
    else:
        flip_mat = np.ones_like(corr_mats)

    corr_mats *= flip_mat # apply flips to correlation matrix

    aggreg_corr_mat = aggregate_correlations(corr_mats, 0, aggreg_stat)
    indices = np.where(np.ones_like(aggreg_corr_mat))

    comps1_order = []
    comps2_order = []
    for o in np.argsort(aggreg_corr_mat[indices])[::-1]: # high to low
        comps1_ind = indices[0][o]
        comps2_ind = indices[1][o]
        if comps1_ind not in comps1_order and comps2_ind not in comps2_order:
            comps1_order.append(comps1_ind)
            comps2_order.append(comps2_ind)
    
    corr_index = (slice(None), comps1_order, comps2_order)

    # align components (applying flips to comp2 component dimensions)
    aligned_factors1 = [factor_mat[:, comps1_order] for factor_mat in factors1]
    aligned_factors2 = [
        factor_mat[:, comps2_order] * flip_mat[corr_index][f]
        for f, factor_mat in enumerate(factors2)
        ]    
    
    aggreg_corr_data = aggreg_corr_mat[comps1_order, comps2_order]
    
    similarity = aggregate_correlations(
        aggreg_corr_data, axis=0, aggreg_stat=aggreg_stat
        )

    return aligned_factors1, aligned_factors2, similarity


#############################################
def reconstruct_data(factors, rank=None):
    """
    reconstruct_data(factors)

    Reconstructs data from components, expressed as a list of factors, up to 
    the specified rank (number of components).

    Required args:
        - factors (list): components, expressed as a list of factor matrices, 
                          structured as dims x [dim_length x rank]
        
    Optional args:
        - rank  (int): number of components to use in reconstruction (in order)
                       default: None

    Returns:
        - reconstructured_data (nd array): reconstructed data, the structure of 
                                           which is the length of each 
                                           component dimension
    """

    rank = get_check_rank(factors, rank=rank)

    dim_lengths = [len(factor_mat) for factor_mat in factors]
    R1s = np.zeros((*dim_lengths, rank))

    for i in range(rank):
        R1s[..., i] = rank_one_tensor(
            *[factor_mat[:, i] for factor_mat in factors]
            )
    
    reconstructed_data = R1s.sum(axis=-1)
        
    return reconstructed_data


#############################################
def get_check_dim_names(n_dims=3, dim_names=None):
    """
    get_check_dim_names()

    If dimension names are provided, checks that there are the correct number. 
    Otherwise, returns dimension names of the form [Dim 1, Dim 2, ...].

    Optional args:
        - n_dims (int)    : number of dimensions
                            default: 3
        - dim_names (list): name for each dimension
                            default: None

    Returns:
        - dim_names (list): name for each dimension
    """
    
    if dim_names is None:
        dim_names = [f"Dim {d + 1}" for d in range(n_dims)]
    elif len(dim_names) != n_dims:
        raise ValueError(
            "If providing dim_names, must be as many as the number of "
            f"component dimensions ({n_dims})."
            )
    
    return dim_names


#############################################
def plot_components(factors, factors2=None, rank=None, dim_names=None, 
                    plot_tog=False, ground_truth=False, title=None):
    """
    plot_components(factors)
    
    Plots components.

    Required args:
        - factors (list): components, expressed as a list of factor matrices, 
                          structured as dims x [dim_length x rank]

    Optional args:
        - factors2 (list)    : second set of components, expressed as a list of 
                               factor matrices, structured as 
                               dims x [dim_length x rank]
        - rank  (int)        : number of components to plot (in order)
                               default: None
        - dim_names (list)   : name for each component/data dimension
                               default: None
        - plot_tog (bool)    : if True, factors from different components are 
                               plotted in the same subplot, for each dimension
                               default: False 
        - ground_truth (bool): if True, factors2 are ground truth factors
                               default: False
        - title (str)        : plot title
                               default: None

    Returns:
        - ax (nd array): array of subplots
    """
    
    n_dims = len(factors)
    if factors2 is None:
        rank = get_check_rank(factors, rank)
    else:
        rank = min([get_check_rank(sub, rank) for sub in [factors, factors2]])

    n_rows = 1 if plot_tog else rank
    fig_width = 4 if plot_tog else int(n_rows * 1.2 + 1)
    fig, ax = plt.subplots(
        n_rows, n_dims, sharex="col", sharey=False, figsize=(9, fig_width),
        gridspec_kw={"wspace": 0.3}
        )

    alpha = 0.8 if plot_tog else 1.0
    label, lw = None, None
    if factors2 is not None:
        alpha, alpha_2 = 0.6, 0.6
        label, label_2 = None, None
        if ground_truth:
            label, label_2 = "Estimate", "Ground truth"
        lw, lw_2 = [2, 3]
        if plot_tog:
            raise ValueError("plot_tog can only be True if factors2 is None.")
        if len(factors2) != n_dims:
            raise ValueError(
                "If factors2 is not None, it must have the same length as "
                "factors."
                )
    
    dim_names = get_check_dim_names(n_dims, dim_names)
    for d in range(n_dims):
        for f in range(rank):
            sub_ax = ax[d] if plot_tog else ax[f, d]
            if f == rank - 1:
                sub_ax.set_xlabel(dim_names[d])
            if d == 0:
                if plot_tog:
                    ylabel = "Component factors"
                else:
                    ylabel = f"Comp. {f + 1}"
                sub_ax.set_ylabel(ylabel)
            if not plot_tog or f == 0:
                sub_ax.axhline(0, ls="dashed", color="k", alpha=0.6)

            sub_ax.plot(factors[d][:, f], alpha=alpha, label=label, lw=lw)
            if factors2 is not None:
                sub_ax.plot(
                    factors2[d][:, f], alpha=alpha_2, label=label_2, lw=lw_2, 
                    zorder=-3
                )
            if f == 0 and d == n_dims - 1 and label is not None:
                sub_ax.legend(frameon=False, fontsize="medium")
            
            # format axis
            sub_ax.spines["top"].set_visible(False)
            sub_ax.spines["right"].set_visible(False)

    if title is not None:
        title_y = 1.02 if plot_tog else 1 - 0.006 * rank
        fig.suptitle(title, y=title_y)

    return ax


#############################################
def run_cpd(data, rank=5, non_neg=False, use_tensorly=False, randst=None, 
            verbose=False):
    """
    run_cpd(data)

    Returns tensor component analysis (TCA) factors.

    Required args:
        - data (nd array): multi-dimensional data on which to run tensor 
                           component analysis.

    Optional args:
        - rank (int)         : decomposition rank (number of components to 
                               identify)
                               default: 5
        - non_neg (bool)     : if True non-negative decomposition is done
                               default: False
        - use_tensorly (bool): if True, tensorly is used
                               default: False
        - randst (int)       : random state
                               default: None
        - verbose (bool)     : if True, decomposition is done in verbose mode
                               default: False

    Returns:
        - CPTensor (KTensor): TCA factors object, with components stored as 
                              lists of factor matrices under the 'factors' 
                              attribute
    """

    decomp_kwargs = {
        "rank"        : rank,
        "random_state": randst,
        "verbose"     : verbose,
    }

    if non_neg:
        if use_tensorly:
            raise NotImplementedError(
                "tensorly version of non-negative CPD is not implemented here."
                )

        # Non-negative CPD by Alternating Least Squares (BCD)
        CPTensor = tensortools.ncp_bcd(data, **decomp_kwargs).factors
    else:
        if use_tensorly:
            CPTensor = tensorly_decomposition.parafac(data, **decomp_kwargs)
            CPTensor = tensortools.tensors.KTensor(CPTensor.factors)
        else:
            # Unconstrained CPD by Alternating Least Squares (ALS)
            CPTensor = tensortools.cp_als(data, **decomp_kwargs).factors
 
    return CPTensor


#############################################
def get_ex_ground_truth(n_comps=5, n_dims=3, non_neg=False, randst=None, 
                        dim_min=3, dim_max=13):
    """
    get_ex_ground_truth()

    Returns example ground truth components.

    Optional args:
        - n_comps (int) : number of components to extract
                          default: 5
        - n_dims (int)  : number of dimensions per component
                          default: 3
        - non_neg (bool): if True, ground truth components are non-negative
                          default: False
        - randst (int)  : int or random state
                          default: None
        - dim_min (int) : minimum dimension length to sample from
                          default: 3
        - dim_max (int) : maximum dimension length to sample from
                          default: 3
  
    Returns:
        - gt_CPTensor (KTensor): TCA factors object, with components stored as 
                                 lists of factor matrices under the 'factors' 
                                 attribute

    """
    
    randst = tensortools.data.random_tensor._check_random_state(randst)

    # sample lengths [3, 13[
    dim_lengths = [
        randst.randint(dim_max - dim_min) + dim_min for _ in range(n_dims)
        ]

    if non_neg:
        ktensor_func = tensortools.data.random_tensor.rand_ktensor
    else:
        ktensor_func = tensortools.data.random_tensor.randn_ktensor

    gt_CPTensor = ktensor_func(dim_lengths, n_comps, random_state=randst)
    
    return gt_CPTensor


#############################################
def load_data(data_path):
    """
    load_data(data_path)

    Loads data from the specified path.

    Required args:
        - data_path (Path): path to data, stored under the "data" key of an h5 
                            file
    
    Returns:
        - data (nd array): data array
    """

    data_path = Path(data_path)
    if not data_path.is_file():
        raise OSError(f"{data_path} not found.")
    if data_path.suffix != ".h5":
        raise ValueError("Expected data_path to be an h5 file.")
    
    with h5py.File(data_path, "r") as f:
        if "data" not in f.keys():
            raise RuntimeError(
                "Expected data_path to contain a 'data' dataset."
                )
        else:
            data = f["data"][()]
    
    return data


#############################################
def run_tca_example(data, ground_truth=False, n_comps=5, non_neg=False, 
                    scale=False, seed=None, dim_names=None, 
                    savename="unnamed", output_dir="tca_results"):
    """
    run_tca_example(data)

    Runs example TCA decomposition, and plots illustrative graphs.

    Required args:
        - data (nd array, list or KTensor): 
            multi-dimensional data array or, if ground_truth is True, ground 
            truth components stored as a KTensor or as a list of factor 
            matrices, structured as dims x [dim_length x rank]

    Optional args:
        - ground_truth (bool): if True, data contains ground truth components. 
                               Otherwise, data contains data on which to run 
                               TCA.
                               default: False
        - n_comps (int)      : number of components to use for TCA
                               default: 5
        - non_neg (bool)     : if True, non-negative TCA is done
                               default: False
        - scale (bool)       : if True, data is scaled before running TCA
                               (does not apply if ground_truth is True)
                               default: False
        - seed (int)         : int or random state
                               default: None
        - dim_names (list)   : names for each data/component dimension
                               default: None
        - savename (str)     : base name under which to save files
                               default: "unnamed"
        - output_dir (Path)  : directory to save output files to
                               default: "tca_results"
    """

    randst = tensortools.data.random_tensor._check_random_state(seed)
    seed_str = f"_seed{seed}" if isinstance(seed, int) else ""


    if ground_truth:
        gt_str = "_gt"
        if scale:
            warnings.warn("Setting 'scale' to False for ground_truth analysis.")
            scale = False
        if not isinstance(data, tensortools.tensors.KTensor):
            gt_CPTensor = tensortools.tensors.KTensor(data)
        else:
            gt_CPTensor = data
        data = reconstruct_data(gt_CPTensor.factors)
    else:
        gt_str = ""
        ground_truth = False

    # get parameter strings, and optionally scale data
    non_neg_str = "_non_neg" if non_neg else ""
    non_neg_str_pr = "non-negative " if non_neg else ""
    scale_str, scale_str_pr = "", ""
    if scale:
        scale_axis = tuple(range(1, len(data.shape)))
        data, _, _ = scale_data(data, axis=scale_axis)
        scale_str, scale_str_pr = "_scaled", " (after scaling)"

    # Run decompositions to get 2 versions of the components, 
    # or 1 and ground-truth
    logger.info("Performing tensor decomposition...")
    CPTensors = []
    fit_to_data = []
    for i in range(2):
        if i == 1 and ground_truth:
            CPTensors.append(gt_CPTensor)
        else:
            CPTensors.append(
                run_cpd(
                    data, rank=n_comps, non_neg=non_neg, randst=randst
                    )
            )
        recon_data = reconstruct_data(CPTensors[-1].factors)
        fit = np.corrcoef(data.reshape(-1), recon_data.reshape(-1))[1, 0]
        fit_to_data.append(fit)


    # Prepare for plotting
    logger.info("Plotting results...")
    gen_title = \
        f"{non_neg_str_pr.capitalize()}Tensor Component Analysis{scale_str_pr}"
    Path(output_dir).mkdir(exist_ok=True, parents=True)


    # Plot first set of components
    title = f"{gen_title}: First fit\n(data fit: {fit_to_data[0]:.2f})"
    for plot_tog in [False, True]:
        tog_str = "_tog" if plot_tog else ""
        ax = plot_components(
            CPTensors[0].factors, dim_names=dim_names, plot_tog=plot_tog, 
            title=title
            )
        full_savename = (f"{savename}{scale_str}{non_neg_str}_"
            f"tca{gt_str}{tog_str}{seed_str}.svg")

        savepath = Path(output_dir, full_savename)
        ax.reshape(-1)[0].figure.savefig(savepath, bbox_inches="tight")


    # Plot sets of components or components vs ground-truth against each other
    gen_title = f"{non_neg_str_pr}TCA{scale_str_pr}"
    for align_type in ["simple", "Kruskal"]:
        if align_type == "simple":
            # reverse order so that, if factors2 is ground truth, its factors 
            # aren't inverted during alignment
            aligned_factors2, aligned_factors1, sim = simple_align_comps(
                CPTensors[1].factors, CPTensors[0].factors, 
            )
        else:
            sim = tensortools.diagnostics.kruskal_align(
                CPTensors[0], CPTensors[1], permute_U=True, permute_V=True
                )
            aligned_factors1 = CPTensors[0].factors
            aligned_factors2 = CPTensors[1].factors

        align_str = f"_{align_type.lower()}_align"
        full_savename = (f"{savename}{scale_str}{non_neg_str}_"
            f"tca_REPLACE{align_str}{seed_str}.svg")

        if ground_truth:
            base_title = (f"Comparing {gen_title} and ground truth "
                f"components{scale_str_pr},\nusing {align_type} alignment")
            full_savename = full_savename.replace("tca_REPLACE", "tca_vs_gt")
        else:
            base_title = (
                f"Comparing 2 sets of {gen_title} components{scale_str_pr},\n"
                f"using {align_type} alignment"
                )
            full_savename = full_savename.replace("tca_REPLACE", "tca_comp")

        title = (f"{base_title}"
            f"\n(data fits: {fit_to_data[0]:.2f} vs {fit_to_data[1]:.2f} / "
            f"comp. similarity: {sim:.2f})")
        
        ax = plot_components(
            aligned_factors1, factors2=aligned_factors2, dim_names=dim_names, 
            title=title, ground_truth=ground_truth
            )
        savepath = Path(output_dir, full_savename)
        ax.reshape(-1)[0].figure.savefig(savepath, bbox_inches="tight")


#############################################
def main(data_path="ground_truth_test", n_comps=5, non_neg=False, 
         scale=False, seed=None, dim_names=None, savename=None, 
         output_dir="tca_results"):
    """
    main()

    Retrieves data or generates example ground truth components, and runs an 
    example TCA decomposition, plotting illustrative graphs.

    Optional args:
        - data_path (Path or str): path to data, stored under the "data" key of 
                                   an h5 file or "ground_truth_test" if 
                                   running analysis from randomly generated 
                                   components
                                   default: "ground_truth_test" 
        - n_comps (int)      : number of components to use for TCA
                               default: 5
        - non_neg (bool)     : if True, non-negative TCA is done
                               default: False
        - scale (bool)       : if True, data is scaled before running TCA
                               (does not apply if ground_truth is True)
                               default: False
        - seed (int)         : random state seed
                               default: None
        - dim_names (list)   : names for each data/component dimension
                               default: None           
        - savename (str)     : base name under which to save output files
                               default: "unknown"                  
        - output_dir (Path)  : directory to save output files to
                               default: "tca_results"

    """

    set_logger()

    if data_path == "ground_truth_test":
        if savename is None:
            savename = "random_test"
        ground_truth = True
        n_dims = 3 if dim_names is None else len(dim_names)
        data = get_ex_ground_truth(
            n_comps, n_dims, non_neg=non_neg, randst=seed
            )
    else:
        if savename is None:
            savename = "unknown"
        ground_truth = False
        data = load_data(data_path)

    run_tca_example(
        data, 
        ground_truth=ground_truth, 
        n_comps=5, 
        non_neg=non_neg, 
        scale=scale, 
        seed=seed, 
        dim_names=dim_names, 
        savename=savename,
        output_dir=output_dir
        )


#############################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path", default="ground_truth_test", 
        help=("path to data to load (under the 'data' key of an h5 file) or "
            "'ground_truth_test' for an example with random data.")
        )
    parser.add_argument(
        "--n_comps", default=5, type=int, 
        help="number of components to extract (i.e., rank or number of factors)"
        )
    parser.add_argument(
        "--non_neg", action="store_true", 
        help="if True, non-negative decomposition is used"
        )
    parser.add_argument(
        "--scale", action="store_true", 
        help=("if True, data is scaled before decomposition, along all but "
            "the first dimension")
        )
    parser.add_argument(
        "--dim_names", default="neurons_trials_frames", 
        help="data/component dimension names, separated by '_'."
        )
    parser.add_argument("--seed", default=None, help="random state seed")
    parser.add_argument("--savename", default=None, 
        help="base name under which to save output files")
    parser.add_argument("--output_dir", default="tca_results", type=Path)

    args = parser.parse_args()

    # format a few arguments
    args.dim_names = [
        f"{name[0].upper()}{name[1:]}" for name in args.dim_names.split("_")
        ]
    if args.seed is not None:
        args.seed = int(args.seed)

    main(**args.__dict__)

    plt.close("all")

