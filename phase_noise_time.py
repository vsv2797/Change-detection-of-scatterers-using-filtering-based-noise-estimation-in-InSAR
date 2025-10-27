"""Change detection with temporal phase unwrapping of the interferometric phases."""
# SpaTZ, SPAtioTemporal Selection of Scatterers
#
# Copyright (c) 2024  Andreas Piter
# (Institute of Photogrammetry and GeoInformation, Leibniz University Hannover)
# Contact: piter@ipi.uni-hannover.de
#
# This software was developed within the context of the PhD thesis of Andreas Piter.
# The software is not yet licensed and used for internal development only.
import logging
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
import cmcrameri as cmc

from miaplpy.objects.slcStack import slcStack
from mintpy.utils import ptime
from mintpy.utils.plot import auto_flip_direction

from sarvey.unwrapping import oneDimSearchTemporalCoherence

from .likelihood import logLikelihoodInterferometricPhase


def runTemporalNoiseEstimation(*,
                               slc_stack_obj: slcStack,
                               slc: np.ndarray,
                               mask_cand: np.ndarray,
                               logger: logging.Logger,
                               adi_p1: float,
                               adi_tcs: float,
                               num_nearest_neighbours: int,
                               loc_inc: np.ndarray,
                               slant_range: np.ndarray,
                               wavelength: float,
                               demerr_bound: float,
                               velocity_bound: float,
                               num_samples: int,
                               num_cores: int = 1,
                               show_plots: bool = False) -> (np.ndarray, np.ndarray, np.ndarray):
    """Run change detection based on phase noise estimation with temporal displacement model.

    Steps:
    1. Compute the amplitude dispersion for all pixels in the SLC stack.
    2. Select the first-order points based on the amplitude dispersion.
    3. Select the TCS candidates based on the amplitude dispersion and intersect them with the given candidate mask.
    For each TCS candidate:
    4. Search the closest first-order point neighbours for the TCS candidate.
    5. Apply change detection assuming a single change point in the data based on the method from Doerr et al. (2022).
    6. Store the change time and coherence of each TCS candidate.

    Parameters
    ----------
    slc_stack_obj : slcStack
        SLC stack object.
    slc : np.ndarray
        SLC stack (number SLC images x length x width).
    mask_cand : np.ndarray
        Binary mask image indicating the location of TCS candidates that shall be examined (length x width).
    logger : logging.Logger
        Logger object.
    adi_p1 : float
        Threshold on amplitude dispersion for selecting first-order points.
    adi_tcs : float
        Threshold on amplitude dispersion for selecting TCS candidates.
    num_nearest_neighbours : int
        Number of nearest first-order point neighbours for computing arcs to a TCS. Nearest first-order point neighbours
        are used to create a virtual reference for the TCS candidate.
    loc_inc : np.ndarray
        Image with local incidence angle of the satellite w.r.t. each pixels (length x width).
    slant_range : np.ndarray
        Image with slant range distance between sensor and pixel (length x width).
    wavelength : float
        Radar wavelength of the sensor.
    demerr_bound : float
        Bound for the DEM error search space for estimating the DEM error in temporal unwrapping.
    velocity_bound : float
        Bound for the velocity search space for estimating the velocity in temporal unwrapping.
    num_samples : int
        Number of samples for the search space in the temporal unwrapping.
    num_cores : int
        Number of cores for parallel processing.
    show_plots : bool
        Show plots for debugging.

    Returns
    -------
    change_time_map : np.ndarray
        Image with the estimated change time index for each pixel (length x width). Change time is within interval
        [0, number of images - 1]. The change time index is np.nan for pixels that could not be processed.
    coherence_map : np.ndarray
        Image with the coherence for each pixel (length x width) after change detection. The coherence is evaluated for
        each of the two temporal subsets resulting from the change time. The returned coherence belongs to the maximum
        coherence of the two subsets.
    coherence_initial_map : np.ndarray
        Image with the coherence for each pixel (length x width) before change detection. The coherence is evaluated for
        the whole time span of the SLC stack.
    """
    logger.info(msg="TEMPORAL NOISE ESTIMATION Change detection.")

    # set up reference network
    amp_disp = computeAmplitudeDispersion(slc=slc)
    mask_p1 = amp_disp <= adi_p1
    coord_xy_p1 = np.array(np.where(mask_p1)).T
    num_p1 = coord_xy_p1.shape[0]

    # mask_cand = mask_p1  # debug: for the TCS candidates use all first-order points.
    mask_cand = (amp_disp <= adi_tcs) & mask_cand

    loc_inc = loc_inc[mask_cand]
    slant_range = slant_range[mask_cand]
    slc_cand = slc[:, mask_cand]
    slc_p1 = slc[:, mask_p1]
    num_cand = slc_cand.shape[1]
    coord_xy_cand = np.array(np.where(mask_cand)).T

    if show_plots:
        fig = plt.figure(figsize=(10, 5))
        ax0 = fig.add_subplot(121)
        plt.imshow(amp_disp, cmap=cmc.cm.cmaps["grayC_r"], interpolation='nearest')
        cbar = plt.colorbar(pad=0.03, shrink=0.5)
        cbar.set_label("Amplitude dispersion")
        plt.clim(0, 1)
        plt.plot(coord_xy_p1[:, 1], coord_xy_p1[:, 0], 'Dr', markersize=2, label="First-order points", linestyle='None')
        plt.plot(coord_xy_cand[:, 1], coord_xy_cand[:, 0], '.b', markersize=2, label="TCS candidates")
        plt.xlabel("Range")
        plt.ylabel("Azimuth")
        plt.legend()
        auto_flip_direction(slc_stack_obj.metadata, ax=ax0, print_msg=False)

        ax1 = fig.add_subplot(122)
        ax1.hist(amp_disp.flatten(), bins=200, color='k')
        ax1.axvline(x=adi_p1, color='r', linestyle='--', label="Threshold for first-order points")
        plt.legend()
        plt.xlim(0, 1)
        plt.xlabel("Amplitude dispersion")
        plt.ylabel("Number of pixels")
        plt.subplots_adjust(wspace=0.5)
        lines = [ax0.plot([], [], '-w', linewidth=1)[0] for i in range(num_nearest_neighbours)]

    if num_p1 < num_nearest_neighbours:
        logger.error(msg=f"Number of pixels found for first-order point selection {num_p1} is smaller than the number "
                         f"of nearest neighbours {num_nearest_neighbours}")
        raise Exception

    if num_p1 == 0:
        logger.error(msg=f"No pixels found for first-order point selection based on threshold {adi_p1}")
        raise Exception

    searchtree = KDTree(coord_xy_p1)

    step_time = np.zeros(num_cand, dtype=np.int16)
    max_coherence = np.zeros(num_cand, dtype=np.float32)
    initial_coherence = np.zeros(num_cand, dtype=np.float32)

    prog_bar = ptime.progressBar(maxValue=num_cand)
    for idx in range(num_cand):
        dist, idx_p1 = searchtree.query(coord_xy_cand[idx], k=num_nearest_neighbours + 1)
        idx_p1 = idx_p1[1:]  # remove the first element, which is the point itself.

        # create virtual reference
        slc_virt_ref = slc_p1[:, idx_p1]

        if show_plots:
            # plot the arcs that are used for the virtual reference
            p_cur = coord_xy_cand[idx]
            for i in range(idx_p1.shape[0]):
                p_ref = coord_xy_p1[idx_p1[i]]
                lines[i].set_data([p_cur[1], p_ref[1]], [p_cur[0], p_ref[0]])
            plt.draw()

        score, max_coh, coherence_0 = singleCD(
            slc_cand=slc_cand[:, idx],
            slc_virt_ref=slc_virt_ref,
            tbase=slc_stack_obj.tbase / 365.25,  # important to convert to years as the velocity is in m/yr
            pbase=slc_stack_obj.pbase,
            loc_inc=float(loc_inc[idx]),
            slant_range=float(slant_range[idx]),
            wavelength=wavelength,
            demerr_bound=demerr_bound,
            velocity_bound=velocity_bound,
            num_samples=num_samples,
            show_plots=show_plots
        )

        step_time[idx] = np.nanargmin(score)  # fixme: check if + 1
        max_coherence[idx] = max_coh
        initial_coherence[idx] = coherence_0

        prog_bar.update(idx + 1, every=2, suffix='{}/{} TCS processed.'.format(idx + 1, num_cand))

    prog_bar.close()

    logger.info(f"{num_cand} TCS candidates processed.")

    img_size = slc.shape[1:]
    change_time_map = np.zeros(img_size, dtype=np.int16)
    change_time_map[mask_cand] = step_time

    coherence_map = np.zeros(img_size, dtype=np.float32)
    coherence_map[mask_cand] = max_coherence

    coherence_initial_map = np.zeros(img_size, dtype=np.float32)
    coherence_initial_map[mask_cand] = initial_coherence
    return change_time_map, coherence_map, coherence_initial_map


def singleCD(*, slc_cand: np.ndarray,
             slc_virt_ref: np.ndarray,
             show_plots: bool = False,
             tbase: np.ndarray,
             pbase: np.ndarray,
             loc_inc: float,
             slant_range: float,
             wavelength: float,
             demerr_bound: float,
             velocity_bound: float,
             num_samples: int) -> (np.ndarray, float, float):
    """Change detection with temporal phase noise estimation assuming a single change point in the data.

    This method temporally unwraps the arc phase for all possible splits of the time series into two distinct temporal
    subsets. The method follows the approach presented in Doerr et al. 2022, IEEE TGRS.

    Steps:
    A) Evaluate the likelihood for the whole time span:
    1. Compute the single-reference interferograms and baselines for the whole time span for both virtual reference and
       TCS candidate.
    2. Compute the wrapped arc phase for the arc between virtual reference and TCS candidate.
    3. Temporally unwrap the arc phase and retrieve the phase noise and coherence.
    4. Center the phase noise around zero in the complex domain.
    5. Evaluate the likelihood for the whole time span.

    B) Evaluate the likelihood for all possible splits of the time series:
    For each possible split of the time series into two subsets:
    6. Compute the single-reference interferograms and baselines for the two subsets for both virtual reference and TCS
       candidate. The reference acquisition of the interferogram networks is chosen to be the image before and after the
       split, respectively.
    7. Do step 2 to 5 for both subsets.
    8. Compute the likelihood ratio test from the likelihood of the whole time span and the likelihoods of the two
       subsets. The likelihood ratio test is normalized by the number of interferograms in the whole time span and the
       number of interferograms in the two subsets, respectively.

    Parameters
    ----------
    slc_cand : np.ndarray
        SLC time series of the TCS candidate (number of images x 1).
    slc_virt_ref : np.ndarray
        SLC stack of the points that will form the virtual reference (number of images x number of points).
    show_plots : bool
        Show plots for debugging.
    tbase : np.ndarray
        Temporal baselines of the SLC stack in decimal years (number of images x 1).
    pbase : np.ndarray
        Perpendicular baselines of the SLC stack in meters (number of images x 1).
    loc_inc : float
        Local incidence angle of the satellite w.r.t. the pixel.
    slant_range : float
        Slant range distance between sensor and pixel.
    wavelength : float
        Radar wavelength of the sensor.
    demerr_bound : float
        Bound for the DEM error search space for estimating the DEM error in temporal unwrapping.
    velocity_bound : float
        Bound for the velocity search space for estimating the velocity in temporal unwrapping.
    num_samples : int
        Number of samples for the search space in the temporal unwrapping.

    Returns
    -------
    score_norm_ifgs : np.ndarray
        Normalized score for all possible splits of the time series (number of images - 1 x 1).
    max_coh : float
        Maximum coherence of the two temporal subsets. The coherence is evaluated for the two subsets resulting from the
        estimated change point.
    coherence_0 : float
        Coherence for the whole time span.
    """
    num_images = slc_cand.shape[0]
    slc_all = np.concatenate((slc_cand.reshape(-1, 1), slc_virt_ref), axis=1)
    mask_time = np.zeros(num_images, dtype=bool)

    score_norm_ifgs = np.zeros((num_images - 1,))
    score_norm_ifgs[:] = np.nan  # indicate values which are not computed by nan

    # Compute the phase noise and coherence for the whole time span
    phase_0 = 0.0  # expected value of phase is assumed to be Zero
    num_0 = num_images - 1

    ifgs_all, tbase_ifg, pbase_ifg = computeIfgsAndBaselines(
        slc_all=slc_all,
        pbase=pbase,
        tbase=tbase,
        ref_idx=slc_all.shape[0] // 2  # take arbitrary image in the middle of time series
    )

    # complex-valued mean phase
    ifgs_virt_ref = np.mean(ifgs_all[:, 1:], axis=1)
    arc_phase = np.angle(ifgs_virt_ref * np.conjugate(ifgs_all[:, 0]))  # re-wrapped
    arc_phase_all = np.angle(ifgs_all[:, 1:] * np.conjugate(ifgs_all[:, :1]))  # re-wrapped

    if show_plots:
        plt.figure()
        plt.plot(arc_phase_all, '.', label="Arc phase to individual first-order points")
        plt.plot(arc_phase, '.k', label="Arc phase to virtual reference")
        plt.xlabel("Interferogram index")
        plt.ylabel("Phase")
        plt.legend()
        plt.show()

    coherence_0, phase_noise_0 = temporallyUnwrapArcPhase(
        arc_phase=arc_phase,
        tbase_ifg=tbase_ifg,
        pbase_ifg=pbase_ifg,
        demerr_bound=demerr_bound,
        velocity_bound=velocity_bound,
        num_samples=num_samples,
        wavelength=wavelength,
        slant_range=slant_range,
        loc_inc=loc_inc,
        show_plots=show_plots
    )
    phase_noise_0 = np.angle(np.exp(1j * phase_noise_0) * np.conjugate(np.mean(np.exp(1j * phase_noise_0))))
    p0 = logLikelihoodInterferometricPhase(coherence=coherence_0, ifg_res=phase_noise_0, phase_0=phase_0)
    """
    The LRT is not computed for the first and last possible subsets which have only two images each. Hence, there is
    only one interferogram possible which yields a coherence of 1 and the marginal distribution is 0. As log(0) is -inf,
    the score is not valid for these change times.
    Addditionally, centering the phase noise around zero in the complex domain leads to 0 values in case of 3 images.
    As a consequence, the minimum size of a subset is 4 images, i.e. 3 single-reference interferograms.
    """
    all_p1 = np.zeros((num_images - 1,))
    all_p2 = np.zeros((num_images - 1,))
    all_coh1 = np.zeros((num_images - 1,))
    all_coh2 = np.zeros((num_images - 1,))
    all_p1[:] = np.nan
    all_p2[:] = np.nan
    all_coh1[:] = np.nan
    all_coh2[:] = np.nan
    for step_time_idx in range(3, num_images - 4):
        mask_time[:] = False  # reset mask
        mask_time[:step_time_idx + 1] = True

        ifgs_all_1, tbase_ifg_1, pbase_ifg_1 = computeIfgsAndBaselines(
            slc_all=slc_all[mask_time, :],
            pbase=pbase[mask_time],
            tbase=tbase[mask_time],
            ref_idx=-1
        )
        ifgs_all_2, tbase_ifg_2, pbase_ifg_2 = computeIfgsAndBaselines(
            slc_all=slc_all[~mask_time, :],
            pbase=pbase[~mask_time],
            tbase=tbase[~mask_time],
            ref_idx=0
        )

        ifgs_virt_ref1 = np.mean(ifgs_all_1[:, 1:], axis=1)
        ifgs_virt_ref2 = np.mean(ifgs_all_2[:, 1:], axis=1)

        arc_phase_1 = np.angle(ifgs_virt_ref1 * np.conjugate(ifgs_all_1[:, 0]))
        arc_phase_2 = np.angle(ifgs_virt_ref2 * np.conjugate(ifgs_all_2[:, 0]))

        coherence_1, phase_noise_1 = temporallyUnwrapArcPhase(
            arc_phase=arc_phase_1,
            tbase_ifg=tbase_ifg_1,
            pbase_ifg=pbase_ifg_1,
            demerr_bound=demerr_bound,
            velocity_bound=velocity_bound,
            num_samples=num_samples,
            wavelength=wavelength,
            slant_range=slant_range,
            loc_inc=loc_inc,
            show_plots=False
        )

        coherence_2, phase_noise_2 = temporallyUnwrapArcPhase(
            arc_phase=arc_phase_2,
            tbase_ifg=tbase_ifg_2,
            pbase_ifg=pbase_ifg_2,
            demerr_bound=demerr_bound,
            velocity_bound=velocity_bound,
            num_samples=num_samples,
            wavelength=wavelength,
            slant_range=slant_range,
            loc_inc=loc_inc,
            show_plots=False
        )
        phase_noise_1 = np.angle(np.exp(1j * phase_noise_1) * np.conjugate(np.mean(np.exp(1j * phase_noise_1))))
        phase_noise_2 = np.angle(np.exp(1j * phase_noise_2) * np.conjugate(np.mean(np.exp(1j * phase_noise_2))))

        num_1 = phase_noise_1.shape[0]
        num_2 = phase_noise_2.shape[0]
        if (num_1 == 1) or (num_2 == 1):
            print("WARNING: Subset has only 1 interferogram.")
        p1 = logLikelihoodInterferometricPhase(coherence=coherence_1, ifg_res=phase_noise_1, phase_0=phase_0)
        p2 = logLikelihoodInterferometricPhase(coherence=coherence_2, ifg_res=phase_noise_2, phase_0=phase_0)

        score_norm_ifgs[step_time_idx] = (p0 / num_0 - ((p1 + p2) / (num_1 + num_2)))
        all_p1[step_time_idx] = p1
        all_p2[step_time_idx] = p2
        all_coh1[step_time_idx] = coherence_1
        all_coh2[step_time_idx] = coherence_2

    max_coh = np.max([all_coh1[np.nanargmin(score_norm_ifgs)], all_coh2[np.nanargmin(score_norm_ifgs)]])

    if show_plots:
        plt.figure()
        plt.plot(all_p1, label="Subset 1")
        plt.plot(all_p2, label="Subset 2")
        plt.legend()
        plt.xlabel("Time index")
        plt.ylabel("Log-likelihood")

        plt.figure()
        plt.plot(all_coh1, label="Subset 1")
        plt.plot(all_coh2, label="Subset 2")
        plt.legend()
        plt.ylim(0, 1)
        plt.xlabel("Time index")
        plt.ylabel("Coherence")

        plt.figure()
        plt.plot(score_norm_ifgs, '-k')
        plt.xlabel("Time index")
        plt.ylabel("Normalized score")
        plt.show()
    return score_norm_ifgs, max_coh, coherence_0


def computeAmplitudeDispersion(*, slc: np.ndarray) -> np.ndarray:
    """Compute amplitude dispersion from SLC stack.

    Amplitude dispersion is defined as the standard deviation of the amplitude divided by the mean amplitude and is
    used as an approximation for the phase noise (Ferretti et al. 2001).

    Parameters
    ----------
    slc : np.ndarray
        SLC stack.

    Returns
    -------
    amp_disp : np.ndarray
        Amplitude dispersion.
    """
    amp_mean = np.mean(np.abs(slc), axis=0)
    amp_std = np.std(np.abs(slc), axis=0)
    amp_disp = amp_std / amp_mean
    return amp_disp


def computeIfgsAndBaselines(*,
                            slc_all: np.ndarray,
                            tbase: np.ndarray,
                            pbase: np.ndarray,
                            ref_idx: int) -> (np.ndarray, np.ndarray, np.ndarray):
    """Compute the single-reference interferograms and baselines for the given SLC stack.

    Parameters
    ----------
    slc_all : np.ndarray
        SLC time series (number of images x number of points).
    tbase : np.ndarray
        Temporal baselines of the SLC stack in decimal years (number of images x 1).
    pbase : np.ndarray
        Perpendicular baselines of the SLC stack in meters (number of images x 1).
    ref_idx : int
        Index of the reference image.

    Returns
    -------
    ifgs_all : np.ndarray
        Complex signal of the interferograms formed with a single-reference (number of images - 1 x number of points).
    tbase_ifg : np.ndarray
        Temporal baselines of the interferograms in decimal years (number of images - 1 x 1).
    pbase_ifg : np.ndarray
        Perpendicular baselines of the interferograms in meters (number of images - 1 x 1).
    """
    ifgs_all = slc_all * np.conjugate(slc_all[ref_idx, :])  # complex-valued phase
    tbase_ifg = tbase - tbase[ref_idx]
    pbase_ifg = pbase - pbase[ref_idx]
    ifgs_all = np.delete(ifgs_all, ref_idx, axis=0)
    tbase_ifg = np.delete(tbase_ifg, ref_idx, axis=0)
    pbase_ifg = np.delete(pbase_ifg, ref_idx, axis=0)
    return ifgs_all, tbase_ifg, pbase_ifg


def temporallyUnwrapArcPhase(*, arc_phase: np.ndarray,
                             tbase_ifg: np.ndarray,
                             pbase_ifg: np.ndarray,
                             demerr_bound: float,
                             velocity_bound: float,
                             num_samples: int,
                             wavelength: float,
                             slant_range: float,
                             loc_inc: float,
                             show_plots: bool = False) -> (float, np.ndarray):
    """Temporally unwrap the arc phase assuming parameters 'DEM error' and 'velocity' (i.e. linear displacement model).

    Parameters
    ----------
    arc_phase : np.ndarray
        Wrapped arc phase (number of interferograms x 1).
    tbase_ifg : np.ndarray
        Temporal baselines of the interferograms in decimal years (number of interferograms x 1).
    pbase_ifg : np.ndarray
        Perpendicular baselines of the interferograms in meters (number of interferograms x 1).
    demerr_bound : float
        Bound for the DEM error search space for estimating the DEM error in temporal unwrapping.
    velocity_bound : float
        Bound for the velocity search space for estimating the velocity in temporal unwrapping.
    num_samples : int
        Number of samples for the search space in the temporal unwrapping.
    wavelength : float
        Radar wavelength of the sensor.
    slant_range : float
        Slant range distance between sensor and pixel.
    loc_inc : float
        Local incidence angle of the satellite w.r.t. the pixel.
    show_plots : bool
        Show plots for debugging.

    Returns
    -------
    coherence : float
        Coherence of the phase noise.
    phase_noise : np.ndarray
        Phase noise after substracting the estimated parameters from the observed phase (number of interferograms x 1).
    """
    design_mat = np.zeros((arc_phase.shape[0], 2), dtype=np.float32)

    demerr_range = np.linspace(-demerr_bound, demerr_bound, num_samples)
    vel_range = np.linspace(-velocity_bound, velocity_bound, num_samples)

    factor = 4 * np.pi / wavelength

    design_mat[:, 0] = factor * pbase_ifg / (slant_range * np.sin(loc_inc))
    design_mat[:, 1] = factor * tbase_ifg

    demerr, vel, gamma = oneDimSearchTemporalCoherence(
        demerr_range=demerr_range,
        vel_range=vel_range,
        obs_phase=arc_phase,
        design_mat=design_mat
    )
    pred_phase = np.matmul(design_mat, np.array([demerr, vel], dtype=np.float32))
    ifg_res = arc_phase - pred_phase
    if show_plots:
        fig = plt.figure(figsize=(10, 5))
        ax0 = fig.add_subplot(121)
        ax0.plot(ifg_res, '.', label="Residual phase")
        ax0.plot(pred_phase, '.', label="Predicted phase")
        ax0.plot(arc_phase, '.', label="Observed phase")
        plt.legend()
        plt.xlabel("Interferogram index")
        plt.ylabel("Phase")

        ax1 = fig.add_subplot(122)
        ax1.hist(ifg_res.flatten(), bins=200, color='k')
        plt.xlabel("Phase noise")
        plt.ylabel("Number of pixels")
        plt.subplots_adjust(wspace=0.3)
    return gamma, ifg_res
