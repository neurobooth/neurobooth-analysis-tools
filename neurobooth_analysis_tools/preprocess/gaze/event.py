"""
An algorithm for detection of gaze events (i.e., saccades and fixations).

Reference code: https://github.com/richardschweitzer/OnlineSaccadeDetection
The R version has an implementation of the Engbert-Kliegl algorithm with endpoint refinement based on post-saccadic
oscillations.
PSO refinement not currently implemented (as it may make it easier to identify dysmetric saccades in our analyses).

Additional reference code for blink detection: https://github.com/tmalsburg/saccades
This code uses the same base algorithm, and has heuristics included for blink detection.
"""

import numpy as np
from typing import NamedTuple, Optional

import scipy.signal as signal
from neurobooth_analysis_tools.preprocess.mask import detect_bool_edges


class DetectionResult(NamedTuple):
    saccade_mask: np.ndarray
    fixation_mask: np.ndarray
    blink_mask: np.ndarray
    radius: np.ndarray  # Parameters of elliptic threshold for saccade detection


class SaccadeDetectionException(Exception):
    def __init__(self, *args):
        super(SaccadeDetectionException, self).__init__(*args)


def detect_gaze_events(
        pos: np.ndarray,
        vel: np.ndarray,
        ts: np.ndarray,
        vfac: float = 5,
        min_duration: float = 0.016,
        smooth_velocity: bool = True,
        smooth_window_size: int = 5,
        resample: bool = False,
        fixed_radius: Optional[np.ndarray] = None,
        dfac: float = 4,
) -> DetectionResult:
    """
    Apply the Engbert-Kliegl (2003) algorithm for saccade detection.
    Fixations are defined as periods between saccades.
    Some fixations are further re-classified as blinks based on their dispersion.
    (It is assumed that the eye tracking system provides constant values (e.g., 0s) during blinks

    :param pos: Nx2 array of gaze position. First column is x, second column is y.
    :param vel: Nx2 array of gaze velocity (dva/s). First column is x, second column is y.
    :param ts: Timestamp of each sample (in seconds).
    :param vfac: Relative velocity thresholds; lambda in the paper.
    :param min_duration: Minimum saccade duration (in seconds).
    :param smooth_velocity: Whether to perform velocity smoothing. May be detrimental with high-precision data.
    :param smooth_window_size: Width of the moving average window used for velocity smoothing.
    :param resample: Whether to resample the position time-series before performing detection.
    :param fixed_radius: If not None, ignore vfac/adaptive estimate and used the provided elliptic threshold.
    ([x, y] as ndarray). Can be useful if calculating the threshold at a population/task level.
    :param dfac: Relative dispersion threshold for blink/artifact detection.
    :return: A named tuple containing Boolean masks for gaze events as well as the saccade detection radius.
    """
    if resample:  # Resample the signal if the sampling rate is unsteady
        pos, ts = signal.resample(pos, pos.shape[0], ts, axis=0)

    # Calculate and (optionally) smooth velocity
    if smooth_velocity:
        vel[:, 0] = _smooth_signal(vel[:, 0], smooth_window_size)
        vel[:, 1] = _smooth_signal(vel[:, 1], smooth_window_size)

    # Detect saccades (and by extension, fixations)
    radius = calc_detection_radius(vel, vfac) if fixed_radius is None else fixed_radius
    saccade_mask = detect_saccades(vel, ts, radius, min_duration)
    fixation_mask = ~saccade_mask

    # Detect which fixations should be re-labeled as blinks
    blink_mask = detect_blinks(pos, fixation_mask, dfac)
    fixation_mask &= ~blink_mask

    return DetectionResult(
        saccade_mask=saccade_mask,
        fixation_mask=fixation_mask,
        blink_mask=blink_mask,
        radius=radius
    )


def calc_detection_radius(
        vel: np.ndarray,
        vfac: float,
) -> np.ndarray:
    """
    Adaptively compute the elliptical radius for saccade detection.

    :param vel: Nx2 array of gaze velocity. First column is x, second column is y.
    :param vfac: Relative velocity thresholds; lambda in the paper.
    :return: A 2-element array containing the x and y detection radii.
    """
    vel_median = np.nanmedian(vel, axis=0, keepdims=True)

    msd = np.sqrt(np.nanmedian(np.square(vel - vel_median), axis=0))
    msd[0] = _correct_small_std(msd[0], vel[:, 0])
    msd[1] = _correct_small_std(msd[1], vel[:, 1])

    radius = vfac * msd
    return radius


def detect_saccades(
        vel: np.ndarray,
        ts: np.ndarray,
        radius: np.ndarray,
        min_duration: float,
) -> np.ndarray:
    """
    Identify saccade and fixation boundaries based on pre-computed elliptical detection radii.

    :param vel: Nx2 array of gaze velocity. First column is x, second column is y.
    :param ts: Timestamp of each sample (in seconds).
    :param radius: A 2-element array containing the x and y detection radii.
    :param min_duration: Minimum saccade duration (in seconds).
    :return: A Boolean mask indicating the presence of saccades
    """
    # Determine when the velocity greater than the detection radius
    test = np.sum(np.square(vel / radius[np.newaxis, :]), axis=1)
    test = (test > 1) & ~np.isnan(test)
    test_edges = detect_bool_edges(test, include_endpoints=True)

    # Create a list of saccade onset and end indices from the identified candidate saccades
    sacc_onsets, sacc_ends = [], []
    for onset, end in zip(test_edges[:-1], test_edges[1:]):
        if not test[onset]:
            continue  # Test condition not met for this chunk
        duration = ts[end] - ts[onset]
        if duration < min_duration:
            continue  # Duration criterion not met for saccade candidate
        sacc_onsets.append(onset)
        sacc_ends.append(end)

    # Populate the mask
    mask = np.zeros(ts.shape, dtype=bool)
    for onset, end in zip(sacc_onsets, sacc_ends):
        mask[onset:end+1] = True
    return mask


def filter_small_saccades(sacc: np.ndarray, ts: np.ndarray, min_dur_sec: float = 0.006) -> np.ndarray:
    """
    Discard small saccades from a given saccade detection time-series.

    :param sacc: A boolean time-series where True values denote saccades and False values denote fixations.
    :param ts: The sample times (in seconds) used to compute durations.
    :param min_dur_sec: Discard saccades at or below the specfied threshold.
        6 ms is the value used for microsaccades in the Engbert & Mergenthaler 2006 paper.
    """
    out = sacc.copy()
    edges = detect_bool_edges(sacc, include_endpoints=True)
    for onset, end in zip(edges[:-1], edges[1:]):
        if not sacc[onset]:
            continue
        duration = ts[end] - ts[onset]
        if duration <= min_dur_sec:
            out[onset:end+1] = False
    return out


def detect_blinks(
        pos: np.ndarray,
        fixation_mask: np.ndarray,
        dfac: float,
) -> np.ndarray:
    """
    Detect blinks based on fixation dispersion.
    See label.blinks.artifacts in: https://github.com/tmalsburg/saccades/blob/master/saccades/R/saccade_recognition.R

    :param pos: Nx2 array of gaze position. First column is x, second column is y.
    :param fixation_mask: A Boolean mask indicating fixations (i.e., not saccades).
    :param dfac: The factor by which the dispersion must exceed MAD(dispersion) to be a blink or artifact.
    :return: A Boolean mask indicated blinks. (This will be a subset of the fixation mask.)
    """
    # Determine fixation onsets and ends based on the Boolean mask
    fix_onsets, fix_ends = [], []
    edges = detect_bool_edges(fixation_mask, include_endpoints=True)
    for onset, end in zip(edges[:-1], edges[1:]):
        if fixation_mask[onset]:
            fix_onsets.append(onset)
            fix_ends.append(end)
    fix_onsets, fix_ends = np.array(fix_onsets), np.array(fix_ends)

    # Calculate fixation dispersion (i.e., MAD of x/y position)
    dispersion = np.zeros((fix_onsets.shape[0], 2))
    for i, (onset, end) in enumerate(zip(fix_onsets, fix_ends)):
        dispersion[i, :] = _median_absolute_deviation(pos[onset:end+1, :], axis=0)

    # Calculate the adaptive blink threshold
    lsd = np.log10(dispersion + 10e-8)
    median_lsd = np.nanmedian(lsd, axis=0)
    mad_lsd = _median_absolute_deviation(lsd, axis=0)
    blink_threshold = median_lsd - dfac * mad_lsd

    # Construct the Boolean mask
    blink_mask = np.zeros_like(fixation_mask)
    for onset, end, lsd_ in zip(fix_onsets, fix_ends, lsd):
        if np.all(lsd_ < blink_threshold):
            blink_mask[onset:end+1] = True
    return blink_mask


def _smooth_signal(x: np.ndarray, window_size: int) -> np.ndarray:
    """Smooth the given signal using a moving average."""
    window = np.ones(window_size, dtype=x.dtype) / window_size
    x_smooth = signal.convolve(x, window, mode='same')

    # Remove convolution artifacts from the signal edges by doing partial window averages
    half_window = window_size // 2
    for i in range(half_window):
        x_smooth[i] = x[:i+half_window].mean()
        j = x.shape[0] - i
        x_smooth[j-1] = x[j-half_window:].mean()

    return x_smooth


def _correct_small_std(msd: float, vel: np.ndarray, thresh: float = 1e-10) -> float:
    """Adjust the standard deviation estimator if the value is too small."""
    if msd < thresh:
        msd = np.sqrt(np.nanmean(np.square(vel))) - np.square(np.nanmean(vel))
        if msd < thresh:
            raise SaccadeDetectionException(f"msd less than {thresh:.1e}. Detection region too small.")
    return msd


def _median_absolute_deviation(x: np.ndarray, constant: float = 1.4826, axis: int = 0):
    """See https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/mad"""
    center = np.nanmedian(x, axis=axis, keepdims=True)
    abs_dev = np.abs(x - center)
    return constant * np.median(abs_dev, axis=axis)


def exclude_blink_saccades(
        blink_mask: np.ndarray,
        saccade_mask: np.ndarray,
        ts: np.ndarray,
        window_sec: float = 0.1,
) -> np.ndarray:
    """
    Create a Boolean mask that is True within 1) D ms of a blink event and 2) the duration of any saccades within D ms
    of a blink event. This will may help to exclude blink-related artifacts (including saccade-blink-saccade).

    :param blink_mask: A Boolean mask indicating blink events
    :param saccade_mask: A Boolean mask indicating saccade events
    :param ts: Time in seconds
    :param window_sec: The duration adjacent to saccade to be considered contaiminated by a blink event
        (D in the description above).
    :return: An updated blink mask that excludes adjacent contaminated data.
    """
    blink_edges = detect_bool_edges(blink_mask, include_endpoints=True)
    saccade_edges = detect_bool_edges(saccade_mask, include_endpoints=True)

    blink_edges_ts = [  # Find edges of blink events in time units
        (ts[start], ts[end])
        for start, end in zip(blink_edges[:-1], blink_edges[1:])
        if blink_mask[start]
    ]
    saccade_edges = [  # Find edges of saccade events in array indices
        (start, end)
        for start, end in zip(saccade_edges[:-1], saccade_edges[1:])
        if saccade_mask[start]
    ]

    # Construct the updated time-series mask
    mask = np.zeros(blink_mask.shape, dtype=bool)
    for blink_start, blink_end in blink_edges_ts:
        # Window begins as blink + D seconds in either direction
        window = (ts >= (blink_start - window_sec)) & (ts <= (blink_end + window_sec))
        for saccade_start, saccade_end in saccade_edges:
            # If the window overlaps with any saccade, extend the window to the saccade boundaries
            window[saccade_start:saccade_end] |= window[saccade_start:saccade_end].any()
        mask |= window  # Update the full time-series mask with the window
    return mask
