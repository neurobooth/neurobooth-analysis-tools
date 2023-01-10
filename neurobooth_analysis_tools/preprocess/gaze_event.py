"""
An algorithm for detection of gaze events (i.e., saccades and fixations).

Reference code: https://github.com/richardschweitzer/OnlineSaccadeDetection
The R version has an implementation of the Engbert-Kliegl algorithm with endpoint refinement based on post-saccadic
oscillations.
PSO refinement not currently implemented (as it may make it easier to identify dysmetric saccades in our analyses).
"""

import numpy as np
from typing import NamedTuple
import scipy.signal as signal
from neurobooth_analysis_tools.preprocess.mask import detect_bool_edges


class SaccadeResult(NamedTuple):
    onset: np.ndarray
    end: np.ndarray
    radius: np.ndarray  # Parameters of elliptic threshold


class SaccadeDetectionException(Exception):
    def __init__(self, *args):
        super(SaccadeDetectionException, self).__init__(*args)


def detect_saccades(
        pos: np.ndarray,
        ts: np.ndarray,
        vfac: float = 5,
        min_duration: float = 0.006,
        window_size: int = 5,
        resample: bool = False,
) -> SaccadeResult:
    """
    Engbert-Kliegl (2003) algorithm for saccade detection.

    :param pos: Nx2 array of gaze position. First column is x, second column is y.
    :param ts: Timestamp of each sample (in seconds).
    :param vfac: Relative velocity thresholds; lambda in the paper.
    :param min_duration: Minimum saccade duration (in seconds).
    :param window_size: Width of the moving average window used for velocity smoothing.
    :param resample: Whether to resample the position time-series before performing detection.
    :return: Indices of saccade onsets and ends, alongside the elliptic threshold.
    """
    if resample:
        pos, ts = signal.resample(pos, pos.shape[0], ts, axis=0)

    # Calculate and smooth velocity
    vel = np.gradient(pos, ts, axis=0)
    vel[:, 0] = _smooth_signal(vel[:, 0], window_size)
    vel[:, 1] = _smooth_signal(vel[:, 1], window_size)

    # Calculate median estimator for std and use it to find the detection radius
    vel_median = np.nanmedian(vel, axis=0, keepdims=True)
    msd = np.sqrt(np.nanmedian(np.square(vel - vel_median), axis=0))
    msd[0] = _correct_small_std(msd[0], vel[:, 0])
    msd[1] = _correct_small_std(msd[1], vel[:, 1])
    radius = vfac * msd

    # Determine when the velocity greater than the detection radius
    test = np.sum(np.square(vel / radius[np.newaxis, :]), axis=1)
    test = (test > 1) & ~np.isnan(test)
    test_edges = detect_bool_edges(test, include_endpoints=True)

    # Create a list of saccade onset and end indices from the identified candidate saccades
    onsets, ends = [], []
    for onset, end in zip(test_edges[:-1], test_edges[1:]):
        if not test[onset]:
            continue  # Test condition not met for this chunk
        duration = ts[end] - ts[onset]
        if duration < min_duration:
            continue  # Duration criterion not met for saccade candidate
        onsets.append(onset)
        ends.append(end)

    return SaccadeResult(
        onset=np.array(onsets),
        end=np.array(ends),
        radius=radius,
    )


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
