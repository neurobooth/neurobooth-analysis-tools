"""
Functions for working with data windows.
"""


import numpy as np


def calc_num_windows(n_samples: int, window_length: int) -> int:
    """
    Calculate the number of non-overlapping windows for a given window length and number of samples.
    Avoids the use of floating-point arithmetic (i.e., ceil(n_samples / window_length)).
    """
    n_windows, remainder = divmod(n_samples, window_length)
    if remainder > 0:
        return n_windows + 1
    else:  # Number of samples is an exact multiple of the window length
        return n_windows


def make_windows_1d(x: np.ndarray, window_length: int) -> np.ndarray:
    """
    Segment a 1D time-series into non-overlapping windows. The last window will be zero-padded.
    """
    if x.ndim > 1:
        raise ValueError("make_windows_1d expects a one-dimensional array.")
    n_samples = x.shape[0]
    n_windows = calc_num_windows(n_samples, window_length)
    n_pad = (n_windows * window_length) - n_samples
    x = np.pad(x, (0, n_pad), mode='constant', constant_values=0)
    return x.reshape((n_windows, -1))


def zero_pad_axis(x: np.ndarray, pad_length: int, axis: int = 0) -> np.ndarray:
    """Zero-pad the end of an array along only one axis."""
    pad = [(0, 0) for _ in range(x.ndim)]
    pad[axis] = (0, pad_length)
    return np.pad(x, pad, mode='constant', constant_values=0)


def make_overlap_windows(x: np.ndarray, window_length: int, window_hop: int) -> np.array:
    """
    A slightly more flexible function than make_windows_1d.
    Vectorized code to allow the creation of overlapping sliding windows along the FIRST (i.e., 0) axis.
    If windows are desired along a different axis, the caller can first reshape the array (e.g., np.moveaxis).
    The last window will be zero-padded.
    """
    N = x.shape[0]
    if window_length > N:  # Edge case: the signal is shorter than the window
        return zero_pad_axis(x, window_length - N)[np.newaxis, :]

    # Pad the signal to the necessary length
    pad_target = min(window_hop, window_length)  # The smaller value determines how many values need to be padded
    _, n_over = divmod(N, pad_target)  # Number of samples beyond the last full window
    n_pad = pad_target - n_over
    if n_pad > 0:
        x = zero_pad_axis(x, n_pad, axis=0)
        N = x.shape[0]

    # Create a 2D matrix of window indices for a vectorized view into the given padded array.
    max_start = N - window_length + 1
    idx = np.arange(0, max_start, window_hop)[:, np.newaxis] + np.arange(window_length)[np.newaxis, :]
    return x[idx]
