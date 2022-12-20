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
