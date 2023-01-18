"""
Functions for working with Boolean and ascending integer masks.
"""


import numpy as np


def detect_edges(mask: np.ndarray, include_endpoints=False) -> np.ndarray:
    """Convenience function to select an appropriate implementation based on the mask data type"""
    if np.issubdtype(mask.dtype, np.bool_):
        return detect_bool_edges(mask, include_endpoints=include_endpoints)
    elif np.issubdtype(mask.dtype, np.int_):
        return detect_int_edges(mask, include_endpoints=include_endpoints)
    else:
        raise NotImplemented(f"detect_edges not implemented for mask arrays with dtype {mask.dtype}")


def detect_bool_edges(mask: np.ndarray, include_endpoints=False) -> np.ndarray:
    """Detect edges in a Boolean mask and return edge indices"""
    if include_endpoints:
        a = np.r_[~mask[0], mask]
        a[-1] = ~a[-2]
    else:
        a = np.r_[mask[0], mask]

    return np.where(np.diff(a))[0]


def detect_int_edges(mask: np.ndarray, include_endpoints=False) -> np.ndarray:
    """Detect edges in an ascending integer mask and return edge indices"""
    if include_endpoints:
        a = np.r_[-1, mask]
        a[-1] = -1
    else:
        a = np.r_[0, mask]

    return np.where(np.diff(a) != 0)[0]
