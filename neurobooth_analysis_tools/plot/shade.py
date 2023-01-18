"""
Utility functions for shading a time-series based on boolean/integer masks.
"""


import numpy as np
from typing import Optional, Dict
from matplotlib import pyplot as plt
from neurobooth_analysis_tools.preprocess.mask import detect_bool_edges, detect_int_edges


def shade_mask(ax: plt.Axes, mask: np.ndarray, ts: np.ndarray, plot_kws: Optional[Dict] = None) -> None:
    """Apply vertical shading to an axis based on a Boolean or ascending integer mask"""
    if plot_kws is None:
        plot_kws = {}

    if np.issubdtype(mask.dtype, np.bool_):
        _shade_bool_mask(ax, mask, ts, plot_kws)
    elif np.issubdtype(mask.dtype, np.int_):
        _shade_int_mask(ax, mask, ts, plot_kws)
    else:
        raise NotImplemented(f"shade_mask not implemented for mask arrays with dtype {mask.dtype}")


def _shade_bool_mask(ax: plt.Axes, mask: np.ndarray, ts: np.ndarray, plot_kws: Dict) -> None:
    """Subroutine for Boolean masks."""
    edges = detect_bool_edges(mask, include_endpoints=True)
    for begin, end in zip(edges[:-1], edges[1:]):
        if mask[begin]:
            ax.axvspan(ts[begin], ts[end], **plot_kws)


def _shade_int_mask(ax: plt.Axes, mask: np.ndarray, ts: np.ndarray, plot_kws: Dict) -> None:
    """Subroutine for ascending integer masks."""
    edges = detect_int_edges(mask, include_endpoints=True)
    for begin, end in zip(edges[:-1], edges[1:]):
        if mask[begin] > 0:
            ax.axvspan(ts[begin], ts[end], **plot_kws)