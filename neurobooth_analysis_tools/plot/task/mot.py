"""
Tools for visualizing marker position and mouse/eye position during the multiple object tracking (MOT) task.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Any

from neurobooth_analysis_tools.task.mot import MOTTrial


# Define where the task region is on the screen
X_RANGE = (710., 1210.)
Y_RANGE = (290., 790.)
SCREEN_RATIO = (X_RANGE[1]-X_RANGE[0]) / (Y_RANGE[1]-Y_RANGE[0])


def make_trial_grid(plot_size: float = 4, titles: bool = False) -> (plt.Figure, List[plt.Axes]):
    plot_height = plot_size
    plot_width = plot_size * SCREEN_RATIO

    fig, axs = plt.subplots(5, 4, figsize=(4*plot_height, 5*plot_width))
    return fig, axs.flatten()


def plot_marker_animation(
        ax: plt.Axes,
        marker_data: MOTTrial,
        gaze_pos: Optional[pd.DataFrame] = None,
        target_color: Any = '#55a868',
        nontarget_color: Any = '#ccb974',
        path_linewidth: float = 1.5,
        endpoint_marker_area: float = 25,
        gaze_linewidth: float = 0.5,
        margin: float = 20,
) -> None:
    circle_ids = marker_data.circle_paths['MarkerTgt'].unique()
    for cid in sorted(circle_ids):
        color = target_color if cid < marker_data.n_targets else nontarget_color
        paths = marker_data.circle_paths.loc[marker_data.circle_paths['MarkerTgt'] == cid]
        ax.plot(paths['MarkerX'], paths['MarkerY'], color=color, linewidth=path_linewidth)
        ax.scatter(paths['MarkerX'].iloc[-1], paths['MarkerY'].iloc[-1], color=color, s=endpoint_marker_area)

    if gaze_pos is not None:
        gaze_pos = gaze_pos.loc[gaze_pos['Time_LSL'] <= marker_data.animation_end_time]
        _plot_gaze(ax, gaze_pos, gaze_linewidth)

    ax.set_title(f"{'Practice: ' if marker_data.practice else ''}{marker_data.n_targets} dots")
    _configure_trial_plot_axes(ax, margin)


def plot_clicks(
        ax: plt.Axes,
        marker_data: MOTTrial,
        gaze_pos: Optional[pd.DataFrame] = None,
        mouse_pos: Optional[pd.DataFrame] = None,
        target_color: Any = '#55a868',
        nontarget_color: Any = '#ccb974',
        endpoint_marker_area: float = 25,
        gaze_linewidth: float = 0.5,
        mouse_linewidth: float = 0.5,
        margin: float = 20,
) -> None:
    circle_ids = marker_data.circle_paths['MarkerTgt'].unique()
    for cid in sorted(circle_ids):
        color = target_color if cid < marker_data.n_targets else nontarget_color
        paths = marker_data.circle_paths.loc[marker_data.circle_paths['MarkerTgt'] == cid]
        ax.scatter(paths['MarkerX'].iloc[-1], paths['MarkerY'].iloc[-1], color=color, s=endpoint_marker_area)

    if gaze_pos is not None:
        gaze_pos = gaze_pos.loc[gaze_pos['Time_LSL'] > marker_data.animation_end_time]
        _plot_gaze(ax, gaze_pos, gaze_linewidth)

    # TODO: Mouse
    # TODO: Denote clicks

    ax.set_title(f"{'Practice: ' if marker_data.practice else ''}{marker_data.n_targets} dots")
    _configure_trial_plot_axes(ax, margin)


def _configure_trial_plot_axes(ax: plt.Axes, margin: float) -> None:
    ax.set_xticks(np.linspace(*X_RANGE, 5))
    ax.set_yticks(np.linspace(*X_RANGE, 5))
    ax.set_xlim([X_RANGE[0] - margin, X_RANGE[1] + margin])
    ax.set_ylim([Y_RANGE[0] - margin, Y_RANGE[1] + margin])


def _plot_gaze(ax: plt.Axes, gaze_pos: pd.DataFrame, linewidth: float) -> None:
    no_blink = (gaze_pos['R_GazeX'] > 0) | (gaze_pos['R_GazeY'] > 0)
    ax.plot(
        gaze_pos.loc[no_blink, 'R_GazeX'],
        gaze_pos.loc[no_blink, 'R_GazeY'],
        color='k', linewidth=linewidth
    )
    no_blink = (gaze_pos['L_GazeX'] > 0) | (gaze_pos['L_GazeY'] > 0)
    ax.plot(
        gaze_pos.loc[no_blink, 'L_GazeX'],
        gaze_pos.loc[no_blink, 'L_GazeY'],
        color='gray', linewidth=linewidth
    )
