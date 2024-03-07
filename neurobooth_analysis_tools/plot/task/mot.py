"""
Tools for visualizing marker position and mouse/eye position during the multiple object tracking (MOT) task.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt
from typing import List, Optional, Any, Literal

from neurobooth_analysis_tools.task.mot import MOTTrial
from neurobooth_analysis_tools.preprocess.gaze.event import exclude_blink_saccades


# Define where the task region is on the screen
X_RANGE = (710., 1210.)
Y_RANGE = (290., 790.)
SCREEN_RATIO = (X_RANGE[1]-X_RANGE[0]) / (Y_RANGE[1]-Y_RANGE[0])


# Low-pass filter for gaze data
GAZE_SOS_COEF = butter(7, 100 / (1000 / 2), btype='lowpass', output='sos')


def make_trial_grid(plot_size: float = 4) -> (plt.Figure, List[plt.Axes]):
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
    _plot_marker_trajectories(ax, marker_data, path_linewidth, endpoint_marker_area, target_color, nontarget_color)

    if gaze_pos is not None:
        gaze_pos = gaze_pos.loc[
            (gaze_pos['Time_LSL'] >= marker_data.start_time) &
            (gaze_pos['Time_LSL'] <= marker_data.animation_end_time)
        ]
        _plot_gaze(ax, 'R', gaze_pos, gaze_linewidth)
        _plot_gaze(ax, 'L', gaze_pos, gaze_linewidth)

    ax.legend()
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
    _plot_marker_trajectories(ax, marker_data, None, endpoint_marker_area, target_color, nontarget_color)

    if gaze_pos is not None:
        gaze_pos = gaze_pos.loc[
            (gaze_pos['Time_LSL'] >= marker_data.animation_end_time) &
            (gaze_pos['Time_LSL'] <= marker_data.end_time)
        ]
        _plot_gaze(ax, 'R', gaze_pos, gaze_linewidth)
        _plot_gaze(ax, 'L', gaze_pos, gaze_linewidth)

    if mouse_pos is not None:
        mouse_pos = mouse_pos.loc[
            (mouse_pos['Time_LSL'] >= marker_data.animation_end_time) &
            (mouse_pos['Time_LSL'] <= marker_data.end_time)
            ]

        ax.plot(mouse_pos['PosX'], mouse_pos['PosY'], color='b', linewidth=mouse_linewidth, label='Mouse')
        click_mask = mouse_pos['MouseState'] == 'Click'
        ax.scatter(
            mouse_pos.loc[click_mask, 'PosX'], mouse_pos.loc[click_mask, 'PosY'],
            s=(endpoint_marker_area / 4), color='b',
        )

    ax.legend()
    _configure_trial_plot_axes(ax, margin)


def _configure_trial_plot_axes(ax: plt.Axes, margin: float) -> None:
    ax.set_xticks(np.linspace(*X_RANGE, 5))
    ax.set_yticks(np.linspace(*Y_RANGE, 5))
    ax.set_xlim([X_RANGE[0] - margin, X_RANGE[1] + margin])
    ax.set_ylim([Y_RANGE[0] - margin, Y_RANGE[1] + margin])


def _plot_marker_trajectories(
        ax: plt.Axes,
        marker_data: MOTTrial,
        path_linewidth: Optional[float] = 1.5,
        endpoint_marker_area: float = 25,
        target_color: Any = '#55a868',
        nontarget_color: Any = '#ccb974',
        set_title: bool = True,
) -> None:
    circle_ids = marker_data.circle_paths['MarkerTgt'].unique()
    for cid in sorted(circle_ids):
        color = target_color if cid < marker_data.n_targets else nontarget_color
        paths = marker_data.circle_paths.loc[marker_data.circle_paths['MarkerTgt'] == cid]
        if path_linewidth is not None:
            ax.plot(paths['MarkerX'], paths['MarkerY'], color=color, linewidth=path_linewidth)
        ax.scatter(
            paths['MarkerX'].iloc[-1], paths['MarkerY'].iloc[-1],
            color=color, s=endpoint_marker_area,
        )

    if set_title:
        ax.set_title(f"{'Practice: ' if marker_data.practice else ''}{marker_data.n_targets} dots")


def _plot_gaze(ax: plt.Axes, eye: Literal['L', 'R'], gaze_pos: pd.DataFrame, linewidth: float) -> None:
    if f'IFlag_{eye}_Blink' in gaze_pos.columns:
        blink = gaze_pos[f'IFlag_{eye}_Blink'].to_numpy() > 0
        saccade = gaze_pos[f'IFlag_{eye}_Saccade'].to_numpy() > 0
        ts = gaze_pos['Time_EDF'].to_numpy() / 1e3
        blink = exclude_blink_saccades(blink, saccade, ts)
    else:
        blink = None

    x, y = _exclude_blinks(gaze_pos[f'{eye}_GazeX'].to_numpy(), gaze_pos[f'{eye}_GazeY'].to_numpy(), blink)
    color = 'k' if eye == 'R' else 'gray'
    ax.plot(x, y, color=color, alpha=0.7, linewidth=linewidth, label=f'{eye} Gaze')


def _exclude_blinks(
        x: np.ndarray, y: np.ndarray, blink: Optional[np.ndarray], filter: bool = True,
) -> (np.ndarray, np.ndarray):
    if blink is None:
        no_blink = (x > 0) | (y > 0)
    else:
        no_blink = ~blink

    if filter:
        x = sosfiltfilt(GAZE_SOS_COEF, x, axis=0)
        y = sosfiltfilt(GAZE_SOS_COEF, y, axis=0)

    return x[no_blink], y[no_blink]
