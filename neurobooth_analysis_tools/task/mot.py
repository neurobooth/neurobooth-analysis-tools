"""
Task-specific processing for multiple object tracking (MOT).
"""

import re
from typing import NamedTuple, List
import numpy as np
import pandas as pd
from neurobooth_analysis_tools.data import hdf5


# Regex Patterns for extracting information from the marker time-series
TRIAL_START = re.compile(r'(.*)Trial_start_(.*)')
TRIAL_END = re.compile(r'(.*)Trial_end_(.*)')
N_TARGET = re.compile(r'number targets:(\d+)_(.*)')
CLICK = re.compile(r'Response_start_(.*)')


class MOTTrial(NamedTuple):
    """Structured representation of marker information for an MOT Trial"""
    practice: bool
    start_time: float
    animation_end_time: float
    end_time: float
    n_targets: int
    circle_paths: pd.DataFrame
    click_times: np.ndarray


class ParserError(Exception):
    pass


def parse_markers(marker: hdf5.DataGroup) -> List[MOTTrial]:
    """Parse the marker time-series and return structured information for each MOT trial."""
    markers = marker.time_series
    timestamps = marker.time_stamps

    # Identify the start and end of each trial
    trial_start_idx = []
    trial_end_idx = []
    for i, marker in enumerate(markers):
        if re.match(TRIAL_START, marker) is not None:
            trial_start_idx.append(i)
        elif re.match(TRIAL_END, marker) is not None:
            trial_end_idx.append(i)
    trial_start_idx, trial_end_idx = np.array(trial_start_idx), np.array(trial_end_idx)

    # Validity Checks
    if len(trial_start_idx) != len(trial_end_idx):
        raise ParserError(
            f'Mismatch between the number of trial start markers ({len(trial_start_idx)})'
            f' and trial end markers ({len(trial_end_idx)})'
        )
    elif (trial_start_idx >= trial_end_idx).sum() > 0:
        raise ParserError('Detected trial start occurring after trial end!')

    # Parse each trial
    return [
        _parse_markers_trial(markers[start_idx:(end_idx+1)], timestamps[start_idx:(end_idx+1)])
        for start_idx, end_idx in zip(trial_start_idx, trial_end_idx)
    ]


def _parse_markers_trial(markers: np.ndarray, timestamps: np.ndarray) -> MOTTrial:
    """
    Parse a set of marker strings for a single trial.
    The general structure of the marker time-series for a trial should be:
    ...
    Trial_start (or PracticeTrial_start)
    number targets
    !V TARGET_POS (for C circles, we get C of these every animation update.)
    ...
    !V TARGET_POS
    Response_start (for each click)
    Trial_end (or PracticeTrial_end)
    ...

    :param markers: The marker strings for a single trial.
    :param timestamps: The associated LSL timestamps.
    :return: A structured representation of the marker information for the trial.
    """
    practice = None
    start_time = None
    end_time = None
    n_targets = None
    circle_id = []
    circle_x = []
    circle_y = []
    circle_ts = []
    click_times = []

    for marker, ts in zip(markers, timestamps):
        # Detect trial start
        match = re.match(TRIAL_START, marker)
        if match is not None:
            practice = 'practice' in match[1].lower()
            start_time = ts
            continue

        # Detect number of targets marker
        match = re.match(N_TARGET, marker)
        if match is not None:
            n_targets = int(match[1])
            continue

        # Detect marker position updates
        match = re.match(hdf5.MARKER_POS_PATTERN, marker)
        if match is not None:
            circle_id.append(int(match[1][1:]))
            circle_x.append(int(match[2]))
            circle_y.append(int(match[3]))
            circle_ts.append(ts)
            continue

        # Detect clicks
        match = re.match(CLICK, marker)
        if match is not None:
            click_times.append(ts)
            continue

        # Detect trial end
        match = re.match(TRIAL_END, marker)
        if match is not None:
            end_time = ts
            break

    return MOTTrial(
        practice=practice,
        start_time=start_time,
        animation_end_time=max([start_time, *circle_ts]),
        end_time=end_time,
        n_targets=n_targets,
        circle_paths=pd.DataFrame.from_dict({
            'MarkerTgt': circle_id,
            'MarkerX': circle_x,
            'MarkerY': circle_y,
            'Time_LSL': circle_ts,
        }),
        click_times=np.array(click_times),
    )
