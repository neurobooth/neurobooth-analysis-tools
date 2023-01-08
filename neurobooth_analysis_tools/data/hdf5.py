"""
Functions for loading Neurobooth data from .hdf5 files.
Some files only provide index and time information and must be synced with an additional source file (e.g., movies).
"""

import os
import re
from h5io import read_hdf5
import pandas as pd
import numpy as np
from functools import partial
from scipy.signal import convolve
from typing import NamedTuple, Dict, Union, Tuple

from neurobooth_analysis_tools.data.files import FileMetadata


class DataException(Exception):
    """Exception for data-related errors."""
    def __init__(self, *args):
        super(DataException, self).__init__(*args)


class DataGroup(NamedTuple):
    """Common substructure presence for 'device_data' and 'marker' in Neurobooth HDF5 files."""
    info: Dict
    footer: Dict
    time_series: np.ndarray
    time_stamps: np.ndarray


class Device(NamedTuple):
    """Represents the structure of a Neurobooth HDF5 file."""
    data: DataGroup
    marker: DataGroup


def load_neurobooth_file(file: Union[str, FileMetadata]) -> Device:
    """Load a neurobooth file and return its contents in a structured form."""
    if isinstance(file, FileMetadata):
        path = os.path.join(file.session_path, file.file_name)
    elif isinstance(file, str):
        path = file
    else:
        raise ValueError("Unsupported argument type.")

    data = read_hdf5(path)
    if not isinstance(data, dict):
        raise DataException(f"Unexpected object type in HDF5 file: {type(data)}")

    return Device(
        data=_extract_data_group(data['device_data']),
        marker=_extract_data_group(data['marker'], flatten_time_series=True),
    )


def _extract_data_group(group: Dict, flatten_time_series: bool = False) -> DataGroup:
    """Utility function for structuring 'device_data' or 'marker' in a Neurobooth HDF5 file."""
    time_series = group['time_series']
    if flatten_time_series:
        time_series = np.array(time_series).flatten()

    return DataGroup(
        info=group['info'],
        footer=group['footer'],
        time_series=time_series,
        time_stamps=group['time_stamps'],
    )


_MARKER_POS_PATTERN = re.compile(r'!V TARGET_POS target (\d+), (\d+) .*')


def extract_marker_position(device: Device) -> pd.DataFrame:
    """Marker data should be identical for each device."""
    marker = device.marker

    x, y, t = [], [], []
    for text, ts in zip(marker.time_series, marker.time_stamps):
        match = re.match(_MARKER_POS_PATTERN, text)
        if match is not None:
            x.append(int(match[1]))
            y.append(int(match[2]))
            t.append(ts)

    return pd.DataFrame.from_dict({
        'MarkerX': x,
        'MarkerY': y,
        'Time_LSL': t,
    })


def extract_marker_event_time(device: Device, event_prefix: str) -> np.ndarray:
    """Extract the time(s) of an event in the marker time-series that starts with the specified prefix"""
    marker = device.marker
    mask = np.char.startswith(marker.time_series, event_prefix)
    return marker.time_stamps[mask]


def extract_event_boundaries(device: Device, start_prefix: str, end_prefix: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the times corresponding to the start and end of an event (based on the specified event prefixes)"""
    start = extract_marker_event_time(device, start_prefix)
    end = extract_marker_event_time(device, end_prefix)

    n_start, n_end = start.shape[0], end.shape[0]
    if n_start == 0 or n_end == 0:
        raise DataException("Event boundaries could not be found.")
    if n_start != n_end:
        raise DataException(f"Mismatched event boundaries! #({start_prefix})={n_start}, #({end_prefix})={n_end}.")
    if np.sum((end - start) < 0) > 0:
        raise DataException("Start of event occurred before end of event!")

    return start, end


extract_task_boundaries = partial(extract_event_boundaries, start_prefix='Task_start', end_prefix='Task_end')
extract_instruction_boundaries = partial(
    extract_event_boundaries,
    start_prefix='Intructions_start',
    end_prefix='Intructions_end'
)


def create_task_mask(
        device: Device,
        time_stamps: np.ndarray,
        allow_multiple: bool = True,
) -> np.ndarray:
    """Create a Boolean mask for the given timestamps indicating task performance."""
    starts, ends = extract_task_boundaries(device)
    if not allow_multiple and starts.shape[0] > 1:
        raise DataException("Multiple task periods detected.")

    mask = np.zeros(time_stamps.shape, dtype=bool)
    for start, end in zip(starts, ends):
        mask[(time_stamps >= start) & (time_stamps <= end)] = True
    return mask


def create_instruction_mask(
        device: Device,
        time_stamps: np.ndarray,
        allow_multiple: bool = True,
) -> np.ndarray:
    """Create a Boolean mask for the given timestamps indicating the instruction delivery period."""
    starts, ends = extract_instruction_boundaries(device)
    if not allow_multiple and starts.shape[0] > 1:
        raise DataException("Multiple instruction periods detected.")

    mask = np.zeros(time_stamps.shape, dtype=bool)
    for start, end in zip(starts, ends):
        mask[(time_stamps >= start) & (time_stamps <= end)] = True
    return mask


def extract_eyelink(device: Device) -> pd.DataFrame:
    """
    Extract a DataFrame for the time-series present in an EyeLink device file.
    Column order is currently hard-coded due to misleading column headers.
    See: https://docs.google.com/document/d/1tx_DxtfMbe1mL72kOw1o9Pog7LSJuvxBRfo2hM_8bBw/
    """
    df = pd.DataFrame(
        data=device.data.time_series,
        columns=(
            'R_GazeX', 'R_GazeY', 'R_PupilSize',
            'L_GazeX', 'L_GazeY', 'L_PupilSize',
            'Target_Distance', 'Target_PositionX', 'Target_PositionY',
            'R_PPD', 'L_PPD',
            'Time_EDF', 'Time_NUC'
        )
    )
    df['Time_LSL'] = device.data.time_stamps
    df['Flag_Instructions'] = create_instruction_mask(device, device.data.time_stamps)
    df['Flag_Task'] = create_task_mask(device, device.data.time_stamps)

    return df


def extract_yeti(device: Device) -> pd.DataFrame:
    """Extract a DataFrame for the time-series present in a Yeti mic device file."""
    # Data is not a continuous stream. Instead, it is a matrix of "chunks".
    # Time needs to be interpolated within each chunk.
    chunks = device.data.time_series
    chunk_ts = device.data.time_stamps
    n_chunks, chunk_size = chunks.shape

    # If odd, the first column of each chunk is the elapsed time since the prior chunk.
    # Discard it and keep the amplitude data.
    if chunk_size % 2:
        chunk_size -= 1
        chunks = chunks[:, 1:]

    # Add an extra timestamp to the start to assist interpolation
    # In the future, consider leveraging the first discarded time-delta above
    avg_spacing = np.diff(chunk_ts).mean()
    chunk_ts = np.r_[chunk_ts[0] - avg_spacing, chunk_ts]

    # Interpolate times within each chunk
    # Reference example of 2D linspace: np.linspace(np.arange(10), np.arange(10)+1, 5, endpoint=False, axis=1)
    ts, steps = np.linspace(chunk_ts[:-1], chunk_ts[1:], chunk_size, endpoint=False, axis=1, retstep=True)

    # Adjust timestamps so that the *last* element of each chunk has the timestamp for the chunk.
    ts += steps[:, np.newaxis]

    # Flatten chunks into a 1D array
    audio = chunks.flatten()
    device_ts = ts.flatten()

    # Package into DataFrame
    df = pd.DataFrame.from_dict({
        'Amplitude': audio,
        'Time_LSL': device_ts
    })
    df['Flag_Instructions'] = create_instruction_mask(device, device_ts)
    df['Flag_Task'] = create_task_mask(device, device_ts)

    return df


def extract_mean_video_rgb(device: Device, exclude_beginning: bool = False) -> pd.DataFrame:
    """Extract a DataFrame representing mean color channels in a processed video time stream."""
    df = pd.DataFrame(
        data=device.data.time_series,
        columns=('MeanColor_R', 'MeanColor_G', 'MeanColor_B',)
    )
    df['Time_LSL'] = device.data.time_stamps

    if exclude_beginning:
        # Discard the beginning of the time-series where the sampling rate/data are untrustworthy
        start_idx = find_idx_stable_sample_rate(device.data.time_stamps)
        df = df.iloc[start_idx:]

    df['Flag_Instructions'] = create_instruction_mask(device, df['Time_LSL'].to_numpy())
    df['Flag_Task'] = create_task_mask(device, df['Time_LSL'].to_numpy())

    return df


def find_idx_stable_sample_rate(ts: np.ndarray, eps_hz: float = 1.0, window_width_sec: int = 1) -> int:
    """Determine when the sampling rate has stabilized to the mean.
    This method is used to trim the beginning off a time-series when data are unreliable.
    """
    time_diff = np.diff(ts)
    n_sample = int(round(1 / time_diff.mean()) * window_width_sec)
    if n_sample % 2 == 0:  # Want odd window for convolution
        n_sample += 1

    # Compute centered moving average via convolution with a uniform window that sums to 1
    # Some artifacts will exist for (n_sample / 2) samples at the endpoints due to zero-padding
    window = np.ones(n_sample) / n_sample
    mvg_avg = convolve(time_diff, window, mode='same')

    # Find when the instantaneous sample rate is within eps from the mean sample rate
    within_band = np.abs((1 / time_diff.mean()) - (1 / mvg_avg)) < eps_hz
    first = np.flatnonzero(within_band)[0]

    return first + 1  # Account for the loss of a sample due to np.diff
