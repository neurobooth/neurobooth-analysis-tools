"""
Functions for loading Neurobooth data from .mov files.
"""

from typing import Union, Tuple
import numpy as np
import pandas as pd
import moviepy.editor as mp

import neurobooth_analysis_tools.data.hdf5 as hdf5


def load_iphone_audio(
        mov_file: hdf5.FILE_PATH,
        json_df: pd.DataFrame,
        hdf_df: pd.DataFrame,
        return_sync_indices: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]:
    """Load audio data from an iPhone .mov file.
    Dataframes from the synchronized HDF5 and JSON files are needed to infer timestamps and marker events.
    Audio is grouped into chunks based on JSON times. Times within each chunk are interpolated using LSL times.
    """
    sync_df = pd.merge(hdf_df, json_df, on='FrameNum', how='outer')
    json_time = sync_df['Time_JSON'].to_numpy()
    lsl_time = sync_df['Time_LSL'].to_numpy()
    if np.sum(np.isnan(json_time) | np.isnan(lsl_time)) > 0:
        raise hdf5.DataException("NaN values in synchronization columns")

    # Load audio from MOV
    mov_file = hdf5.resolve_filename(mov_file)
    with mp.VideoFileClip(mov_file) as clip:
        audio = clip.audio.to_soundarray()

    # Checks on loaded audio
    if audio.ndim != 2:
        raise NotImplementedError("Unexpected dimensionality of audio data. Only stereo audio currently supported.")
    n_samples, n_channels = audio.shape
    if n_channels != 2:
        raise NotImplementedError("Only stereo audio currently supported.")

    # Average stereo channels to get mono
    audio = audio.mean(axis=1)

    # Calculate audio sync indices corresponding to the spacing of json timestamps
    sync_idx = json_time - json_time[0]  # Start at 0
    sync_idx /= sync_idx[-1]  # End at 1
    sync_idx *= n_samples  # Convert 0..1 to indices 0..N
    sync_idx = np.round(sync_idx).astype(int)
    sync_idx[0], sync_idx[-1] = 0, n_samples  # Ensure proper first and last indices

    if np.sum(np.diff(sync_idx) <= 0):  # Ensure sync index spacing makes logical sense
        raise hdf5.DataException("Invalid audio sync indices... check json timing.")

    # The sync indices represent the boundaries of audio chunks. Linspace times within each chunk.
    audio_ts_lsl = np.zeros(audio.shape[0])
    audio_ts_json = np.zeros_like(audio_ts_lsl)
    for i, (start_idx, end_idx) in enumerate(zip(sync_idx[:-1], sync_idx[1:])):
        n = end_idx - start_idx
        audio_ts_lsl[start_idx:end_idx] = np.linspace(lsl_time[i], lsl_time[i+1], n, endpoint=False)
        audio_ts_json[start_idx:end_idx] = np.linspace(json_time[i], json_time[i+1], n, endpoint=False)

    # Package into DataFrame
    df = pd.DataFrame.from_dict({
        'Amplitude': audio,
        'Time_JSON': audio_ts_json,
        'Time_LSL': audio_ts_lsl,
    })

    if return_sync_indices:
        return df, sync_idx
    else:
        return df


def load_iphone_audio_uniform(
        mov_file: hdf5.FILE_PATH,
        hdf_df: pd.DataFrame,
        exclude_beginning: bool = True,
) -> pd.DataFrame:
    """Load audio data from an iPhone .mov file.
    Data from a synchronized HDF5 file is needed to infer timestamps and marker events.
    Assumes uniform spacing of audio times based on "first" and "last" LSL timestamps.
    """
    video_ts = hdf_df['Time_LSL'].to_numpy()

    # Load audio from MOV
    mov_file = hdf5.resolve_filename(mov_file)
    with mp.VideoFileClip(mov_file) as clip:
        audio = clip.audio.to_soundarray()
        audio_sample_rate = clip.audio.fps

    # Checks on loaded audio
    if audio.ndim != 2:
        raise NotImplementedError("Unexpected dimensionality of audio data. Only stereo audio currently supported.")
    n_samples, n_channels = audio.shape
    if n_channels != 2:
        raise NotImplementedError("Only stereo audio currently supported.")

    # Average stereo channels to get mono
    audio = audio.mean(axis=1)

    if exclude_beginning:  # Discard beginning of the video time-series as the sampling rate/data are untrustworthy
        video_start_idx = hdf5.find_idx_stable_sample_rate(video_ts)
        start_time = video_ts[video_start_idx]
        end_time = video_ts[-1]
    else:
        start_time = video_ts[0]
        end_time = video_ts[-1]

    # Discard a similar amount of audio data
    duration = end_time - start_time
    audio_start_idx = audio.shape[0] - int(round(duration * audio_sample_rate))
    audio = audio[audio_start_idx:]

    # Interpolate audio timestamps based on the video start and end time-stamps
    audio_ts = np.linspace(start_time, end_time, audio.shape[0])

    # Package into DataFrame
    df = pd.DataFrame.from_dict({
        'Amplitude': audio,
        'Time_LSL': audio_ts,
    })
    return df
