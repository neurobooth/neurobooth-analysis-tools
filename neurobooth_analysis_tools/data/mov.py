"""
Functions for loading Neurobooth data from .mov files.
"""

from typing import Union, Tuple
import numpy as np
import pandas as pd
import moviepy.editor as mp

from neurobooth_analysis_tools.data.hdf5 import FILE_PATH, resolve_filename, find_idx_stable_sample_rate
from neurobooth_analysis_tools.data.json import IPhoneJsonResult
from neurobooth_analysis_tools.data.types import DataException


def load_iphone_audio(
        mov_file: FILE_PATH,
        json_data: IPhoneJsonResult,
        hdf_df: pd.DataFrame,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, np.ndarray]]:
    """Load audio data from an iPhone .mov file.
    Use the audio information in the JSON file to synchronize to the LSL times in the HDF file.
    """
    if json_data.audio.shape[0] == 0:
        raise DataException("Audio buffer information not present in JSON. Use load_iphone_audio_uniform.")
    if hdf_df['FrameNum'].iloc[0] == 1:
        raise DataException("HDF frames should start from 0 for recent App versions compatible with this sync.")

    audio, _ = load_audio(mov_file)
    n_samples_mov, _ = audio.shape
    audio = audio.mean(axis=1)  # Average stereo channels to get mono

    # Reconstruct the cumulative time-series index and durations from the JSON audio samples (batches)
    sample_counts = json_data.audio['SampleCount'].to_numpy()
    sample_counts_cum = np.cumsum(sample_counts)
    sample_counts_cum = np.r_[0, sample_counts_cum]
    sample_dur_cum = np.cumsum(json_data.audio['SampleDuration'].to_numpy())
    sample_dur_cum = np.r_[0, sample_dur_cum]

    n_samples_json = sample_counts_cum[-1]
    if n_samples_json > n_samples_mov:
        raise DataException("More audio samples in JSON than MOV")
    audio = audio[-n_samples_json:]  # Trim appropriate number of samples from start of MOV audio

    # Construct a relative time-series for full JSON audio
    relative_time = np.zeros(n_samples_json)
    for i in range(len(sample_counts)):
        start_idx, end_idx = sample_counts_cum[i], sample_counts_cum[i+1]
        start_time, end_time = sample_dur_cum[i], sample_dur_cum[i+1]

        relative_time[start_idx:end_idx] = np.linspace(start_time, end_time, sample_counts[i], endpoint=False)

    # Current strategy: Sync to last video frame. (as buffer may not be flushed at start)
    relative_time -= relative_time[-1]  # Count up to zero
    audio_ts_json = hdf_df['Time_iPhone'].iloc[-1] + relative_time
    audio_ts_lsl = hdf_df['Time_LSL'].iloc[-1] + relative_time

    return pd.DataFrame.from_dict({
        'Amplitude': audio,
        'Time_JSON': audio_ts_json,
        'Time_LSL': audio_ts_lsl,
    })


def load_iphone_audio_uniform(
        mov_file: FILE_PATH,
        hdf_df: pd.DataFrame,
        exclude_beginning: bool = True,
) -> pd.DataFrame:
    """Load audio data from an iPhone .mov file.
    Data from a synchronized HDF5 file is needed to infer timestamps and marker events.
    Assumes uniform spacing of audio times based on "first" and "last" LSL timestamps.
    """
    video_ts = hdf_df['Time_LSL'].to_numpy()

    audio, audio_sample_rate = load_audio(mov_file)
    audio = audio.mean(axis=1)  # Average stereo channels to get mono

    if exclude_beginning:  # Discard beginning of the video time-series as the sampling rate/data are untrustworthy
        video_start_idx = find_idx_stable_sample_rate(video_ts)
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


def load_audio(mov_file: FILE_PATH, enforce_stereo=True) -> Tuple[np.ndarray, float]:
    """Load audio data and the sample rate from a movie file."""
    # Load audio from MOV
    mov_file = resolve_filename(mov_file)
    with mp.VideoFileClip(mov_file) as clip:
        audio = clip.audio.to_soundarray()
        audio_sample_rate = clip.audio.fps

    # Checks on loaded audio
    if enforce_stereo:
        if audio.ndim != 2:
            raise NotImplementedError("Unexpected dimensionality of audio data. Only stereo audio currently supported.")

        n_samples, n_channels = audio.shape
        if n_channels != 2:
            raise NotImplementedError("Only stereo audio currently supported.")

    return audio, audio_sample_rate
