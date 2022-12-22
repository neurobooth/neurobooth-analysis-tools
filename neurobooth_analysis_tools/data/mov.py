"""
Functions for loading Neurobooth data from .mov files.
"""
import os
import numpy as np
import pandas as pd
from typing import Union
import moviepy.editor as mp

import neurobooth_analysis_tools.data.hdf5 as hdf5
from neurobooth_analysis_tools.data.files import FileMetadata


def load_iphone_audio(mov_file: Union[str, FileMetadata], sync_device: hdf5.Device) -> pd.DataFrame:
    """Load audio data from an iPhone .mov file.
    Data from a synchronized HDF5 file is needed to infer timestamps and marker events.
    """
    video_ts = sync_device.data.time_stamps
    if isinstance(mov_file, FileMetadata):
        mov_path = os.path.join(mov_file.session_path, mov_file.file_name)
    elif isinstance(mov_file, str):
        mov_path = mov_file
    else:
        raise ValueError("Unsupported type for argument mov_file.")

    # Load audio from MOV
    with mp.VideoFileClip(mov_path) as clip:
        audio = clip.audio.to_soundarray()
        audio_sample_rate = clip.audio.fps

    if audio.ndim != 2:
        raise NotImplementedError("Unexpected dimensionality of audio data. Only stereo audio currently supported.")

    n_samples, n_channels = audio.shape
    if n_channels != 2:
        raise NotImplementedError("Only stereo audio currently supported.")

    # Average stereo channels to get mono
    audio = audio.mean(axis=1)

    # Discard beginning of the video time-series as the sampling rate/data are untrustworthy
    video_start_idx = hdf5.find_idx_stable_sample_rate(video_ts)
    start_time = video_ts[video_start_idx]
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
    df['Flag_Instructions'] = hdf5.create_instruction_mask(sync_device, audio_ts)
    df['Flag_Task'] = hdf5.create_task_mask(sync_device, audio_ts)

    return df
