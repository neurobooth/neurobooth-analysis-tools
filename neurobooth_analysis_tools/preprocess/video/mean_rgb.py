"""
Functions to read raw movie files (.avi, .mov) to extract the mean RGB value of each frame.
"""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from typing import Tuple

import cv2
from imutils.video import FileVideoStream

from neurobooth_analysis_tools.data.files import FileMetadata, resolve_filename, FILE_PATH
from neurobooth_analysis_tools.data.types import NeuroboothDevice
import neurobooth_analysis_tools.data.hdf5 as hdf5


def mean_frame_rgb(
        video_file: FILE_PATH,
        input_hdf: FileMetadata,
        output_hdf: FILE_PATH,
        progress_bar: bool = False
) -> None:
    """
    Extract the mean RGB value of each frame in the video file.
    Uses the frame numbers in the input HDF to synchronize to timestamps.
    Output the result to the specified HDF file.
    :param video_file: The path to the raw video file.
    :param input_hdf: The FileMetadata structure for the input HDF file used for time synchronization.
    :param output_hdf: The path of the new HDF5 file containing the RGB means.
    :param progress_bar: Whether to display a tdqm progress bar (processing will take a noticeable amount of time).
    """
    if input_hdf.extension != '.hdf5':
        raise ValueError(
            f"A source HDF5 file is needed for time synchronization. Received: {resolve_filename(input_hdf)}"
        )

    # Load the input HDF5 file
    device = hdf5.load_neurobooth_file(input_hdf)
    if input_hdf.device == NeuroboothDevice.IPhone:
        df = hdf5.extract_iphone(device, exclude_beginning=False, include_event_flags=False)
        sync_column = 'FrameNum'
    elif input_hdf.device == NeuroboothDevice.FLIR:
        df = hdf5.extract_flir(device, include_event_flags=False)
        sync_column = 'FrameNum'
    elif input_hdf.device == NeuroboothDevice.RealSense:
        df = hdf5.extract_realsense(device, include_event_flags=False)
        sync_column = 'FrameNum_RealSense'
    else:
        raise NotImplementedError(f"Mean RGB extraction not implemented for {input_hdf.device.name}.")

    # Process the video file
    mean_rgb = process_video_mean_rgb(video_file, progress_bar=progress_bar)

    # Synchronize timestamps
    sync_df = pd.merge(mean_rgb, df, how='inner', left_on='FrameNum', right_on=sync_column)
    mean_rgb = sync_df[['R', 'G', 'B']].to_numpy()
    time_stamps = sync_df['Time_LSL'].to_numpy()

    # Alter the appropriate data and write the new HDF5 file
    new_info = device.data.info
    new_info['name'] = 'RGB frame mean'
    new_info['desc'][0]['column_names'] = ["['R', 'G', 'B']"]  # Looks odd, but is the current unfortunate format
    new_device = hdf5.Device(
        data=hdf5.DataGroup(
            info=new_info,
            footer=device.data.footer,
            time_series=mean_rgb,
            time_stamps=time_stamps,
        ),
        marker=device.marker,
    )
    hdf5.save_neurobooth_file(output_hdf, new_device, overwrite=True)


def process_video_mean_rgb(video_file: FILE_PATH, progress_bar: bool = False) -> pd.DataFrame:
    """
    Extract the mean RGB value of each frame in the video file.
    :param video_file: The file to processes.
    :param progress_bar: Whether to display a tdqm progress bar (processing will take a noticeable amount of time).
    :return: A DataFrame with frame numbers and the mean color channel vales for each frame.
    """
    video_file = resolve_filename(video_file)
    cap = FileVideoStream(video_file).start()

    n_frames = int(cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_iterator = range(n_frames)
    if progress_bar:
        frame_iterator = tqdm(frame_iterator, unit='frames')

    frame_means = np.full((n_frames, 3), np.nan)
    for i in frame_iterator:
        frame = cap.read()
        frame_means[i] = frame.mean(axis=(0, 1))

    cap.stop()
    cv2.destroyAllWindows()

    return pd.DataFrame(
        data=frame_means,
        columns=['R', 'G', 'B'],
    ).reset_index(names='FrameNum')
