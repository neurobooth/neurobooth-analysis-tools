"""
Functions to read raw movie files (.avi, .mov) to extract the mean RGB value of each frame.
"""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

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
        raise ValueError("RealSense data should be processed using mean_frame_rgb_realsense")
    else:
        raise NotImplementedError(f"Mean RGB extraction not implemented for {input_hdf.device.name}.")

    # Process the video file
    mean_rgb = process_video_mean_rgb(video_file, progress_bar=progress_bar)

    # Synchronize timestamps
    sync_df = pd.merge(mean_rgb, df, how='inner', left_on='FrameNum', right_on=sync_column)

    write_processed_hdf5(
        time_series=sync_df[['R', 'G', 'B']],
        time_stamps=sync_df['Time_LSL'],
        device=device,
        output_hdf=output_hdf,
    )


def mean_frame_rgb_realsense(
        video_file: FILE_PATH,
        timestamp_npy: FILE_PATH,
        input_hdf: FileMetadata,
        output_hdf: FILE_PATH,
        progress_bar: bool = False
) -> None:
    """
    Extract the mean RGB value of each frame in an extracted color video file.
    Synchronize results to the frames in the input HDF file based on the camera-provided timestamps.
    Output the result to the specified HDF file.
    :param video_file: The path to an extracted color video file (e.g., obtained using bag2avi).
    :param timestamp_npy: A .npy file containing frame timestamps (in ms) obtained from the camera.
    :param input_hdf: The FileMetadata structure for the input HDF file used for time synchronization.
    :param output_hdf: The path of the new HDF5 file containing the RGB means.
    :param progress_bar: Whether to display a tdqm progress bar (processing will take a noticeable amount of time).
    """
    if input_hdf.extension != '.hdf5':
        raise ValueError(
            f"A source HDF5 file is needed for time synchronization. Received: {resolve_filename(input_hdf)}"
        )

    # Load the input HDF5 file
    if input_hdf.device == NeuroboothDevice.RealSense:
        device = hdf5.load_neurobooth_file(input_hdf)
        df = hdf5.extract_realsense(device, include_event_flags=False)
    else:
        raise ValueError("mean_frame_rgb_realsense should only be used for RealSense data.")

    # Process the video file
    mean_rgb = process_video_mean_rgb(video_file, progress_bar=progress_bar)
    mean_rgb = mean_rgb.rename(columns={'FrameNum': 'FrameNum_Bag'})

    # Load the frame timestamps and convert from ms to s
    mean_rgb['Time_Bag'] = np.load(resolve_filename(timestamp_npy)) / 1e3

    # Synchronize timestamps
    sync_df = fuzzy_join_realsense_timestamps(df, mean_rgb)

    write_processed_hdf5(
        time_series=sync_df[['FrameNum_Bag', 'FrameNum', 'Time_Bag', 'Time_RealSense', 'R', 'G', 'B']],
        time_stamps=sync_df['Time_LSL'],
        device=device,
        output_hdf=output_hdf,
    )


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


def fuzzy_join_realsense_timestamps(
        lsl_df: pd.DataFrame,
        rgb_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge LSL frame/timestamp data with frame/color/timestamp data derived from the .bag file.

    :param lsl_df: A dataframe containing LSL-captured frame numbers and timestamps.
    :param rgb_df: A dataframe containing BAG-derived mean color values, frame numbers, and timestamps.
    :return: The joined dataframe.
    """
    lsl_df['FrameNum_Bag'] = pd.Series(pd.NA, index=lsl_df.index, dtype='UInt32')
    for _, row in rgb_df.iterrows():
        match_idx = (lsl_df['Time_RealSense'] - row['Time_Bag']).abs().argmin()
        lsl_df['FrameNum_Bag'].iloc[match_idx] = row['FrameNum_Bag']
    return pd.merge(lsl_df, rgb_df, how='right', on='FrameNum_Bag')


def write_processed_hdf5(
        time_series: pd.DataFrame,
        time_stamps: pd.Series,
        device: hdf5.Device,
        output_hdf: FILE_PATH,
) -> None:
    """
    Write an HDF5 file containing the synchronized mean color data.

    :param time_series: The mean color time-series (alongside any other relevant time-series)
    :param time_stamps: LSL timestamps for the time-series.
    :param device: The device object representing the input HDF5 file used for synchronization.
    :param output_hdf: The path to write the new HDF5 file to.
    """
    columns = time_series.columns
    time_series = time_series.to_numpy(dtype='float64', na_value=np.nan)
    time_stamps = time_stamps.to_numpy(dtype='float64')

    # Alter the appropriate data and write the new HDF5 file
    new_info = device.data.info
    new_info['name'] = 'RGB frame mean'
    new_info['desc'][0]['column_names'] = [  # Looks odd, but is the current unfortunate format
        "[" + ", ".join([f"'{c}'" for c in columns]) + "]"
    ]
    new_device = hdf5.Device(
        data=hdf5.DataGroup(
            info=new_info,
            footer=device.data.footer,
            time_series=time_series,
            time_stamps=time_stamps,
        ),
        marker=device.marker,
    )
    hdf5.save_neurobooth_file(output_hdf, new_device, overwrite=True)
