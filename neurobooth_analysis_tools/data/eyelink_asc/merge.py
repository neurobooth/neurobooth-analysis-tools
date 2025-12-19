"""
Code for matching and synchronizing data from HDF5 and EDF files
"""

import os
from typing import NamedTuple, Union, Tuple
import numpy as np
import pandas as pd

from neurobooth_analysis_tools.data.files import FileMetadata
from neurobooth_analysis_tools.data.hdf5 import (
    DataException, load_neurobooth_file, extract_eyelink,
)
from neurobooth_analysis_tools.data.eyelink_asc.event_parser import EventData, Span, parse_asc_events
from neurobooth_analysis_tools.data.eyelink_asc.edf2asc import (
    extract_events_ascii, extract_href_ascii, extract_gaze_ascii,
)
from neurobooth_analysis_tools.data.eyelink_asc.parser import href_velocity, parse_href, parse_gaze
from neurobooth_analysis_tools.preprocess.mask import find_continguous_masks


class FileMatchException(Exception):
    """Indicates trouble matching HDF5 and EDF files"""
    pass


class EyeLinkFiles(NamedTuple):
    """Structure to bundle associated hdf5 and edf files."""
    hdf5: FileMetadata
    edf: FileMetadata


def align_eyelink_files(
        hdf5_files: list[FileMetadata],
        edf_files: list[FileMetadata],
        join_key: list[str] = ('subject_id', 'datetime', 'task', 'device'),
) -> list[EyeLinkFiles]:
    """Align two file arrays based on the specified fields (i.e., the join key).
    Raises an exception if there are no matches in the session.
    Otherwise, returns the last match. (If there was a repeat, we discard the first occurrence.)

    :param hdf5_files: A list of metadata objects for HDF5 files
    :param edf_files: A list of metadata objects for EDF files
    :param join_key: The metadata fields used to match files
    :returns: A list of tuples with paired up files.
    """
    hdf5_df = pd.DataFrame(hdf5_files, columns=FileMetadata._fields)
    hdf5_df['hdf5'] = hdf5_files
    hdf5_df['task'] = hdf5_df['task'].apply(lambda x: x.name)
    edf_df = pd.DataFrame(edf_files, columns=FileMetadata._fields)
    edf_df['edf'] = edf_files
    edf_df['task'] = edf_df['task'].apply(lambda x: x.name)

    join_df = pd.merge(hdf5_df, edf_df, on=join_key, how='inner')
    if join_df.shape[0] == 0:
        raise FileMatchException('Could not find matching hdf5 and edf files for session!')
    join_df = join_df[[*join_key, 'hdf5', 'edf']]
    join_df = join_df.sort_values(by='datetime').groupby(by=list(join_key)).last()  # Take latest match in session

    return [EyeLinkFiles(hdf5=row['hdf5'], edf=row['edf']) for _, row in join_df.iterrows()]


def merge_eyelink_files(
        file_pair: EyeLinkFiles,
        separate_eyes: bool = False,
        task_only: bool = False,
        min_task_duration: float = 5,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Preprocess eyetracker data. Load raw files, extract saccades from EyeLink EDF files, join information together.

    NOTE: It is a good idea to first run check_edf2asc() to make sure edf2asc is installed on the system
    NOTE: This function may overwrite ASCII files, so work on a copied data slice.

    :param file_pair: Paired HDF5 and EDF file from the same session/recording (see align_eyelink_files).
    :param separate_eyes: If True, return separate data frames for the left and right eye.
    :param task_only: Whether to trim the data to just the (last) task performance.
    :param min_task_duration: The minimum duration (s) of a task to be processed.
    :returns: The merged data frames for each eye: (left_eye, right_eye)
    """
    try:
        # Load EDF gaze events
        edf_path = os.path.join(file_pair.edf.session_path, file_pair.edf.file_name)
        asc_events = parse_asc_events(extract_events_ascii(edf_path, rm=True))

        # Load HREF position and velocity from EDF
        edf_href_df = parse_href(extract_href_ascii(edf_path, rm=True))
        edf_href_df = href_velocity(edf_href_df)

        # Load GAZE position and velocity from EDF
        edf_gaze_df = parse_gaze(extract_gaze_ascii(edf_path, rm=True))

        if edf_gaze_df.shape[0] == 0 or edf_href_df.shape[0] == 0:
            raise DataException(f"Unable to parse samples from {file_pair.edf.file_name}.")

        # Load gaze time-series
        hdf5_data = load_neurobooth_file(file_pair.hdf5)
        if hdf5_data.data.time_series.shape[0] <= 2:
            raise DataException('Missing sufficient time-series samples in HDF5.')
        gaze_data = extract_eyelink(hdf5_data)

        # Synchronize gaze position and resolution from EDF with data from HDF5
        gaze_data = pd.merge(gaze_data, edf_gaze_df, how='left', on='Time_EDF', suffixes=('', '_EDF'))

        # Calculate gaze velocity and convert from px/s to dva/s
        ts = gaze_data['Time_EDF'].to_numpy() / 1e3  # ms -> s
        # Use EDF resolution since the resolution from the HDF5 files was bugged
        xres, yres = gaze_data['ResolutionX_EDF'].to_numpy(), gaze_data['ResolutionY_EDF'].to_numpy()
        with np.errstate(divide='ignore', invalid='ignore'):
            # See EyeLink Portable Duo manual section 4.4.2.4, "Gaze Resolution Data"
            gaze_data['R_GazeX_Vel'] = np.gradient(gaze_data['R_GazeX'].to_numpy(), ts) / xres
            gaze_data['R_GazeY_Vel'] = np.gradient(gaze_data['R_GazeY'].to_numpy(), ts) / yres
            gaze_data['L_GazeX_Vel'] = np.gradient(gaze_data['L_GazeX'].to_numpy(), ts) / xres
            gaze_data['L_GazeY_Vel'] = np.gradient(gaze_data['L_GazeY'].to_numpy(), ts) / yres

        # Synchronize HREF positions and velocity (in dva) with gaze data
        gaze_data = pd.merge(gaze_data, edf_href_df, how='left', on='Time_EDF')

        # Synchronize gaze events with gaze time-series
        gaze_data = sync_events(gaze_data, asc_events)

        if task_only:  # Limit data to task performance (the last if there are multiple)
            task_mask = gaze_data['Flag_Task'].to_numpy(dtype=bool)
            task_regions = find_continguous_masks(task_mask)
            # We sometimes get regions with a handful of samples. Filter out smaller regions.
            task_regions = [t for t in task_regions if t.sum() >= (min_task_duration * 1000)]
            if not task_regions:
                raise DataException('No sufficient-length task regions.')
            task_mask = task_regions[-1]
            gaze_data = gaze_data.loc[task_mask]

        # Convert units
        gaze_data['Time_EDF'] /= 1e3  # ms -> s
        gaze_data['Target_Distance'] /= 1e3  # mm -> m

        # Choose which source to pull some data from and rename columns
        gaze_data = gaze_data.drop(columns=[
            'ResolutionX', 'ResolutionY',
        ]).rename(columns={
            'Target_Distance': 'TargetDistance',
            'ResolutionX_EDF': 'ResolutionX',
            'ResolutionY_EDF': 'ResolutionY',
            'Time_EDF': 'SampleTime',
        })

        # Return output data frames
        return split_eyes(gaze_data) if separate_eyes else gaze_data

    except Exception as e:
        raise DataException(f"Exception occurred for {file_pair.hdf5.file_name} or {file_pair.edf.file_name}.") from e


def sync_events(data: pd.DataFrame, events: EventData) -> pd.DataFrame:
    """Update the given data frame with new columns representing saccade/fixation/blink events detected by Eyelink."""
    ts = data['Time_EDF']
    data['Flag_R_Saccade'] = _event_int_mask(ts, events.right_eye.saccades) > 0
    data['Flag_R_Fixation'] = _event_int_mask(ts, events.right_eye.fixations) > 0
    data['Flag_R_Blink'] = _event_int_mask(ts, events.right_eye.blinks) > 0
    data['Flag_L_Saccade'] = _event_int_mask(ts, events.left_eye.saccades) > 0
    data['Flag_L_Fixation'] = _event_int_mask(ts, events.left_eye.fixations) > 0
    data['Flag_L_Blink'] = _event_int_mask(ts, events.left_eye.blinks) > 0
    return data


def _event_int_mask(device_ts: pd.Series, events: list[Span]) -> np.ndarray:
    """Create an ascending integer mask reflecting the presence and order of the specified events."""
    mask = np.zeros(device_ts.shape, dtype=int)
    for i, e in enumerate(events):
        ts_mask = (device_ts >= e.start) & (device_ts < e.end)
        mask[ts_mask] = i + 1
    return mask


def split_eyes(gaze_data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """Split a combined data frame into separate data frames for each eye"""
    results = []
    for eye in ('L', 'R'):
        # Split out a single eye from the joint data
        eye_df = gaze_data[[
            f'{eye}_GazeX', f'{eye}_GazeY',
            f'{eye}_GazeX_HREF', f'{eye}_GazeY_HREF',
            f'{eye}_GazeX_Vel', f'{eye}_GazeY_Vel',
            'TargetDistance', 'ResolutionX', 'ResolutionY',
            'SampleTime', 'Time_LSL',
            'Flag_Instructions', 'Flag_Task',
            f'Flag_{eye}_Saccade', f'Flag_{eye}_Fixation', f'Flag_{eye}_Blink',
        ]]
        eye_df = eye_df.rename(columns={
            f'{eye}_GazeX': 'GazeX',
            f'{eye}_GazeY': 'GazeY',
            f'{eye}_GazeX_HREF': 'GazeX_HREF',
            f'{eye}_GazeY_HREF': 'GazeY_HREF',
            f'{eye}_GazeX_Vel': f'GazeX_Vel',
            f'{eye}_GazeY_Vel': f'GazeY_Vel',
            f'Flag_{eye}_Saccade': 'Flag_Saccade',
            f'Flag_{eye}_Fixation': 'Flag_Fixation',
            f'Flag_{eye}_Blink': 'Flag_Blink',
        })
        results.append(eye_df)

    return results[0], results[1]
