"""
Functions for loading Neurobooth data from .json files
The primary use-case are JSON files produced by the iPhone.
"""

import numpy as np
import pandas as pd
import json
from typing import NamedTuple, List, Dict

from neurobooth_analysis_tools.data.files import FILE_PATH, resolve_filename


class IPhoneMetadata(NamedTuple):
    ios_version: str
    app_version: str
    device_id: str
    device_type: str  # See https://gist.github.com/adamawolf/3048717


class IPhoneJsonResult(NamedTuple):
    metadata: IPhoneMetadata | None  # Older JSON files do not contain metadata
    video: pd.DataFrame
    dropped_video: pd.DataFrame
    audio: pd.DataFrame


def parse_iphone_json(file: FILE_PATH) -> IPhoneJsonResult:
    """
    Parse an iPhone JSON file and return its contests in a structured format.
    :param file: The file to parse.
    :return: A structure containing the video and audio synchronization data.
    """
    with open(resolve_filename(file), 'r') as f:
        data = json.load(f)

    # Handle newer file versions. Older files did not have metadata and are a list at the outermost level.
    if isinstance(data, dict):
        metadata = _iphone_json_extract_metadata(data['Metadata'])
        data = data['Framedata']
    else:
        metadata = None

    # Each string in the list can itself be parsed as JSON
    data = list(map(json.loads, data))

    return IPhoneJsonResult(
        metadata=metadata,
        video=_iphone_json_extract_video(data),
        dropped_video=_iphone_json_extract_dropped_video(data),
        audio=_iphone_json_extract_audio(data),
    )


def _iphone_json_extract_video(data: List[Dict[str, str]]) -> pd.DataFrame:
    frame_data = [x for x in data if 'FrameNumber' in x]

    df = pd.DataFrame.from_dict({
        'FrameNum': np.array([x['FrameNumber'] for x in frame_data], dtype=int),
        'Time_JSON': np.array([x['Timestamp'] for x in frame_data], dtype=float),
    })

    return df.sort_values('FrameNum', ignore_index=True)


def _iphone_json_extract_dropped_video(data: List[Dict[str, str]]) -> pd.DataFrame:
    frame_data = [x for x in data if 'DroppedFrameNumber' in x]

    df = pd.DataFrame.from_dict({
        'FrameNum': np.array([x['DroppedFrameNumber'] for x in frame_data], dtype=int),
        'Time_JSON': np.array([x['DroppedFrameTimestamp'] for x in frame_data], dtype=float),
    })

    return df.sort_values('FrameNum', ignore_index=True)


def _iphone_json_extract_audio(data: List[Dict[str, str]]) -> pd.DataFrame:
    sample_data = [x for x in data if 'AudioSampleCount' in x]
    sample_time_data = [json.loads(x['AudioSampleTimeReceived']) for x in sample_data]

    df = pd.DataFrame.from_dict({
        'FrameNum': np.array([x['FrameNumber'] for x in sample_time_data], dtype=int),
        'Time_JSON': np.array([x['Timestamp'] for x in sample_time_data], dtype=float),
        'SampleCount': np.array([x['AudioSampleCount'] for x in sample_data], dtype=int),
        'SampleDuration': np.array([x['AudioSampleDuration'] for x in sample_data], dtype=float),
    })

    return df


def _iphone_json_extract_metadata(metadata: Dict[str, str]) -> IPhoneMetadata:
    return IPhoneMetadata(
        ios_version=metadata['iOSVersionNumber'],
        app_version=metadata['appVersionNumber'],
        device_id=metadata['deviceID'],
        device_type=metadata['deviceType'],
    )
