"""
Functions for loading Neurobooth data from .json files
The primary use-case are JSON files produced by the iPhone.
"""

import numpy as np
import pandas as pd
import json
from typing import NamedTuple, List, Dict

from neurobooth_analysis_tools.data.files import FILE_PATH, resolve_filename


class IPhoneJsonResult(NamedTuple):
    video: pd.DataFrame
    dropped_video: pd.DataFrame
    audio: pd.DataFrame


def parse_iphone_json(file: FILE_PATH) -> IPhoneJsonResult:
    with open(resolve_filename(file), 'r') as f:
        data = json.load(f)
    data = list(map(json.loads, data))

    return IPhoneJsonResult(
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

    df = pd.DataFrame.from_dict({
        'SampleCount': np.array([x['AudioSampleCount'] for x in sample_data], dtype=int),
        'SampleDuration': np.array([x['AudioSampleDuration'] for x in sample_data], dtype=float),
    })

    return df
