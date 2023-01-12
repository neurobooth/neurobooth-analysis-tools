"""
Functions for loading Neurobooth data from .json files
The primary use-case are JSON files produced by the iPhone.
"""

import numpy as np
import pandas as pd
import json

from neurobooth_analysis_tools.data.files import FILE_PATH, resolve_filename


def parse_iphone_json(file: FILE_PATH) -> pd.DataFrame:
    with open(resolve_filename(file), 'r') as f:
        data = json.load(f)
    data = list(map(json.loads, data))

    frames = np.array([x['FrameNumber'] for x in data], dtype=int)
    timestamps = np.array([x['Timestamp'] for x in data], dtype=float)

    # Sort based on frame number (should already be sorted, but this will ensure it)
    sort_idx = np.argsort(frames)
    frames = frames[sort_idx]
    timestamps = timestamps[sort_idx]

    return pd.DataFrame.from_dict({
        'FrameNum': frames + 1,  # Make sure the frame number matches what the HDF file has
        'Time_JSON': timestamps,
    })
