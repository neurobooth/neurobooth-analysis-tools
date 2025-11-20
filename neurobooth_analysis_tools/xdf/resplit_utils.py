import json
import time
from typing import NamedTuple, List, Any, Optional
import numpy as np
import pylsl
import pyxdf
from importlib import resources
from pydantic import BaseModel, Field
from functools import partial
import os
import re
from typing import NamedTuple, Tuple, List, Union, Optional

SUBJECT_YYYY_MM_DD = re.compile(r'(\d+)_(\d\d\d\d)[_-](\d\d)[_-](\d\d).*')

def has_extension(file: str, extension: str) -> bool:
    _, ext = os.path.splitext(file)
    return ext.lower() == extension.lower()

is_xdf = partial(has_extension, extension='.xdf')

def default_source_directories() -> List[str]:
    # lines = resources.read_text(__package__,'default_source_directories.txt').strip().splitlines(keepends=False)
    with open('default_source_directories.txt', 'r') as f:
        text_content = f.read()
    lines = text_content.strip().splitlines(keepends=False)
    return [os.path.abspath(line) for line in lines]

def is_valid_identifier(identifier: str, pattern: re.Pattern = SUBJECT_YYYY_MM_DD) -> bool:
    """Test if a string starts with a SUBJECT_YYYY-MM-DD pattern. (Both - and _ permitted between date fields.)"""
    matches = re.fullmatch(pattern, identifier)
    return matches is not None

def discover_session_directories(data_dirs: List[str]) -> Tuple[List[str], List[str]]:
    """Discover a list of Neurobooth sessions from within the given data directories."""
    sessions = []
    session_dirs = []
    for d in data_dirs:
        for session in os.listdir(d):
            session_path = os.path.join(d, session)
            if os.path.isdir(session_path) and is_valid_identifier(session):
                sessions.append(session)
                session_dirs.append(session_path)

    return sessions, session_dirs
#all classes
class DeviceData(NamedTuple):
    """A structured representation of data parsed for a single device."""
    device_id: str
    device_data: Any
    marker_data: Any
    video_files: List[str]
    sensor_ids: List[str]
    hdf5_path: str

class DatabaseConfig(BaseModel):
    """Pydantic model for database connection details."""
    dbname: str
    user: str
    password: str
    host: str
    port: int
    remote_host: Optional[str] = None
    remote_user: Optional[str] = None

class AppConfig(BaseModel):
    """Pydantic model for the overall application configuration."""
    database: DatabaseConfig
    #if we need another configs in future
    

#functions
def load_config(config_path: str) -> AppConfig:
    """
    Loads a JSON configuration file and parses it into Pydantic models.
    
    :param config_path: Path to the neurobooth_os_config.json file.
    :return: An AppConfig object with the loaded settings.
    """
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    return AppConfig(**config_data)

def compute_clocks_diff() -> float:
    """
    Compute difference between local LSL and Unix clock.
    
    :returns: The offset between the clocks (in seconds).
    """
    return time.time() - pylsl.local_clock()

def _make_hdf5_path(xdf_path: str, device_id: str, sensor_ids: List[str]) -> str:
    """
    Generate a path for a device HDF5 file extracted from an XDF file.

    :param xdf_path: Full path to the XDF file.
    :param device_id: ID string for the device.
    :param sensor_ids: List of ID strings for each included sensor.
    :returns: A standardized file name for corresponding device HDF5 file.
    """
    sensor_list = "-".join(sensor_ids)
    head, _ = os.path.splitext(xdf_path)
    return f"{head}-{device_id}-{sensor_list}.hdf5"
    
def parse_xdf(xdf_path: str, device_ids: Optional[List[str]] = None) -> List[DeviceData]:
    """
    Split an XDF file into device/stream-specific HDF5 files.

    :param xdf_path: The path to the XDF file to parse.
    :param device_ids: If provided, only parse files corresponding to the specified devices.
    :returns: A structured representation of information extracted from the XDF file for each device.
    """
    data, _ = pyxdf.load_xdf(xdf_path, dejitter_timestamps=False)

    # Find marker stream to associate with each device
    marker_streams = [d for d in data if d["info"]["name"] == ["Marker"]]
    if not marker_streams:
        raise ValueError("Could not find Marker stream in the XDF file.")
    marker = marker_streams[0]

    # Get video file names if "videofiles" marker is present
    video_files = {}
    video_data_streams = [v for v in data if v.get("info", {}).get("name") == ["videofiles"]]
    if video_data_streams:
        # Video file marker format is ["streamName, fname.mov"]
        for d in video_data_streams[0]["time_series"]:
            if not d or not d[0]:
                continue
            try:
                stream_id, file_id = d[0].split(",", 1)
                if stream_id in video_files:
                    video_files[stream_id].append(file_id)
                else:
                    video_files[stream_id] = [file_id]
            except ValueError:
                print(f"Warning: Could not parse video file marker: {d[0]}")


    # Parse device data into a more structured format
    results = []
    for device_data in data:
        stream_name = device_data.get("info", {}).get("name", [None])[0]
        
        # Exclude streams that aren't primary data streams
        if stream_name in ["Marker", "videofiles"] or stream_name is None:
            continue
            
        desc = device_data.get("info", {}).get("desc", [None])[0]
        if desc is None:
            print(f"Warning: Stream '{stream_name}' has no description. Skipping.")
            continue
            
        try:
            device_id = desc["device_id"][0]
            sensor_id_str = desc["sensor_ids"][0]
            sensor_ids = json.loads(sensor_id_str.replace("'", '"'))
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"Warning: Could not parse metadata for stream '{stream_name}'. Error: {e}. Skipping.")
            continue

        if (device_ids is not None) and (device_id not in device_ids):
            continue

        results.append(DeviceData(
            device_id=device_id,
            device_data=device_data,
            marker_data=marker,
            video_files=video_files.get(stream_name, []),
            sensor_ids=sensor_ids,
            hdf5_path=_make_hdf5_path(xdf_path, device_id, sensor_ids),
        ))

    return results