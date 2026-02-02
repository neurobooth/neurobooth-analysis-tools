"""
Utility Functions and Data Structures for XDF File Processing.

This module provides core utilities for working with Neurobooth XDF (Extensible Data Format)
files, including:
- Data structures for device data representation
- Configuration loading and parsing
- XDF file discovery and parsing
- Time synchronization utilities

The utilities are designed to be lightweight and avoid dependencies on the full
Neurobooth-OS installation.
"""


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

# Pattern for Neurobooth session directories: SUBJECT_YYYY-MM-DD or SUBJECT_YYYY_MM_DD
SUBJECT_YYYY_MM_DD = re.compile(r'(\d+)_(\d\d\d\d)[_-](\d\d)[_-](\d\d).*')

def has_extension(file: str, extension: str) -> bool:
    """
    Check if a file has a specific extension (case-insensitive).
    
    Args:
        file (str): The filename to check.
        extension (str): The extension to check for (should include the dot, e.g., '.xdf').
        
    Returns:
        bool: True if the file has the specified extension, False otherwise.
    """
    _, ext = os.path.splitext(file)
    return ext.lower() == extension.lower()

# Partial function for checking XDF files specifically
is_xdf = partial(has_extension, extension='.xdf')

def default_source_directories() -> List[str]:
    """
    Load the default source directories for Neurobooth data from configuration file.
    
    Reads from 'default_source_directories.txt' in the current directory, where each
    line contains a path to a data directory.
    
    Returns:
        List[str]: List of absolute paths to data directories.
        
    Raises:
        FileNotFoundError: If default_source_directories.txt is not found.
        
    Note:
        The file should contain one directory path per line. Empty lines are ignored.
    """
    # lines = resources.read_text(__package__,'default_source_directories.txt').strip().splitlines(keepends=False)
    with open('default_source_directories.txt', 'r') as f:
        text_content = f.read()
    lines = text_content.strip().splitlines(keepends=False)
    return [os.path.abspath(line) for line in lines]

def is_valid_identifier(identifier: str, pattern: re.Pattern = SUBJECT_YYYY_MM_DD) -> bool:
    """
    Test if a string matches the expected Neurobooth session identifier pattern.
    
    The default pattern is SUBJECT_YYYY-MM-DD or SUBJECT_YYYY_MM_DD, where:
    - SUBJECT is a numeric subject ID
    - YYYY is a 4-digit year
    - MM is a 2-digit month
    - DD is a 2-digit day
    Both hyphens (-) and underscores (_) are permitted between date fields.
    
    Args:
        identifier (str): The string to validate.
        pattern (re.Pattern, optional): Regex pattern to match against. 
                                        Defaults to SUBJECT_YYYY_MM_DD.
        
    Returns:
        bool: True if the identifier matches the pattern, False otherwise.
    """
    matches = re.fullmatch(pattern, identifier)
    return matches is not None

def discover_session_directories(data_dirs: List[str]) -> Tuple[List[str], List[str]]:
    """
    Discover valid Neurobooth session directories within the given data directories.
    
    Scans each data directory and identifies subdirectories that match the expected
    Neurobooth session naming pattern (SUBJECT_YYYY-MM-DD).
    
    Args:
        data_dirs (List[str]): List of root data directory paths to search.
        
    Returns:
        Tuple[List[str], List[str]]: A tuple containing:
            - List of session names (directory names only)
            - List of full paths to session directories
    """
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
    """
    Structured representation of data parsed for a single device from an XDF file.
    
    This structure encapsulates all information needed to write device-specific HDF5
    files and log the split operation to the database.
    
    Attributes:
        device_id (str): Unique identifier for the device 
        device_data (Any): Raw device data stream from the XDF file, including
                          time_stamps, time_series, and metadata.
        marker_data (Any): Marker stream data associated with this device, used for
                          event synchronization.
        video_files (List[str]): List of video filenames associated with this device
                                 stream (empty if no videos).
        sensor_ids (List[str]): List of sensor identifiers for this device
        hdf5_path (str): Full path where the device HDF5 file should be written.
    """
    device_id: str
    device_data: Any
    marker_data: Any
    video_files: List[str]
    sensor_ids: List[str]
    hdf5_path: str

class DatabaseConfig(BaseModel):
    """
    Pydantic model for database connection configuration.
    
    Attributes:
        dbname (str): Name of the PostgreSQL database.
        user (str): Database user name.
        password (str): Database password.
        host (str): Database host address.
        port (int): Database port number.
        remote_host (Optional[str]): Remote host for SSH tunneling (if needed).
        remote_user (Optional[str]): Remote user for SSH tunneling (if needed).
    """
    dbname: str
    user: str
    password: str
    host: str
    port: int
    remote_host: Optional[str] = None
    remote_user: Optional[str] = None

class AppConfig(BaseModel):
    """
    Pydantic model for the overall application configuration.
    
    This is the top-level configuration object that contains all subsystem
    configurations (currently only database, but extensible for future needs).
    
    Attributes:
        database (DatabaseConfig): Database connection configuration.
    """
    database: DatabaseConfig
    #if we need another configs in future
    

#functions
def load_config(config_path: str) -> AppConfig:
    """
    Load and parse a JSON configuration file into Pydantic models.
    
    Args:
        config_path (str): Path to the neurobooth_os_config.json configuration file.
        
    Returns:
        AppConfig: Parsed configuration object with validated fields.
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        pydantic.ValidationError: If the configuration structure is invalid.
    """
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    return AppConfig(**config_data)

def compute_clocks_diff() -> float:
    """
    Compute the offset between the local LSL clock and Unix (epoch) time.
    
    This is used to convert LSL timestamps (used internally by XDF streams) to
    Unix timestamps for database storage and cross-system synchronization.
    
    Returns:
        float: The offset in seconds to add to LSL timestamps to get Unix time.
               Calculated as: unix_time - lsl_time
    """
    return time.time() - pylsl.local_clock()

def _make_hdf5_path(xdf_path: str, device_id: str, sensor_ids: List[str]) -> str:
    """
    Generate a standardized path for a device-specific HDF5 file extracted from XDF.
    
    Args:
        xdf_path (str): Full path to the source XDF file.
        device_id (str): ID string for the device .
        sensor_ids (List[str]): List of sensor ID strings .
        
    Returns:
        str: Full path for the corresponding device HDF5 file.
        
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