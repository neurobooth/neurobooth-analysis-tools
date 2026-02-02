"""
This file splits an XDF file into constituent HDF5 files. 
It now uses a local utility file instead of a full Neurobooth-OS installation.
"""

import os
import re
import argparse
import datetime
import importlib
import numpy as np
from typing import NamedTuple, List, Dict, Optional, Any, Callable, ClassVar, Tuple
import h5io
import yaml
import psycopg2 as pg
from pydantic import BaseModel


import resplit_utils as nb_utils

class SplitException(Exception):
    """For generic errors that occur when splitting an XDF file."""
    pass


class HDF5CorrectionSpec(BaseModel):
    """
    Specification for applying corrections to device data streams.
    
    This class loads a YAML configuration file that maps device IDs to correction
    functions. Corrections are applied to legacy data to update metadata structures
    to current standards.
    
    Attributes:
        marker (Optional[str]): Function specification for marker stream corrections
                               in format "module.py::function()".
        devices (Dict[str, str]): Mapping of device IDs to correction function
                                 specifications.
    """
    marker: Optional[str] = None
    devices: Dict[str, str] = {}

    @staticmethod
    def load(path: str) -> 'HDF5CorrectionSpec':
        """
        Load correction specifications from a YAML configuration file.
        
        Args:
            path (str): Path to the YAML configuration file.
            
        Returns:
            HDF5CorrectionSpec: Parsed correction specification.
            
        Raises:
            SplitException: If the file cannot be loaded or parsed.
        """
        try:
            with open(path, 'r') as stream:
                return HDF5CorrectionSpec(**yaml.safe_load(stream))
        except Exception as e:
            raise SplitException(f'Unable to load correction functions from {path}!') from e
    # Pattern for parsing function specifications: "module.py::function()"
    FUNC_STR_PATTERN: ClassVar = re.compile(r'(.*)\.py::(.*)\(\)')

    @staticmethod
    def import_function(func_str: str) -> Callable:
        """
        Import and return a function specified by a module.py::function() string.
        
        Args:
            func_str (str): Function specification in format "module.py::function()".
                           Example: "hdf5_corrections.py::correct_eyelink()"
            
        Returns:
            Callable: The imported function object.
            
        Raises:
            SplitException: If the function specification is invalid or import fails.
        """
        match = re.match(HDF5CorrectionSpec.FUNC_STR_PATTERN, func_str)
        if match is None:
            raise SplitException(f'The function specification does not match the expected pattern: {func_str}')
        module, func = match.groups()

        try:
            module = importlib.import_module(module)
            return getattr(module, func)
        except Exception as e:
            raise SplitException(f'Unable to import {func_str}') from e

    def correct_device(self, device: nb_utils.DeviceData) -> nb_utils.DeviceData:
        """
        Apply configured corrections to device data.
        
        This method applies both marker corrections (if specified) and device-specific
        corrections to the provided device data structure. Corrections modify metadata
        but do not alter the actual time-series data.
        
        Args:
            device (DeviceData): Device data structure loaded from XDF file.
            
        Returns:
            DeviceData: Device data with corrections applied.
            
        Note:
            Corrections are only applied if specified in the configuration.
            If no correction is specified for a device, it is returned unchanged.
        """
        # Apply marker corrections first (if specified)
        if self.marker is not None:
            func = HDF5CorrectionSpec.import_function(self.marker)
            device = func(device)
        # Apply device-specific corrections (if specified)
        device_id = device.device_id
        if device_id in self.devices:
            func = HDF5CorrectionSpec.import_function(self.devices[device_id])
            device = func(device)

        return device

# Pattern for Neurobooth XDF filenames:
# SUBJECT_YYYY-MM-DD_HHh-MMm-SSs_TASKID_RXXX.xdf
XDF_NAME_PATTERN = re.compile(r'(\d+)_(\d\d\d\d-\d\d-\d\d)_\d\dh-\d\dm-\d\ds_(.*)_R\d\d\d\.xdf', flags=re.IGNORECASE)


class XDFInfo(NamedTuple):
    """
    Structured representation of information extracted from an XDF filename.
    
    Neurobooth XDF files follow a standardized naming convention that encodes
    metadata about the recording session. This class parses and stores that
    information.
    
    Attributes:
        parent_dir (str): Directory containing the XDF file.
        name (str): Full filename of the XDF file.
        subject_id (str): Numeric subject identifier.
        date (datetime.date): Date of the recording session.
        task_id (str): Task identifier (e.g., 'timing_test', 'fixation_task').
        xdf_pathd (str): Full path to the XDF file (for database logging).
    """
    parent_dir: str
    name: str
    subject_id: str
    date: datetime.date
    task_id: str
    xdf_pathd: str

    @property
    def path(self) -> str:
        """
        Get the full path to the XDF file.
        
        Returns:
            str: Absolute path constructed from parent_dir and name.
        """
        return os.path.join(self.parent_dir, self.name)

    @staticmethod
    def parse_xdf_name(xdf_path: str) -> 'XDFInfo':
        """
        Parse an XDF file path to extract metadata from the filename.
        
        Args:
            xdf_path (str): Full or relative path to the XDF file.
            
        Returns:
            XDFInfo: Parsed information from the filename.
            
        Raises:
            SplitException: If the filename doesn't match the expected pattern.
            
        """
        parent_dir, filename = os.path.split(xdf_path)
        match = re.match(XDF_NAME_PATTERN, filename)
        if match is None:
            raise SplitException(f'Unable to parse file name: {filename}')

        subject_id, date_str, task_id = match.groups()
        return XDFInfo(
            parent_dir=parent_dir,
            name=filename,
            subject_id=subject_id,
            date=datetime.date.fromisoformat(date_str),
            task_id=task_id,
            xdf_pathd=xdf_path,
        )


class DatabaseConnection:
    """
    Manages database connections and queries for Neurobooth data processing.
    
    This class handles:
    - Establishing connections to PostgreSQL database
    - Optionally creating SSH tunnels for remote access
    - Querying device configurations
    - Logging split operations
    
    Attributes:
        connection (psycopg2.extensions.connection): Active database connection.
    """

    def __init__(self, config_path: str, tunnel: bool, override_host: Optional[str] = None, override_port: Optional[int] = None):
        """
        Create a new DatabaseConnection from a configuration file.
        
        Args:
            config_path (str): Path to the Neurobooth configuration JSON file.
            tunnel (bool): Whether to establish an SSH tunnel before connecting.
            override_host (Optional[str]): Override the host from the config file.
            override_port (Optional[int]): Override the port from the config file.
            
        Note:
            When using SSH tunnel, the connection will be made through localhost
            on the specified local port.
        """
        self.connection = self.connect(config_path, tunnel, override_host, override_port)

    @staticmethod
    def connect(config_path: str, tunnel: bool, override_host: Optional[str], override_port: Optional[int]) -> pg.extensions.connection:
        """
        Load configuration and establish a PostgreSQL database connection.
        
        Args:
            config_path (str): Path to configuration file.
            tunnel (bool): If True, establish SSH tunnel before connecting.
            override_host (Optional[str]): Host override (useful for tunneling).
            override_port (Optional[int]): Port override (useful for tunneling).
            
        Returns:
            psycopg2.extensions.connection: Active database connection.
        
        """
        # Load and parse configuration
        config = nb_utils.load_config(config_path)
        database_info = config.database
        # Apply overrides if provided
        if override_host:
            database_info.host = override_host
        if override_port:
            database_info.port = override_port
        
        if tunnel:
            from sshtunnel import SSHTunnelForwarder
            
            # Use remote_host and remote_user from config if they exist
            remote_host = database_info.remote_host or 'neurodoor.nmr.mgh.harvard.edu'
            remote_user = database_info.remote_user or os.environ.get('USER')
            
            print(f"Starting the tunnel with {remote_user}@{remote_host}")
            ssh_tunnel = SSHTunnelForwarder(
                remote_host,
                ssh_username=remote_user,
                ssh_pkey="~/.ssh/id_rsa",
                remote_bind_address=(database_info.host, database_info.port),
                local_bind_address=("localhost", 6543),
            )
            ssh_tunnel.start()
            host = ssh_tunnel.local_bind_host
            port = ssh_tunnel.local_bind_port
            print(f"Host and Port are {host} and {port}")
        else:
            host = database_info.host
            port = database_info.port
        # Establish database connection
        return pg.connect(
            database=database_info.dbname,
            user=database_info.user,
            password=database_info.password,
            host=host,
            port=port,
        )
    
    def close(self):
        """Closes the database connection."""
        if self.connection:
            self.connection.close()
    # SQL query to retrieve device IDs for a specific session and task
    DEVICE_ID_QUERY = """
    WITH device AS (
        SELECT UNNEST(tparam.log_device_ids) AS log_device_id
        FROM log_session sess
        JOIN log_task task ON sess.log_session_id = task.log_session_id
        JOIN log_task_param tparam ON task.log_task_id = tparam.log_task_id
        WHERE sess.subject_id = %(subject_id)s
            AND sess.date = %(session_date)s
            AND task.task_id = %(task_id)s
    )
    SELECT dparam.device_id
    FROM device
    JOIN log_device_param dparam ON device.log_device_id = dparam.id
    """

    def get_device_ids(self, xdf_info: XDFInfo) -> List[str]:
        """
        Retrieve the list of device IDs used in a specific recording session.
        
        Queries the database to find which devices were configured for the given
        subject, date, and task combination.
        
        Args:
            xdf_info (XDFInfo): Parsed information from the XDF filename.
            
        Returns:
            List[str]: List of device IDs (e.g., ['eyelink', 'mbient', 'intel']).
            
        """
        query_params = {
            'subject_id': xdf_info.subject_id,
            'session_date': xdf_info.date.isoformat(),
            'task_id': xdf_info.task_id,
        }
        with self.connection.cursor() as cursor:
            cursor.execute(DatabaseConnection.DEVICE_ID_QUERY, query_params)
            return [row[0] for row in cursor.fetchall()]
    
    def log_split(self, xdf_info: XDFInfo, device_data: List[Dict[str, Any]]) -> None:
        """
        Log HDF5 file creation to the database (using slim dictionary format).
        
        Creates entries in the log_split table for each sensor, recording information
        about the extracted HDF5 files, timestamps, and associated video files.
        
        Args:
            xdf_info (XDFInfo): Information about the source XDF file.
            device_data (List[Dict[str, Any]]): List of dictionaries containing:
                - device_id: Device identifier
                - sensor_ids: List of sensor IDs
                - hdf5_path: Path to HDF5 file
                - timestamps: Array of LSL timestamps
                - video_files: List of associated video filenames
                
        Note:
            This version uses a "slim" dictionary format to avoid passing large
            data structures. The actual time-series data is not included, only
            metadata and timestamps.
        """
        with self.connection.cursor() as cursor:
            for device in device_data: # 'device' is now a dictionary
                
                #Use the timestamps directly from the dictionary
                timestamps = device["timestamps"]
                if len(timestamps) < 2:
                    continue

                time_offset = nb_utils.compute_clocks_diff()
                start_time = datetime.datetime.fromtimestamp(timestamps[0] + time_offset).strftime("%Y-%m-%d %H:%M:%S")
                end_time = datetime.datetime.fromtimestamp(timestamps[-1] + time_offset).strftime("%Y-%m-%d %H:%M:%S")
                temporal_resolution = 1 / np.median(np.diff(timestamps))
                
                #get other data using dictionary keys
                hdf5_path = device["hdf5_path"]
                video_files = device["video_files"]
                sensor_ids = device["sensor_ids"]
                device_id = device["device_id"]
                
                # Extract file paths
                hdf5_folder, hdf5_file = os.path.split(hdf5_path)
                _, session_folder = os.path.split(hdf5_folder)
                
                # Build sensor file paths array (HDF5 + associated videos)
                sensor_files = [f'{session_folder}/{hdf5_file}'] + [f'{session_folder}/{f}' for f in video_files]
                
                quoted = ['"' + s.replace('"', '\\"') + '"' for s in sensor_files]
                sensor_file_paths = '{' + ','.join(quoted) + '}'

                # Insert log entry for each sensor
                for sensor_id in sensor_ids:
                    query_params = {
                        'subject_id': xdf_info.subject_id,
                        'date': xdf_info.date.isoformat(),
                        'task_id': xdf_info.task_id,
                        'temporal_resolution': temporal_resolution,
                        'start_time': start_time,
                        'end_time': end_time,
                        'device_id': device_id,
                        'sensor_id': sensor_id,
                        'hdf5_file_path': f'{session_folder}/{hdf5_file}',
                        'xdf_path': xdf_info.xdf_pathd,
                        'sensor_file_paths': sensor_file_paths,
                    }
                    cursor.execute(
                        """
                        INSERT INTO log_split (subject_id, date, task_id, true_temporal_resolution, file_start_time, file_end_time, device_id, sensor_id, hdf5_file_path, xdf_path, sensor_file_path)
                        VALUES (%(subject_id)s, %(date)s, %(task_id)s, %(temporal_resolution)s, %(start_time)s, %(end_time)s, %(device_id)s, %(sensor_id)s, %(hdf5_file_path)s, %(xdf_path)s, %(sensor_file_paths)s)
                        """,
                        query_params
                    )
        # Commit all insertions
        self.connection.commit()


def device_id_from_yaml(file: str, task_id: str) -> List[str]:
    """
    Load device IDs for a task from a YAML configuration file.
    
    This is used as an alternative to database queries when a static task-to-device
    mapping is available. Useful for reprocessing old data where database entries
    may not exist or be accurate.
    
    Args:
        file (str): Path to YAML file containing task_id -> [device_ids] mapping.
        task_id (str): Task identifier to look up.
        
    Returns:
        List[str]: List of device IDs for the specified task.
        
    Raises:
        SplitException: If the file cannot be loaded or task not found.
    """
    try:
        with open(file, 'r') as stream:
            task_device_map = yaml.safe_load(stream)
        return task_device_map[task_id]
    except Exception as e:
        raise SplitException(f'Could not locate task {task_id} using map file {file}.') from e

def _remake_hdf5_path(xdf_path: str, device_id: str, sensor_ids: List[str]) -> str:
    """
    Generate a path for device HDF5 file in the processed data directory.
    
    This function places HDF5 files in a centralized processed data location
    rather than alongside the source XDF files. It maintains the session folder
    structure from the original path.
    
    Args:
        xdf_path (str): Full path to the source XDF file.
        device_id (str): Device identifier.
        sensor_ids (List[str]): List of sensor identifiers.
        
    Returns:
        str: Full path for the HDF5 file in the processed data directory.
        
    Note:
        Creates the target directory if it doesn't exist.
        The processed data location is hardcoded to:
        /space/billnted/4/neurobooth/processed_data/
    """
    sensor_list = "-".join(sensor_ids)
    head, _ = os.path.splitext(xdf_path)
    new_path = "/space/billnted/4/neurobooth/processed_data/"
    base_folder = os.path.basename(os.path.dirname(head))
    directory_to_create = os.path.join(new_path, base_folder)
    new_file_path = os.path.join(directory_to_create, os.path.basename(head))
    if not os.path.exists(directory_to_create):
        os.makedirs(directory_to_create, exist_ok=True)
    return f"{new_file_path}-{device_id}-{sensor_list}.hdf5"

def rewrite_device_hdf5(xdf_path: str, device_data: List[nb_utils.DeviceData]) -> List[nb_utils.DeviceData]:
    """
    Write HDF5 files for each device and return updated DeviceData objects.
    
    For each device, this function:
    1. Constructs the output HDF5 path
    2. Checks if the file already exists (skips if so)
    3. Writes marker and device data to HDF5 format
    4. Returns updated DeviceData with new path
    
    Args:
        xdf_path (str): Path to the source XDF file.
        device_data (List[DeviceData]): List of parsed device data.
        
    Returns:
        List[DeviceData]: Updated device data with new HDF5 paths.
                         Only includes successfully written files.
        
    Note:
        Existing files are skipped to avoid overwriting. Delete old files first
        if you want to regenerate them.
    """
    new_device_data = []
    for dev in device_data:
        # Prepare data structure for HDF5 writing
        data_to_write = {"marker": dev.marker_data, "device_data": dev.device_data}

        # Generate new HDF5 path in processed data directory
        new_hdf5_path = _remake_hdf5_path(xdf_path, dev.device_id, dev.sensor_ids)
        if os.path.exists(new_hdf5_path):
            print(f"HDF5 file {new_hdf5_path} already exists. Skipping.")
            continue
        
        # Create updated DeviceData with new path
        new_dev = nb_utils.DeviceData(
            device_id=dev.device_id,
            device_data=dev.device_data,
            marker_data=dev.marker_data,
            video_files=dev.video_files,
            sensor_ids=dev.sensor_ids,
            hdf5_path=new_hdf5_path,
        )
        new_device_data.append(new_dev)
        # Write to HDF5 file
        h5io.write_hdf5(new_hdf5_path, data_to_write, overwrite=True)
    return new_device_data

def split(
        xdf_path: str,
        database_conn: DatabaseConnection,
        task_map_file: Optional[str] = None,
        corrections: Optional[HDF5CorrectionSpec] = None,
) -> Tuple[XDFInfo, List[Dict[str, Any]]]:
    """
    Split a single XDF file into device-specific HDF5 files.
    
    This is the main entry point for the splitting process. It:
    1. Parses the XDF filename to extract metadata
    2. Determines which devices to extract (from database or YAML)
    3. Parses the XDF file
    4. Applies corrections to legacy data
    5. Writes device-specific HDF5 files
    6. Returns slim data for database logging
    
    Args:
        xdf_path (str): Full path to the XDF file to split.
        database_conn (DatabaseConnection): Active database connection for queries.
        task_map_file (Optional[str]): Path to YAML file with task->device mapping.
                                       If provided, uses this instead of database query.
        corrections (Optional[HDF5CorrectionSpec]): Correction specification to apply.
        
    Returns:
        Tuple[XDFInfo, List[Dict[str, Any]]]: A tuple containing:
            - XDFInfo: Parsed metadata from the filename
            - List of slim device dictionaries for database logging containing:
                - device_id: Device identifier
                - sensor_ids: List of sensor IDs
                - hdf5_path: Path to written HDF5 file
                - timestamps: Array of timestamps
                - video_files: List of video filenames
        
    Raises:
        SplitException: If device IDs cannot be determined or XDF parsing fails.
    """

    # Parse XDF filename to extract metadata
    xdf_info = XDFInfo.parse_xdf_name(xdf_path)
    # Determine which devices to extract
    if task_map_file is not None:
        device_ids = device_id_from_yaml(task_map_file, xdf_info.task_id)
    else:
        device_ids = database_conn.get_device_ids(xdf_info)
        print(f"Got from the database here {device_ids}")

    if not device_ids:
        raise SplitException(f'Could not locate task ID {xdf_info.task_id} for session {xdf_info.subject_id}_{xdf_info.date.isoformat()}.')

    # Parse XDF file
    try:
        device_data = nb_utils.parse_xdf(xdf_path, device_ids)
    except Exception as e:
        print(f"[ERROR] Error parsing XDF file {xdf_path}: {e}. Skipping file.")
        return xdf_info, []
    # Apply corrections to legacy data
    if corrections is not None:
        device_data = [corrections.correct_device(dev) for dev in device_data]
    # Write HDF5 files
    device_data = rewrite_device_hdf5(xdf_path, device_data)
    
    #Slim down data for returning,thsi isto avoid passing large data structures
    slim_data = []
    for dev in device_data:
        timestamps = dev.device_data.get("time_stamps", [])
        slim_data.append({
            "device_id": dev.device_id,
            "sensor_ids": dev.sensor_ids,
            "hdf5_path": dev.hdf5_path,
            "timestamps": timestamps,
            "video_files": dev.video_files,
        })

    return xdf_info, slim_data


def parse_arguments() -> Dict[str, Any]:
    """
    Parse command-line arguments for the split function.
    
    Returns:
        Dict[str, Any]: Dictionary of arguments to pass to split() function.
        
    Command-line arguments:
        --xdf: Path to XDF file (required)
        --config-path: Path to configuration JSON file
        --ssh-tunnel: Flag to enable SSH tunneling
        --task-device-map: Path to YAML task-device mapping file
        --hdf5-corrections: Path to YAML corrections specification
    """
    parser = argparse.ArgumentParser(description='Split an XDF file into device-specific HDF5 files.')
    parser.add_argument(
        '--xdf',
        required=True,
        type=str,
        help="Path to the XDF file to split."
    )
    parser.add_argument(
        '--config-path',
        default=None,
        type=str,
        help="Specify a path to a Neurobooth configuration file with a 'database' entry."
    )
    parser.add_argument(
        '--ssh-tunnel',
        action='store_true',
        help=(
            "Specify this flag to SSH tunnel before connecting to the database. "
            "This is flag is not needed if running on the same machine as the database."
        )
    )
    parser.add_argument(
        '--task-device-map',
        type=str,
        default=None,
        help="If provided, the specified YAML file will be used to define a preset map of task ID -> device IDs."
    )
    parser.add_argument(
        '--hdf5-corrections',
        type=str,
        default=None,
        help="If provided, the specified YAML file will be used to locate correction functions for each device ID."
    )

    def abspath(path: Optional[str]) -> Optional[str]:
        return os.path.abspath(path) if path is not None else path

    args = parser.parse_args()

    # Process paths
    task_map_file = abspath(args.task_device_map)
    database_conn = DatabaseConnection(abspath(args.config_path), args.ssh_tunnel)
    corrections = abspath(args.hdf5_corrections)
    if corrections is not None:
        corrections = HDF5CorrectionSpec.load(corrections)

    return {
        'xdf_path': os.path.abspath(args.xdf),
        'database_conn': database_conn,
        'task_map_file': task_map_file,
        'corrections': corrections,
    }

def main() -> None:
    """
    Entry point for command-line execution.
    
    Parses arguments and calls the split() function.
    """
    split(**parse_arguments())

if __name__ == '__main__':
    main()