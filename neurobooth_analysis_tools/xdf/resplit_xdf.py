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
from typing import NamedTuple, List, Dict, Optional, Any, Callable, ClassVar
import h5io
import yaml
import psycopg2 as pg
from pydantic import BaseModel


import resplit_utils as nb_utils

class SplitException(Exception):
    """For generic errors that occur when splitting an XDF file."""
    pass


class HDF5CorrectionSpec(BaseModel):
    marker: Optional[str] = None
    devices: Dict[str, str] = {}

    @staticmethod
    def load(path: str) -> 'HDF5CorrectionSpec':
        """
        Load the correction specification from a YAML configuration file.
        :param path: The path to the YAML file.
        :return: The correction specification.
        """
        try:
            with open(path, 'r') as stream:
                return HDF5CorrectionSpec(**yaml.safe_load(stream))
        except Exception as e:
            raise SplitException(f'Unable to load correction functions from {path}!') from e

    FUNC_STR_PATTERN: ClassVar = re.compile(r'(.*)\.py::(.*)\(\)')

    @staticmethod
    def import_function(func_str: str) -> Callable:
        """
        Import and return the function specified by a fully.qualified.module.py::func() string.
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
        Apply in-memory corrections to device data if corrections were specified for the given device/marker.
        :param device: The device structure loaded from the XDF file.
        :return: The corrected device structure.
        """
        if self.marker is not None:
            func = HDF5CorrectionSpec.import_function(self.marker)
            device = func(device)

        device_id = device.device_id
        if device_id in self.devices:
            func = HDF5CorrectionSpec.import_function(self.devices[device_id])
            device = func(device)

        return device


XDF_NAME_PATTERN = re.compile(r'(\d+)_(\d\d\d\d-\d\d-\d\d)_\d\dh-\d\dm-\d\ds_(.*)_R\d\d\d\.xdf', flags=re.IGNORECASE)


class XDFInfo(NamedTuple):
    """Structured representation of an XDF file name."""
    parent_dir: str
    name: str
    subject_id: str
    date: datetime.date
    task_id: str
    xdf_pathd: str

    @property
    def path(self) -> str:
        return os.path.join(self.parent_dir, self.name)

    @staticmethod
    def parse_xdf_name(xdf_path: str) -> 'XDFInfo':
        """
        Attempt to infer the subject ID, date, and task ID from the XDF file path.
        :param xdf_path: The path to the XDF file.
        :return: A structured representation of the XDF file name.
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
    """Handles limited interactions with the Neurobooth database"""

    def __init__(self, config_path: str, tunnel: bool, override_host: Optional[str] = None, override_port: Optional[int] = None):
        """
        Create a new DatabaseConnection based on the provided configuration file.
        """
        self.connection = self.connect(config_path, tunnel, override_host, override_port)

    @staticmethod
    def connect(config_path: str, tunnel: bool, override_host: Optional[str], override_port: Optional[int]) -> pg.extensions.connection:
        """
        Load and parse a configuration, then create a psycopg2 connection.
        """
        
        config = nb_utils.load_config(config_path)
        database_info = config.database

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
        Retrieve the list of device IDs associated with a given task and session.
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
        Create entries in the log_split table to reflect created HDF5 files.
        This version is updated to work with the "slim" dictionary format.
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
                
                hdf5_folder, hdf5_file = os.path.split(hdf5_path)
                _, session_folder = os.path.split(hdf5_folder)
                
                sensor_files = [f'{session_folder}/{hdf5_file}'] + [f'{session_folder}/{f}' for f in video_files]
                
                quoted = ['"' + s.replace('"', '\\"') + '"' for s in sensor_files]
                sensor_file_paths = '{' + ','.join(quoted) + '}'

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
        self.connection.commit()

    def log_split_old(self, xdf_info: XDFInfo, device_data: List[nb_utils.DeviceData]) -> None:
        """
        Create entries in the log_split table to reflect created HDF5 files.
        """
        
        with self.connection.cursor() as cursor:
            for device in device_data:
                
                time_offset = nb_utils.compute_clocks_diff()
                timestamps = device.device_data["time_stamps"]
                if len(timestamps) < 2:
                    continue

                start_time = datetime.datetime.fromtimestamp(timestamps[0] + time_offset).strftime("%Y-%m-%d %H:%M:%S")
                end_time = datetime.datetime.fromtimestamp(timestamps[-1] + time_offset).strftime("%Y-%m-%d %H:%M:%S")
                temporal_resolution = 1 / np.median(np.diff(timestamps))
                
                hdf5_folder, hdf5_file = os.path.split(device.hdf5_path)
                _, session_folder = os.path.split(hdf5_folder)
                
                # Correctly build the sensor file paths array
                sensor_files = [f'{session_folder}/{hdf5_file}'] + [f'{session_folder}/{f}' for f in device.video_files]
                sensor_file_paths = '{' + ', '.join(sensor_files) + '}'

                for sensor_id in device.sensor_ids:
                    query_params = {
                        'subject_id': xdf_info.subject_id,
                        'date': xdf_info.date.isoformat(),
                        'task_id': xdf_info.task_id,
                        'temporal_resolution': temporal_resolution,
                        'start_time': start_time,
                        'end_time': end_time,
                        'device_id': device.device_id,
                        'sensor_id': sensor_id,
                        'hdf5_file_path': f'{session_folder}/{hdf5_file}',
                        'xdf_path': xdf_info.xdf_pathd,
                        'sensor_file_paths': sensor_file_paths,
                    }
                    cursor.execute(
                        """
                        INSERT INTO log_split (subject_id, date, task_id, temporal_resolution, start_time, end_time, device_id, sensor_id, hdf5_file_path, xdf_path, sensor_file_paths)
                        VALUES (%(subject_id)s, %(date)s, %(task_id)s, %(temporal_resolution)s, %(start_time)s, %(end_time)s, %(device_id)s, %(sensor_id)s, %(hdf5_file_path)s, %(xdf_path)s, %(sensor_file_paths)s)
                        """,
                        query_params
                    )
        self.connection.commit()


def device_id_from_yaml(file: str, task_id: str) -> List[str]:
    """
    Load a YAML file defining preset task ID -> device ID mappings and look up the given task.
    """
    try:
        with open(file, 'r') as stream:
            task_device_map = yaml.safe_load(stream)
        return task_device_map[task_id]
    except Exception as e:
        raise SplitException(f'Could not locate task {task_id} using map file {file}.') from e

def _remake_hdf5_path(xdf_path: str, device_id: str, sensor_ids: List[str]) -> str:
    """
    Generate a path for a device HDF5 file extracted from an XDF file.
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
    Write the HDF5 files containing extracted device data.
    """
    new_device_data = []
    for dev in device_data:
        data_to_write = {"marker": dev.marker_data, "device_data": dev.device_data}
        new_hdf5_path = _remake_hdf5_path(xdf_path, dev.device_id, dev.sensor_ids)
        if os.path.exists(new_hdf5_path):
            print(f"HDF5 file {new_hdf5_path} already exists. Skipping.")
            continue

        new_dev = nb_utils.DeviceData(
            device_id=dev.device_id,
            device_data=dev.device_data,
            marker_data=dev.marker_data,
            video_files=dev.video_files,
            sensor_ids=dev.sensor_ids,
            hdf5_path=new_hdf5_path,
        )
        new_device_data.append(new_dev)
        h5io.write_hdf5(new_hdf5_path, data_to_write, overwrite=True)
    return new_device_data

def split(
        xdf_path: str,
        database_conn: DatabaseConnection,
        task_map_file: Optional[str] = None,
        corrections: Optional[HDF5CorrectionSpec] = None,
) -> None:
    """
    Split a single XDF file into device-specific HDF5 files.
    """
    xdf_info = XDFInfo.parse_xdf_name(xdf_path)

    if task_map_file is not None:
        device_ids = device_id_from_yaml(task_map_file, xdf_info.task_id)
    else:
        device_ids = database_conn.get_device_ids(xdf_info)
        print(f"Got from the database here {device_ids}")

    if not device_ids:
        raise SplitException(f'Could not locate task ID {xdf_info.task_id} for session {xdf_info.subject_id}_{xdf_info.date.isoformat()}.')

    try:
        device_data = nb_utils.parse_xdf(xdf_path, device_ids)
    except Exception as e:
        print(f"[ERROR] Error parsing XDF file {xdf_path}: {e}. Skipping file.")
        return xdf_info, []

    if corrections is not None:
        device_data = [corrections.correct_device(dev) for dev in device_data]
    
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
    Parse command line arguments.
    :return: Dictionary of keyword arguments to split().
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
    """Entry point for command-line calls."""
    split(**parse_arguments())

if __name__ == '__main__':
    main()