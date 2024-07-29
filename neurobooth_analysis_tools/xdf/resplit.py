"""
This script is a wrapper around the resplit_xdf script in Neurobooth-OS.
It identifies all XDF files on the cluster and splits them in parallel.
It is advised to run the split script on a single file to make sure configurations are correct before running this
larger script.

Example (running on neurodoor):
conda activate neurobooth-os
cd /space/neo/3/neurobooth//applications/neurobooth-analysis-tools/neurobooth_analysis_tools/xdf
python resplit.py --config-path /space/drwho/3/neurobooth/applications/config/neurobooth_os_config.json \
--task-device-map /space/drwho/3/neurobooth/applications/neurobooth-os/examples/split_task_device_map.yml \
--hdf5-corrections /space/drwho/3/neurobooth/applications/neurobooth-os/examples/hdf5_corrections.yml

Since the split can take a very long time, it may be wise to run this in the background with nohup.
"""


import os
import argparse
import datetime
import traceback
from itertools import chain
from functools import partial
from typing import Dict, List, Any
from tqdm.contrib.concurrent import process_map

import neurobooth_os.iout.resplit_xdf as xdf
from neurobooth_analysis_tools.io import make_directory
from neurobooth_analysis_tools.data.files import (
    discover_session_directories,
    default_source_directories,
    is_xdf,
)


# Use presets for all XDF files on or before this date, when the parameter table was implemented.
LOG_DEVICE_PARAM_DATE = datetime.date(2024, 5, 10)


def find_xdf(path: str) -> List[str]:
    """
    Find all files with a .xdf extension in a given directory.
    """
    return [
        os.path.join(path, file)
        for file in os.listdir(path)
        if is_xdf(file)
    ]


def split_process(
        xdf_file: str,
        config_path: str,
        ssh_tunnel: bool,
        task_map_file: str,
        correction_spec: str,
        log_file_dir: str,
) -> bool:
    """
    This is the function called in parallel processes to split each XDF file.
    :param xdf_file: The XDF file to split.
    :param config_path: The path to the JSON config containing database connection info.
    :param ssh_tunnel: Whether SSH tunneling is necessary.
    :param task_map_file: The path to the file specifying preset task-device mappings.
    :param correction_spec: The path specifying HDF5 correction functions.
    :param log_file_dir: The log file directory. Log files only get written if a SplitException is raised.
    :return: True if there was a SplitException, False otherwise.
    """
    # It is important we create a new connection in each process for thread safety!
    db_conn = xdf.DatabaseConnection(config_path, ssh_tunnel)
    correction_spec = xdf.HDF5CorrectionSpec.load(correction_spec)

    # We only provide the preset parameters if the file is from before the cutoff date!
    xdf_info = xdf.XDFInfo.parse_xdf_name(xdf_file)
    task_map_file = None if xdf_info.date > LOG_DEVICE_PARAM_DATE else task_map_file

    try:
        xdf.split(
            xdf_path=xdf_file,
            database_conn=db_conn,
            task_map_file=task_map_file,
            corrections=correction_spec,
        )
        return False
    except xdf.SplitException as e:
        with open(os.path.join(log_file_dir, f'{xdf_info.name}.log'), 'w') as f:
            f.write(str(e))
            f.write(traceback.format_exc())
        return True


def main(
        config_path: str,
        ssh_tunnel: bool,
        task_map_file: str,
        correction_spec: str,
        log_file_dir: str,
        max_workers: int,
) -> None:
    """
    Identify and split all XDF files in separate processes.
    :param config_path: The path to the JSON config containing database connection info.
    :param ssh_tunnel: Whether SSH tunneling is necessary.
    :param task_map_file: The path to the file specifying preset task-device mappings.
    :param correction_spec: The path specifying HDF5 correction functions.
    :param log_file_dir: The log file directory. Log files only get written if a SplitException is raised.
    :param max_workers: The number of parallel processes to employ.
    """
    # (Clear and) Create log file directory
    make_directory(log_file_dir, clear=True)

    # Identify all files to split
    _, session_dirs = discover_session_directories(default_source_directories())
    xdf_files = process_map(
        find_xdf, session_dirs, chunksize=10, desc='Finding XDF Files', unit='dir', max_workers=max_workers,
    )
    xdf_files = list(chain(*xdf_files))  # Flatten nested list

    # Perform the split for each file in parallel
    split_process_ = partial(
        split_process,
        config_path=config_path,
        ssh_tunnel=ssh_tunnel,
        task_map_file=task_map_file,
        correction_spec=correction_spec,
        log_file_dir=log_file_dir,
    )
    error_flags = process_map(
        split_process_, xdf_files, chunksize=1, desc='Spitting XDF', unit='file', max_workers=max_workers,
    )

    print(f"There were {sum(error_flags)} errors encountered during the split process.")


def parse_arguments() -> Dict[str, Any]:
    """
    Parse command line arguments.
    :return: Dictionary of keyword arguments to main().
    """
    parser = argparse.ArgumentParser(description='Resplit all XDF files on the Martinos cluster')
    parser.add_argument(
        '--config-path',
        required=True,
        type=str,
        help="Specify a path to a Neurobooth configuration file with a 'database' entry.",
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
        required=True,
        help="A YAML file will be used to define a preset map of task ID -> device IDs.",
    )
    parser.add_argument(
        '--hdf5-corrections',
        type=str,
        required=True,
        help="YAML file specifying correction functions for each device ID.",
    )
    parser.add_argument(
        '--log-file-dir',
        type=str,
        default='resplit_xdf_logs/',
        help="Specify a directory containing the output of each split function call.",
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=16,
        help="The number of parallel processes to run."
    )

    args = parser.parse_args()
    return {
        'config_path': os.path.abspath(args.config_path),
        'ssh_tunnel': args.ssh_tunnel,
        'task_map_file': os.path.abspath(args.task_device_map),
        'correction_spec': os.path.abspath(args.hdf5_corrections),
        'log_file_dir': os.path.abspath(args.log_file_dir),
        'max_workers': args.max_workers if args.max_workers > 0 else 1,
    }


if __name__ == '__main__':
    main(**parse_arguments())
