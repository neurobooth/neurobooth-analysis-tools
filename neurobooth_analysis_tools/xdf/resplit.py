"""
This script is a wrapper around the resplit_xdf script in Neurobooth-OS.
It identifies all XDF files on the cluster and splits them in parallel.
It is advised to run the split script on a single file to make sure configurations are correct before running this
larger script.

Example (running on neurodoor):
conda activate neurobooth-os
cd /space/neo/3/neurobooth/applications/neurobooth-analysis-tools/neurobooth_analysis_tools/xdf
python resplit.py --config-path /space/drwho/3/neurobooth/applications/config/neurobooth_os_config.json --task-device-map /space/billnted/7/analyses/dk028/other_work/neurobooth-analysis-tools-dev/neurobooth-analysis-tools/neurobooth_analysis_tools/xdf/split_task_device_map.yml --hdf5-corrections /space/billnted/7/analyses/dk028/other_work/neurobooth-analysis-tools-dev/neurobooth-analysis-tools/neurobooth_analysis_tools/xdf/hdf5_corrections.yml

Since the split can take a very long time, it may be wise to run this in the background with nohup.
"""


# resplit.py

import os
import argparse
import datetime
import traceback
import time
import sys
from itertools import chain
from functools import partial
from typing import Dict, List, Any, Tuple
from tqdm.contrib.concurrent import process_map
from sshtunnel import SSHTunnelForwarder
from shutil import rmtree
# script_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(script_dir)
# sys.path.insert(0, parent_dir)
# # import io
# # from data.files import (
# #     discover_session_directories,
# #     default_source_directories,
# #     is_xdf,
# # )

import resplit_xdf as xdf
import resplit_utils as nb_utils
from resplit_utils import (discover_session_directories,is_xdf,default_source_directories)
# import default_source_directories


#Use presets for all XDF files on or before this date
LOG_DEVICE_PARAM_DATE = datetime.date(2024, 5, 10)

def make_directory(path: str, clear=False) -> None:
    if os.path.exists(path):
        if clear:
            rmtree(path)
        else:
            return

    os.makedirs(path)

def new_discover_session_directories(data_dirs: List[str]) -> Tuple[List[str], List[str]]:
    """Discover a list of Neurobooth sessions from within the given data directories."""
    sessions = []
    session_dirs = []
    for d in data_dirs:
        if os.path.isdir(d) :
            session_dirs.append(d)
    return sessions, session_dirs

def find_xdf(path: str) -> List[str]:
    """Find all files with a .xdf extension in a given directory."""
    return [
        os.path.join(path, file)
        for file in os.listdir(path)
        if is_xdf(file)
    ]

def split_one_file(
    xdf_path: str,
    config_path: str,
    ssh_tunnel: bool,
    task_map_file: str,
    corrections_path: str
) -> Tuple[xdf.XDFInfo, list]:
    """
    Helper function that finds device IDs, calls split(), and returns results.
    """
    db_conn = None
    xdf_info = xdf.XDFInfo.parse_xdf_name(xdf_path)
    task_map_file = None if xdf_info.date > LOG_DEVICE_PARAM_DATE else task_map_file

    #pass a host and port override to connect through it.
    
    try:
        db_conn = xdf.DatabaseConnection(config_path, tunnel=False, override_host='localhost', override_port=6543)
        
        correction_spec = xdf.HDF5CorrectionSpec.load(corrections_path)
        
        xdf_info, dev_data = xdf.split(
            xdf_path=xdf_path,
            database_conn=db_conn,
            task_map_file=task_map_file,
            corrections=correction_spec,
        )
        return xdf_info, dev_data
    finally:
        if db_conn:
            db_conn.close()


def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def postgres_quote(s: str) -> str:
    return '"' + s.replace('"', '\\"') + '"'

def main(
        config_path: str,
        ssh_tunnel: bool,
        task_map_file: str,
        correction_spec: str,
        log_file_dir: str,
        max_workers: int,
) -> None:
    make_directory(log_file_dir, clear=True)

    # Find all XDF files
    _, session_dirs = discover_session_directories(default_source_directories())
    print(f"Found {len(session_dirs)} session directories.")
    
    xdf_files_unfiltered = process_map(
        find_xdf, session_dirs, chunksize=10, desc='Finding XDF Files', max_workers=max_workers,
    )
    xdf_files_unfiltered = list(chain(*xdf_files_unfiltered))
    # xdf_files_unfiltered = xdf_files_unfiltered[:2] # testing only two files
    print(f"Discovered {len(xdf_files_unfiltered)} total XDF files.")

    tunnel = None
    db_conn = None
    try:
        # Establish SSH tunnel if required
        if ssh_tunnel:
            config = nb_utils.load_config(config_path)
            remote_user = config.database.remote_user or os.environ.get('USER')
            remote_host = config.database.remote_host or 'neurodoor.nmr.mgh.harvard.edu'
            
            tunnel = SSHTunnelForwarder(
                (remote_host, 22),
                ssh_username=remote_user,
                ssh_pkey=os.path.expanduser(f"~/.ssh/id_rsa"),
                remote_bind_address=('neurodoor.nmr.mgh.harvard.edu', 5432),
                local_bind_address=("localhost", 6543)
            )
            print(f"Starting SSH tunnel...")
            tunnel.start()
            print(f"SSH tunnel established on {tunnel.local_bind_host}:{tunnel.local_bind_port}")
            # Connect through the tunnel to get existing files
            db_conn = xdf.DatabaseConnection(config_path, tunnel=False, override_host=tunnel.local_bind_host, override_port=tunnel.local_bind_port)
        else:
            # Connect directly
            db_conn = xdf.DatabaseConnection(config_path, tunnel=False)

        # Get existing XDF files from the database
        processed_xdf = set()
        print("Querying database for already processed XDF files...")
        with db_conn.connection.cursor() as cursor:
            cursor.execute("SELECT DISTINCT xdf_path FROM log_split")
            processed_xdf = {row[0] for row in cursor.fetchall() if row and row[0]}
        print(f"Found {len(processed_xdf)} previously processed files.")

        xdf_files = [f for f in xdf_files_unfiltered if os.path.abspath(f) not in processed_xdf]
        print(f"After filtering, {len(xdf_files)} new files will be processed.")
        
        if not xdf_files:
            print("No new files to process. Exiting.")
            return

        # Use partial to pre-fill arguments for the parallel worker function
        process_file_ = partial(
            split_one_file,
            config_path=config_path,
            ssh_tunnel=ssh_tunnel, 
            task_map_file=task_map_file,
            corrections_path=correction_spec)

        batch_size = 10 # Process in batches to log incrementally
        for i, batch_files in enumerate(chunk_list(xdf_files, batch_size), 1):
            print(f"\n--- Processing batch {i} of {len(batch_files)} files ---")
            
            results = process_map(
                process_file_,
                batch_files,
                max_workers=max_workers,
                desc="Splitting XDF files"
            )
            results = [r for r in results if r is not None and r[1]] 

            if not results:
                print("No files were successfully processed in this batch.")
                continue

            print(f"Successfully split {len(results)} files. Logging to database...")
            for (xdf_info, device_data) in results:
                try:
                    db_conn.log_split(xdf_info, device_data)
                except Exception as e:
                    print(f"[ERROR] Database log failed for {xdf_info.path}: {e}")
                    traceback.print_exc()
            print(f"Finished logging batch {i}.")

    except Exception as e:
        print(f"[ERROR] An error occurred in the main process: {e}")
        traceback.print_exc()
    finally:
        if db_conn:
            db_conn.close()
        if tunnel:
            tunnel.stop()
            print("SSH tunnel closed.")
        print("All done!")


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
    start_time = time.time()
    args = parse_arguments()
    main(**args)
    elapsed = time.time() - start_time
    print(f"\nTotal script execution time: {elapsed:.2f} seconds.")