"""
This script is a wrapper around the resplit_xdf script in Neurobooth-OS.
It identifies all XDF files on the cluster and splits them in parallel.
It is advised to run the split script on a single file to make sure configurations are correct before running this
larger script.

Example (running on neurodoor no need of ssh tunnel):
conda activate neurobooth-os
cd /space/neo/3/neurobooth//applications/neurobooth-analysis-tools/neurobooth_analysis_tools/xdf
python resplit.py --config-path /space/drwho/3/neurobooth/applications/config/neurobooth_os_config.json \
--task-device-map /space/drwho/3/neurobooth/applications/neurobooth-os/examples/split_task_device_map.yml \
--hdf5-corrections /space/drwho/3/neurobooth/applications/neurobooth-os/examples/hdf5_corrections.yml --ssh-tunnel

specifying --ssh-tunnel can run it on the database

The respective paths of resplit.py, config-path, task-device-map and hdf5-corrections needs to be changed as per the to-date installation of 
neurobooth-os and neurobooth-analysis-tools. config-path is on the cluster and not on github repo.

Since the split can take a very long time, it may be wise to run this in the background with nohup.

"""


import os
import argparse
import datetime
import traceback
import time
from itertools import chain
from functools import partial
from typing import Dict, List, Any, Tuple
from tqdm.contrib.concurrent import process_map
import sshtunnel
from sshtunnel import SSHTunnelForwarder

import neurobooth_os.config as cfg
import neurobooth_os.iout.resplit_xdf as xdf
from neurobooth_analysis_tools.io import make_directory
from neurobooth_analysis_tools.data.files import (
    discover_session_directories,
    default_source_directories,
    is_xdf,
)

#added on January 9th 2025
from neurobooth_os.iout.resplit_xdf import (
    SplitException, DatabaseConnection, device_id_from_yaml,
    HDF5CorrectionSpec, XDFInfo, split
)

# Use presets for all XDF files on or before this date, when the parameter table was implemented.
LOG_DEVICE_PARAM_DATE = datetime.date(2024, 5, 10)

#this is new discovery of session directories
def new_discover_session_directories(data_dirs: List[str]) -> Tuple[List[str], List[str]]:
    """Discover a list of Neurobooth sessions from within the given data directories."""
    sessions = []
    session_dirs = []
    for d in data_dirs:
        if os.path.isdir(d) :
            session_dirs.append(d)
    return sessions, session_dirs

#discovering all the xdf files
def find_xdf(path: str) -> List[str]:
    """
    Find all files with a .xdf extension in a given directory.
    """
    return [
        os.path.join(path, file)
        for file in os.listdir(path)
        if is_xdf(file)
    ]

#---------------------- January 9th added ------------------------------
#Wrting new function 
def split_one_file(
    xdf_path: str,
    config_path: str,
    ssh_tunnel: bool,
    task_map_file: str,
    corrections: xdf.HDF5CorrectionSpec
) -> Tuple[xdf.XDFInfo,list]:
    """
    Helper function that:
    1) Finds the device IDs for the given XDF (from YAML or DB).
    2) Calls `split()` from split_xdf.py to do the splitting.
    3) Returns (xdf_info, device_data).

    We do NOT log to the DB here; that will happen later.
    """

    xdf_info = xdf.XDFInfo.parse_xdf_name(xdf_path)
    if task_map_file:
        device_ids = device_id_from_yaml(task_map_file, xdf_info.task_id)
    else:
        # db_conn = xdf.DatabaseConnection(config_path, ssh_tunnel)
        db_conn = xdf.DatabaseConnection(config_path, tunnel=False, override_host='localhost', override_port=6543)
        device_ids = db_conn.get_device_ids(xdf_info)
        db_conn.close()
    print(f"Trying to call split here on {xdf_path}")
    # db_connn = xdf.DatabaseConnection(config_path, ssh_tunnel)
    db_conn = xdf.DatabaseConnection(config_path, tunnel=False, override_host='localhost', override_port=6543)
    
    correction_spec = xdf.HDF5CorrectionSpec.load(corrections)
    #here we do the splitting - as per the updated split file in resplit_xdf.py
    xdf_info, dev_data = xdf.split(
        xdf_path=xdf_path,
        database_conn=db_conn,
        task_map_file=task_map_file,
        corrections=correction_spec,
    )
    # db_conn.close()
    return xdf_info, dev_data

def split_process(
        xdf_file: str,
        db_conn: xdf.DatabaseConnection,
        config_path: str, #commenting config_path and ssh_tunnel and using existing database connection
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

#this is the new database connection method 
def create_db_connection(config_path: str, tunnel: sshtunnel.SSHTunnelForwarder = None) -> xdf.DatabaseConnection:
    """
    Creates a DatabaseConnection with or without an SSH tunnel.
    """
    if tunnel:
        # Connect using the provided tunnel
        # Extract connection parameters from the tunnel
        with tunnel.get_connection() as conn:
            host = tunnel.local_bind_host
            port = tunnel.local_bind_port

            # Temporarily modify the database config with the tunnel's local bind address
            cfg.load_config(config_path, validate_paths=False)
            database_info = cfg.neurobooth_config.database
            database_info.host = tunnel.local_bind_host   # Update the host in the config
            database_info.port = tunnel.local_bind_port   # Update the port in the config

            # Create DatabaseConnection (it will now use the modified config)
            db_conn = xdf.DatabaseConnection(config_path, tunnel=True)

            # ... (any additional actions needed with db_conn) ...
            return db_conn
    else:
        # Connect without tunneling
        return xdf.DatabaseConnection(config_path, tunnel=False)

# 3) Split each file (possibly in parallel), but do NOT log yet
def process_file(xdf_path: str, config_path:str, ssh_tunnel:bool,task_map_file,correction_spec ):
    try:
        return split_one_file(
            xdf_path,
            config_path,
            ssh_tunnel=ssh_tunnel,
            task_map_file=task_map_file,
            corrections=correction_spec
        )
    except Exception as e:
        # Optionally handle or record errors
        print(f"[ERROR] Failed splitting {xdf_path}: {e}")
        traceback.print_exc()
        return None  # or raise if you want to stop

#this is to chunk the files into smaller files
def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def postgres_quote(s: str) -> str:
    # For simplicity, wrap each item in double quotes and escape internal quotes
    return '"' + s.replace('"', '\\"') + '"'

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
    # session_dirs = [session_dirs_all[50]]
    print(f"These is length of the session dirs {len(session_dirs)} and this is the content {session_dirs}")
    start_time_split=time.time()
    print(f"Starting the process at {start_time_split} with {session_dirs}")
    xdf_files_unfiltered = process_map(
        find_xdf, session_dirs, chunksize=10, desc='Finding XDF Files', unit='dir', max_workers=max_workers,
    )
    xdf_files_unfiltered = list(chain(*xdf_files_unfiltered))  # Flatten nested list
    #process 50 files
    # xdf_files_unfiltered = xdf_files_unfiltered[:5] 
    print(f"the xdf files are {xdf_files_unfiltered}")
    print(f"The number is {len(xdf_files_unfiltered)}")
    batch_size=10
    cfg.load_config(config_path, validate_paths=False)
    database_info = cfg.neurobooth_config.database

    # Establish SSH tunnel if required
    tunnel = None
    if ssh_tunnel:
        tunnel = SSHTunnelForwarder(
            ('neurodoor.nmr.mgh.harvard.edu', 22),
            ssh_username='dk028',  # Ensure this matches your config
            ssh_pkey=os.path.expanduser("/homes/9/dk028/.ssh/id_rsa"),
            remote_bind_address=('neurodoor.nmr.mgh.harvard.edu', 5432),
            local_bind_address=("localhost", 6543)
        )
        try:
            print(f"Starting the SSH tunnel to neurodoor.nmr.mgh.harvard.edu:5433")
            tunnel.start()
            print(f"--------*********SSH tunnel established on {tunnel.local_bind_host}:{tunnel.local_bind_port}")
        except Exception as e:
            print(f"[ERROR] Failed to establish SSH tunnel: {e}")
            traceback.print_exc()
            return

    #get the existing xdf files from the database
    processed_xdf=set()
    try:
        print("Connecting to the database and getting the processed xdf files")
        if tunnel and tunnel.is_active:
            db_conn=xdf.DatabaseConnection(config_path,tunnel=False,override_host=tunnel.local_bind_host, override_port=tunnel.local_bind_port)
        else:
            db_conn=xdf.DatabaseConnection(config_path,ssh_tunnel)

        with db_conn.connection.cursor() as cursor:
            cursor.execute("SELECT DISTINCT xdf_path FROM log_split")
            results=cursor.fetchall()
            processed_xdf={row[0] for row in results if row and row[0]}
        # db_conn.close()
        print(f"Found {len(processed_xdf)} XDF files that are already processed in the database")
    except Exception as e:
        print(f"[ERROR] Could not run the query on the database to obtain the xdf files")
        traceback.print_exc()
        if tunnel:
            tunnel.stop()
        return

    xdf_files=[
        f for f in xdf_files_unfiltered if os.path.abspath(f) not in processed_xdf
    ]

    
    print(f"These are the files that will be processed {xdf_files}")
    print(f"Discovered {len(xdf_files_unfiltered)} total xdf files")
    print(f"After filtering, will be processing only {len(xdf_files)} new files will be processed")

    # xdf_files=[]

    process_file_=partial(
        split_one_file,
        config_path=config_path,
        ssh_tunnel=True,
        task_map_file=task_map_file,
        corrections=correction_spec)

    for batch_index, batch_files in enumerate(chunk_list(xdf_files, batch_size), start=1):
        print(f"Processing batch {batch_index} with {len(batch_files)} files.")
        # For concurrency, use process_map or run sequentially if desired
        results = process_map(
            process_file_,
            batch_files,
            max_workers=max_workers,
            desc="Splitting XDF files"
        )

        results = [r for r in results if r is not None]
        print(f"Successfully processed {len(results)} XDF file(s).")

        

        print("Inserting log_split rows into DB ...")
        # db_conn = xdf.DatabaseConnection(config_path, ssh_tunnel)
        db_conn = xdf.DatabaseConnection(config_path, tunnel=False, override_host='localhost', override_port=6543)

        #commenting the database write temporarily
        
        for (xdf_info, device_data) in results:
            try:
                # Insert an entry for each device
                # Implementation of log_split (on your side) might require
                # rewriting or referencing the older code. For example:
                if not device_data:
                    print("Will not be writing to db because of empty device_data")
                with db_conn.connection.cursor() as cursor:
                    for dev in device_data:
                        # This is an example of how you might do it:
                        # (This snippet is adapted from your original code, but
                        #  be sure to handle timestamps or sensor_file_paths carefully!)
                        # If you have a helper function, you can call that here instead.
                        import datetime as dt
                        import numpy as np

                        # Suppose your device data has timestamps in dev.device_data["time_stamps"]
                        # timestamps = dev.device_data.get("time_stamps", [])
                        timestamps = dev["timestamps"]
                        if len(timestamps) == 0:
                            continue

                        # Suppose you have a utility function to get an offset
                        # For demonstration, we do zero offset:
                        time_offset = 0.0
                        start_time = dt.datetime.fromtimestamp(
                            timestamps[0] + time_offset
                        ).strftime("%Y-%m-%d %H:%M:%S")
                        end_time = dt.datetime.fromtimestamp(
                            timestamps[-1] + time_offset
                        ).strftime("%Y-%m-%d %H:%M:%S")
                        temporal_resolution = 0.0
                        if len(timestamps) > 1:
                            temporal_resolution = 1.0 / np.median(np.diff(timestamps))

                        # Build sensor_file_paths if you need them
                        # For example, dev.video_files or dev.sensor_ids
                        # sensor_file_paths = ["placeholder_filename.dat"]
                        hdf5_folder, hdf5_file = os.path.split(dev["hdf5_path"])
                        sensor_file_paths = [hdf5_file, *dev["video_files"]]

                        # Prepend the session folder
                        _, session_folder = os.path.split(hdf5_folder)
                        sensor_file_paths = [f"{session_folder}/{f}" for f in sensor_file_paths]

                        # Build a valid array-literal string, quoting each path
                        quoted = [postgres_quote(path) for path in sensor_file_paths]
                        sensor_file_paths_str = '{' + ','.join(quoted) + '}'

                        # Then pass 'sensor_file_paths_str' into your query_params
                        # query_params['sensor_file_paths'] = sensor_file_paths_str

                        # Build query
                        query_params = {
                            'subject_id': xdf_info.subject_id,
                            'date': xdf_info.date.isoformat(),
                            'task_id': xdf_info.task_id,
                            'true_temporal_resolution': temporal_resolution,
                            'file_start_time': start_time,
                            'file_end_time': end_time,
                            'device_id': dev["device_id"],
                            'sensor_id': dev["sensor_ids"][0] if dev["sensor_ids"] else None,
                            'hdf5_file_path': dev["hdf5_path"],
                            'xdf_path': xdf_info.xdf_pathd,
                            'sensor_file_paths': sensor_file_paths_str
                        }

                        sql = """
                            INSERT INTO log_split (
                                subject_id,
                                date,
                                task_id,
                                true_temporal_resolution,
                                file_start_time,
                                file_end_time,
                                device_id,
                                sensor_id,
                                hdf5_file_path,
                                xdf_path,
                                sensor_file_path
                            ) VALUES (
                                %(subject_id)s,
                                %(date)s,
                                %(task_id)s,
                                %(true_temporal_resolution)s,
                                %(file_start_time)s,
                                %(file_end_time)s,
                                %(device_id)s,
                                %(sensor_id)s,
                                %(hdf5_file_path)s,
                                %(xdf_path)s,
                                %(sensor_file_paths)s
                            )
                        """
                        cursor.execute(sql, query_params)

                db_conn.connection.commit()
                if device_data:
                    print(f"Done with batch {batch_index}. Wrote to DB.")

            except Exception as e:
                print(f"[ERROR] log_split insert failed for {xdf_info.path}: {e}")
                traceback.print_exc()
        
    # db_conn.close()
    # Close the SSH tunnel if it was established
    if tunnel:
        tunnel.stop()
        print("SSH tunnel closed.")
    print("All done!")


    #Creating a tunnel in the main function
    # with sshtunnel.SSHTunnelForwarder(
    #     database_info.remote_host,
    #     ssh_username=database_info.remote_user,
    #     ssh_config_file="~/.ssh/config",
    #     ssh_pkey="~/.ssh/id_rsa",
    #     remote_bind_address=(database_info.host, database_info.port),
    #     local_bind_address=("localhost", 6543)
    # ) as tunnel:
    #     split_process_ = partial(
    #     split_process,
    #     config_path=config_path,
    #     ssh_tunnel=ssh_tunnel,
    #     task_map_file=task_map_file,
    #     correction_spec=correction_spec,
    #     log_file_dir=log_file_dir,
    #     )

    #------- THIS IS ORIGINAL 
    # # Perform the split for each file in parallel - this is without tunneling - commenting now
    # split_process_ = partial(
    #     split_process,
    #     config_path=config_path,
    #     ssh_tunnel=ssh_tunnel,
    #     task_map_file=task_map_file,
    #     correction_spec=correction_spec,
    #     log_file_dir=log_file_dir,
    # )
    #----------

    # error_flags = process_map(
    #         split_process_, 
    #         [(xdf_file, create_db_connection(config_path, tunnel)) for xdf_file in xdf_files],  # Pass connection object
    #         chunksize=1, 
    #         desc='Spitting XDF', 
    #         unit='file', 
    #         max_workers=max_workers,
    #     )

    #------- THIS IS ORIGINAL
    #this is the one with the parallel processing
    # error_flags = process_map(
    #     split_process_, xdf_files, chunksize=1, desc='Spitting XDF', unit='file', max_workers=max_workers,
    # )
    #--------------


    end_time_split=time.time()
    print(f"Ending the process at {end_time_split} with {len(session_dirs)}")
    elapsed_time = end_time_split - start_time_split  # Calculate the elapsed time

    # print(f"There were {sum(error_flags)} errors encountered during the split process and the time taken is {elapsed_time}.")
    print(f" time taken is {elapsed_time}.")


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
