"""
This script is a wrapper around the resplit_xdf script in Neurobooth-OS.
It identifies all XDF files on the cluster and splits them in parallel.
It is advised to run the split script on a single file to make sure configurations are correct before running this
larger script.

Example (running on neurodoor no need to provide the ssh-tunnel flag, if running from other servers, specify ssh-tunnel):
conda activate neurobooth-os
cd /space/neo/3/neurobooth/applications/neurobooth-analysis-tools/neurobooth_analysis_tools/xdf
python resplit.py --config-path /space/drwho/3/neurobooth/applications/config/neurobooth_os_config.json --task-device-map /space/billnted/7/analyses/dk028/other_work/neurobooth-analysis-tools-dev/neurobooth-analysis-tools/neurobooth_analysis_tools/xdf/split_task_device_map.yml --hdf5-corrections /space/billnted/7/analyses/dk028/other_work/neurobooth-analysis-tools-dev/neurobooth-analysis-tools/neurobooth_analysis_tools/xdf/hdf5_corrections.yml --ssh-tunnel

Since the split can take a very long time, it may be wise to run this in the background with nohup.

This script:
    - Uses multiprocessing for parallel XDF file processing
    - Processes files in batches to enable incremental database logging
    - Default 16 workers suitable for cluster environments
    - Processing time depends on file sizes and device count
"""


import os
import argparse
import datetime
import traceback
import time
import sys
from itertools import chain
from functools import partial
from typing import Dict, List, Any, Tuple, Optional
from tqdm.contrib.concurrent import process_map
from sshtunnel import SSHTunnelForwarder
from shutil import rmtree


import resplit_xdf as xdf
import resplit_utils as nb_utils
from resplit_utils import (discover_session_directories,is_xdf,default_source_directories)


#Use preset YAML device mapping for all XDF files on or before this date
#After this date, the database will be queried for device configurations
LOG_DEVICE_PARAM_DATE = datetime.date(2024, 5, 10)

def make_directory(path: str, clear=False) -> None:
    """
    Create a directory, optionally clearing it first if it exists.
    
    Args:
        path (str): Path to the directory to create.
        clear (bool): If True and directory exists, remove it and all contents
                     before recreating. If False and directory exists, do nothing.
        
    """
    if os.path.exists(path):
        if clear:
            rmtree(path)
        else:
            return

    os.makedirs(path)

def find_xdf(path: str) -> List[str]:
    """
    Find all XDF files in a given directory (non-recursive).
    
    Args:
        path (str): Directory path to search.
        
    Returns:
        List[str]: Full paths to all .xdf files in the directory.
    """
    return [
        os.path.join(path, file)
        for file in os.listdir(path)
        if is_xdf(file)
    ]

def chunk_list(lst, chunk_size):
    """
    Split a list into chunks of specified size.
    
    Args:
        lst (List[Any]): List to split into chunks.
        chunk_size (int): Maximum size of each chunk.
        
    Yields:
        List[Any]: Successive chunks of the list.
        
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def postgres_quote(s: str) -> str:
    return '"' + s.replace('"', '\\"') + '"'

def split_one_file(
    xdf_path: str,
    config_path: str,
    ssh_tunnel: bool,
    task_map_file: str,
    corrections_path: str
) -> Optional[Tuple[xdf.XDFInfo, List[Dict[str, Any]]]]:
    """
    Worker function to split a single XDF file (designed for parallel execution).
    
    This function is called by process_map for each XDF file. It:
    1. Parses the XDF filename
    2. Determines if task mapping file should be used (based on date)
    3. Establishes database connection
    4. Calls the split function
    5. Returns results for database logging
    
    Args:
        xdf_path (str): Full path to the XDF file to process.
        config_path (str): Path to the database configuration JSON file.
        ssh_tunnel (bool): Whether SSH tunneling is required (currently overridden).
        task_map_file (str): Path to YAML task-device mapping file.
        corrections_path (str): Path to YAML corrections specification file.
        
    Returns:
        Tuple[XDFInfo, List[Dict]]: XDF metadata and slim device data for logging.
        or None if the processing failed

    """
    db_conn = None

    # Parse filename to get date
    xdf_info = xdf.XDFInfo.parse_xdf_name(xdf_path)

    # Use task map for old files, database query for newer ones
    task_map_file = None if xdf_info.date > LOG_DEVICE_PARAM_DATE else task_map_file

    #pass a host and port override to connect through it.
    
    try:
        # Establish database connection
        db_conn = xdf.DatabaseConnection(config_path, tunnel=False, override_host='localhost', override_port=6543)
        
        # Load correction specifications
        correction_spec = xdf.HDF5CorrectionSpec.load(corrections_path)
        
        # Split the XDF file
        xdf_info, dev_data = xdf.split(
            xdf_path=xdf_path,
            database_conn=db_conn,
            task_map_file=task_map_file,
            corrections=correction_spec,
        )
        return xdf_info, dev_data
    
    except Exception as e:
        # Log the error and return None to indicate failure
        print(f"[ERROR] Failed to process {xdf_path}: {e}")
        traceback.print_exc()
        return None
        
    finally:
        if db_conn:
            db_conn.close()

def main(
        config_path: str,
        ssh_tunnel: bool,
        task_map_file: str,
        correction_spec: str,
        log_file_dir: str,
        max_workers: int,
) -> None:
    """
    Main function to discover and process all XDF files in parallel.
    
    Processing workflow:
    1. Create/clear log directory
    2. Discover all session directories
    3. Find all XDF files in those directories
    4. Establish SSH tunnel to database (if needed)
    5. Query database for already-processed files
    6. Filter out processed files
    7. Process remaining files in parallel batches
    8. Log results to database after each batch
    
    Args:
        config_path (str): Path to database configuration JSON file.
        ssh_tunnel (bool): Whether to establish SSH tunnel for database access.
        task_map_file (str): Path to YAML task-device mapping file.
        correction_spec (str): Path to YAML corrections specification file.
        log_file_dir (str): Directory for log files (currently unused but created).
        max_workers (int): Number of parallel worker processes.
        
    Note:
        The function processes files in batches (default 10 per batch) to enable
        incremental database logging. This ensures progress is saved even if the
        script is interrupted.
        SSH tunnel (if used) is established once at the start and shared by all
        worker processes
    """

    # Set up log directory (cleared if exists)
    make_directory(log_file_dir, clear=True)

    # Discover all session directories
    _, session_dirs = discover_session_directories(default_source_directories())
    print(f"Found {len(session_dirs)} session directories.")
    
    # Find all XDF files in session directories
    xdf_files_unfiltered = process_map(
        find_xdf, session_dirs, chunksize=10, desc='Finding XDF Files', max_workers=max_workers,
    )
    xdf_files_unfiltered = list(chain(*xdf_files_unfiltered))
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

        # Query database for already-processed files
        processed_xdf = set()
        print("Querying database for already processed XDF files...")
        with db_conn.connection.cursor() as cursor:
            cursor.execute("SELECT DISTINCT xdf_path FROM log_split")
            processed_xdf = {row[0] for row in cursor.fetchall() if row and row[0]}
        print(f"Found {len(processed_xdf)} previously processed files.")

        # Filter out already-processed files
        xdf_files = [f for f in xdf_files_unfiltered if os.path.abspath(f) not in processed_xdf]
        print(f"After filtering, {len(xdf_files)} new files will be processed.")
        
        if not xdf_files:
            print("No new files to process. Exiting.")
            return

        # Prepare worker function with pre-filled arguments
        process_file_ = partial(
            split_one_file,
            config_path=config_path,
            ssh_tunnel=ssh_tunnel, 
            task_map_file=task_map_file,
            corrections_path=correction_spec)

        batch_size = 10 # Process in batches to log incrementally
        for i, batch_files in enumerate(chunk_list(xdf_files, batch_size), 1):
            print(f"\n--- Processing batch {i} of {len(batch_files)} files ---")
            
            # Process batch in parallel
            results = process_map(
                process_file_,
                batch_files,
                max_workers=max_workers,
                desc="Splitting XDF files"
            )
            # Filter out failed/empty results
            results = [r for r in results if r is not None and r[1]] 

            if not results:
                print("No files were successfully processed in this batch.")
                continue
            
            # Log results to database
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
        # Clean up resources
        if db_conn:
            db_conn.close()
        if tunnel:
            tunnel.stop()
            print("SSH tunnel closed.")
        print("All done!")


def parse_arguments() -> Dict[str, Any]:
    """
    Parse command-line arguments for the resplit script.
    
    Returns:
        Dict[str, Any]: Dictionary of keyword arguments for main().
        
    Command-line arguments:
        --config-path: Path to Neurobooth configuration JSON (required)
        --ssh-tunnel: Flag to enable SSH tunneling for database access
        --task-device-map: Path to YAML task-device mapping file (required)
        --hdf5-corrections: Path to YAML corrections specification (required)
        --log-file-dir: Directory for log files (default: resplit_xdf_logs/)
        --max-workers: Number of parallel worker processes (default: 16)
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

    # Parse arguments and run main function
    args = parse_arguments()
    main(**args)
    elapsed = time.time() - start_time

    # Report total execution time
    print(f"\nTotal script execution time: {elapsed:.2f} seconds.")