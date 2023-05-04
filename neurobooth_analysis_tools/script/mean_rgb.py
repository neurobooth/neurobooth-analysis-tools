"""
Script to process video files, synchronize them with correspond HDF5 files, and produce and HDF5 file containing the
mean color channel value in each frame of the video.
"""


import os
import argparse
from typing import List
import datetime
from tqdm.contrib.concurrent import process_map
from tqdm.auto import tqdm
from itertools import chain
from functools import partial

from neurobooth_analysis_tools.data.files import (
    discover_session_directories,
    parse_files,
    FileMetadata,
    discover_associated_files,
)
from neurobooth_analysis_tools.data.types import NeuroboothDevice, NeuroboothTask
from neurobooth_analysis_tools.script.file_util import (
    validate_source_directories,
    check_valid_directory,
    sibling_directory,
)
from neurobooth_analysis_tools.preprocess.video.bag import bag2avi
from neurobooth_analysis_tools.preprocess.video.mean_rgb import mean_frame_rgb, mean_frame_rgb_realsense


def main() -> None:
    args = parse_arguments()
    hdf5_files = get_matching_files(args)

    # Parameterize the preprocessing function
    preprocess_video_ = partial(preprocess_video, dest=args.dest)

    # Run the preprocessing logic for each file in this process or in worker processes depending on CLI arguments
    if args.n_jobs == 1:
        map(preprocess_video_, tqdm(hdf5_files, desc='Extracting Mean RGB'))
    else:
        process_map(preprocess_video_, hdf5_files, desc='Extracting Mean RGB', chunksize=1, max_workers=args.n_jobs)


def get_matching_files(args: argparse.Namespace) -> List[FileMetadata]:
    """Create a list of metadata objects for data files that match conditions specified on the command line."""
    _, session_dirs = discover_session_directories(args.source)
    metadata = process_map(parse_files, session_dirs, desc="Parsing File Names", unit='sessions', chunksize=1)
    metadata = chain(*metadata)  # Flatten list of lists

    metadata = filter(lambda m: m.extension == '.hdf5', metadata)
    metadata = filter(lambda m: m.task == NeuroboothTask.TimingTest, metadata)
    metadata = filter(lambda m: m.device in args.devices, metadata)
    if args.subject is not None:
        metadata = filter(lambda m: m.subject_id == args.subject, metadata)
    if args.date is not None:
        metadata = filter(lambda m: m.datetime.date() == args.date, metadata)

    return list(metadata)  # Resolve filters


def preprocess_video(hdf5_file: FileMetadata, dest: str) -> None:
    # Discover video files associated with the HDF5 file
    bag_file = discover_associated_files(hdf5_file, extensions=['.bag'], use_device_info=True)
    bag_file = bag_file[0] if bag_file else None
    video_file = discover_associated_files(hdf5_file, extensions=['.mov', '.avi'], use_device_info=True)
    video_file = video_file[0] if video_file else None

    if bag_file is not None:  # If there is a bag file, extract its video and timestamp contents
        video_file, timestamp_file = convert_bag2avi(bag_file, dest)

    if video_file is None:  # Nothing to do...
        return

    output_hdf = hdf5_file._replace(
        session_path=os.path.join(dest, os.path.basename(hdf5_file.session_path)),
        file_name=hdf5_file.file_name.replace(".hdf5", "_RGB_frame_means.hdf5"),
    )

    if not os.path.exists(output_hdf.session_path):
        os.mkdir(output_hdf.session_path)

    # Finally, we actually process the video files to extract mean RGB from each frame
    if video_file.device == NeuroboothDevice.RealSense:
        mean_frame_rgb_realsense(video_file, timestamp_file, hdf5_file, output_hdf, progress_bar=False)
    else:
        mean_frame_rgb(video_file, hdf5_file, output_hdf, progress_bar=False)


def convert_bag2avi(bag_file: FileMetadata, dest: str) -> (FileMetadata, FileMetadata):
    # Figure out where to save the output video and timestamp files
    avi_file = bag_file._change_extension('.avi')._replace(
        session_path=os.path.join(dest, os.path.basename(bag_file.session_path))
    )
    npy_file = avi_file._change_extension('.npy')

    # Create the processed_data session directory if not present
    if not os.path.exists(avi_file.session_path):
        os.mkdir(avi_file.session_path)

    bag2avi(bag_file, avi_file, npy_file)  # All the heavy lifting happens here

    return avi_file, npy_file


def parse_arguments() -> argparse.Namespace:
    parser = configure_parser()
    args = parser.parse_args()
    validate_arguments(parser, args)
    return args


def configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process mean RGB time-series from video files.")
    add_general_opts_group(parser)
    add_directory_group(parser)
    add_session_group(parser)
    add_device_group(parser)
    return parser


def validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Perform input validation checks on command line arguments."""
    # Check that the source directories are valid
    args.source = validate_source_directories(parser, args.source)

    # If no destination directory, create a processed_data directory that is a sibling to the first source.
    if args.dest is None or not args.dest:
        args.dest = sibling_directory(args.source[0], 'processed_data', create=True)
    else:
        args.dest = os.path.abspath(args.dest)
    check_valid_directory(parser, args.dest)

    # Default to all devices if no filters were specified
    if args.devices is None:
        args.devices = [
            NeuroboothDevice.RealSense,
            NeuroboothDevice.FLIR,
            NeuroboothDevice.IPhone,
        ]


def add_general_opts_group(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(title="General Options")
    group.add_argument(
        '-p',
        dest='n_jobs',
        type=int,
        default=None,
        help="How many concurrent processes to run when processing data.",
    )


def add_directory_group(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(title="Data Directories")
    group.add_argument(
        '--dest',
        type=str,
        required=False,
        default=None,
        help="Destination directory of the preprocessed outputs."
    )
    group.add_argument(
        '--source',
        action='append',
        type=str,
        required=False,
        help="Overwrite source directory defaults. Specify multiple times for multiple source directories."
    )


def add_session_group(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        title="Session Filters",
        description="Only files satisfying the filter conditions will be processed."
    )
    group.add_argument(
        '--date',
        type=datetime.date.fromisoformat,
        default=None,
        help="Only process files for the given date."
    )
    group.add_argument(
        '--subject',
        type=str,
        default=None,
        help="Only process files for the given subject."
    )


def add_device_group(parser: argparse.ArgumentParser) -> None:
    group_description = (
        "If provided, only process video files belonging to the specified devices."
    )
    group = parser.add_argument_group(
        title="Device Filters",
        description=group_description
    )
    group.add_argument(
        '--real-sense',
        dest='devices',
        action='append_const',
        const=NeuroboothDevice.RealSense,
        help="Intel RealSense"
    )
    group.add_argument(
        '--iphone',
        dest='devices',
        action='append_const',
        const=NeuroboothDevice.IPhone,
        help="IPhone"
    )
    group.add_argument(
        '--flir',
        dest='devices',
        action='append_const',
        const=NeuroboothDevice.FLIR,
        help="FLIR Camera"
    )


if __name__ == '__main__':
    main()
