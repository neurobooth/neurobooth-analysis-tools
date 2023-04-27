"""
Script to process video files, synchronize them with correspond HDF5 files, and produce and HDF5 file containing the
mean color channel value in each frame of the video.
"""


import os
import argparse
from typing import List
from datetime import datetime
from tqdm.contrib.concurrent import process_map
from itertools import chain

from neurobooth_analysis_tools.data.files import (
    discover_session_directories,
    parse_files,
    FileMetadata,
)
from neurobooth_analysis_tools.data.types import NeuroboothDevice, NeuroboothTask
from neurobooth_analysis_tools.script.util import check_valid_directory, validate_source_directories


def main() -> None:
    args = parse_arguments()
    file_metadata = get_matching_files(args)
    print(file_metadata)
    print('WARNING: Nothing Done! Script implementation still in progress.')


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


def parse_arguments() -> argparse.Namespace:
    parser = configure_parser()
    args = parser.parse_args()
    validate_arguments(parser, args)
    return args


def configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process mean RGB time-series from video files.")
    add_directory_group(parser)
    add_session_group(parser)
    add_device_group(parser)
    return parser


def validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Perform input validation checks on command line arguments."""
    # Check that the source directories are valid
    args.source = validate_source_directories(parser, args.source)

    # TODO: If no destination directory, create a processed_data directory that is a sibling to the first source.
    args.dest = os.path.abspath(args.dest)
    check_valid_directory(parser, args.dest)

    # Default to all devices if no filters were specified
    if args.devices is None:
        args.devices = [
            NeuroboothDevice.RealSense,
            NeuroboothDevice.FLIR,
            NeuroboothDevice.IPhone,
        ]


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
