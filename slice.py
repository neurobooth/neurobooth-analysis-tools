#! /usr/bin/env python

"""
Script to create a slice of Neurobooth data at the specified location.
The slice can be specified based on date ranges, devices, and tasks.
Because of storage concerns, video data is not currently supported as part of a slice.
"""


import os
import argparse
import datetime
from importlib import resources
from typing import List

import data
from data.files import discover_session_directories, parse_files, FileMetadata
from data.types import NeuroboothDevice


def main() -> None:
    args = parse_arguments()
    file_metadata = get_matching_files(args)
    # TODO: Perform sync


def get_matching_files(args: argparse.Namespace) -> List[FileMetadata]:
    """Create a list of metadata objects for data files that match conditions specified on the command line."""
    metadata = []
    _, session_dirs = discover_session_directories(args.source)
    for d in session_dirs:
        metadata.extend(parse_files(d))

    if args.hdf5_only:
        metadata = filter(lambda m: m.extension == '.hdf5', metadata)

    if args.exclude:
        metadata = filter(lambda m: m.extension not in args.exclude, metadata)

    metadata = filter(lambda m: m.datetime >= args.start_date, metadata)
    metadata = filter(lambda m: m.datetime <= args.end_date, metadata)
    metadata = filter(lambda m: m.device in args.devices, metadata)
    # TODO: Implement task-based filters

    return list(metadata)  # Resolve filters


def parse_arguments() -> argparse.Namespace:
    parser = configure_parser()
    args = parser.parse_args()
    validate_arguments(parser, args)
    return args


def configure_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a slice of Neurobooth data.")
    add_directory_group(parser)
    add_filter_group(parser)
    add_device_group(parser)
    add_task_group(parser)
    return parser


def add_directory_group(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(title='Data Directories')
    group.add_argument(
        '--dest',
        type=str,
        required=True,
        help="Destination of the slice (on the local machine)."
    )
    group.add_argument(
        '--source',
        action='append',
        type=str,
        required=False,
        help="Overwrite source directory defaults. Specify multiple times for multiple source directories."
    )


def add_filter_group(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        title='General Filters',
        description='Only data files satisfying the filter conditions will be included in the slice.'
    )
    group.add_argument(
        '--start-date',
        type=datetime.date.fromisoformat,
        default=datetime.date.min,
        help="Beginning date of slice."
    )
    group.add_argument(
        '--end-date',
        type=datetime.date.fromisoformat,
        default=datetime.date.max,
        help="Ending date of slice."
    )
    group.add_argument(
        '--hdf5-only',
        action='store_true',
        help="Only include HDF5 files in the slice."
    )
    group.add_argument(
        '--exclude',
        action='append',
        type=str,
        required=False,
        default=[],
        help="Exclude files with the given extension (including the .). Can specify multiple times."
    )


def add_device_group(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        title='Device Flags',
        description='At least one device must be specified. Video devices currently not supported.'
    )
    group.add_argument(
        '--real-sense',
        dest='devices',
        action='append_const',
        const=NeuroboothDevice.RealSense,
        help='Intel RealSense Video (Not supported)'
    )
    group.add_argument(
        '--iphone',
        dest='devices',
        action='append_const',
        const=NeuroboothDevice.IPhone,
        help='IPhone Video/Audio (Not supported)'
    )
    group.add_argument(
        '--flir',
        dest='devices',
        action='append_const',
        const=NeuroboothDevice.FLIR,
        help='FLIR Camera Video (Not supported)'
    )
    group.add_argument(
        '--eyelink',
        dest='devices',
        action='append_const',
        const=NeuroboothDevice.EyeLink,
        help='EyeLink Gaze Data'
    )
    group.add_argument(
        '--yeti',
        dest='devices',
        action='append_const',
        const=NeuroboothDevice.Yeti,
        help='Yeti Mic Audio'
    )
    group.add_argument(
        '--mbient',
        dest='devices',
        action='append_const',
        const=NeuroboothDevice.Mbient,
        help='Mbient Inertial Data'
    )
    group.add_argument(
        '--mouse',
        dest='devices',
        action='append_const',
        const=NeuroboothDevice.Mouse,
        help='Mouse Position'
    )


def add_task_group(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        title='Task Flags',
        description='Specify a task flag to only include the specified tasks in the slice. Multiple flags can be set.'
    )
    # TODO: Add flags similarly to how the device flags function


def validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Perform input validation checks on command line arguments."""
    # Check that the destination directory is valid
    args.dest = os.path.abspath(args.dest)
    check_valid_directory(parser, args.dest)

    # Load default source directories if necessary, then check that each source directory is valid.
    if args.source is None:
        args.source = load_default_source_directories()
    else:
        args.source = [os.path.abspath(d) for d in args.source]
    for d in args.source:
        check_valid_directory(parser, d)

    # Check that each extension exclusion starts with .
    for e in args.exclude:
        if not e.startswith('.'):
            raise parser.error(f"{e} is not a valid file extension. Did you forget the leading .?")

    # Check that at least one device was specified
    if args.devices is None:
        raise parser.error(f"Must specify at least one device flag.")

    # Check that video is not included in the slice
    unsupported_devices = [NeuroboothDevice.FLIR, NeuroboothDevice.RealSense, NeuroboothDevice.IPhone]
    for d in args.devices:
        if d in unsupported_devices:
            raise parser.error(f"Unsupported Device: {d.name}. File sizes are too large for video slices.")


def load_default_source_directories() -> List[str]:
    lines = resources.read_text(data, 'default_source_directories.txt').strip().splitlines(keepends=False)
    return [os.path.abspath(line) for line in lines]


def check_valid_directory(parser: argparse.ArgumentParser, directory: str) -> None:
    if not os.path.isdir(directory):
        parser.error(f"{directory} is not a valid directory.")


if __name__ == '__main__':
    main()
