#! /usr/bin/env python

"""
Script to create a slice of Neurobooth data at the specified location.
The slice can be specified based on date ranges, data types, and tasks.
Because of storage concerns, video data is not currently supported as part of a slice.
"""


import os
import argparse
import datetime
from importlib import resources
from typing import List

import data
from data.files import discover_session_directories, parse_files


def main() -> None:
    args = parse_arguments()

    _, session_dirs = discover_session_directories(args.source)
    metadata = [parse_files(d) for d in session_dirs]

    if args.hdf5_only:
        pass  # TODO: apply filter

    for extension in args.exclude:
        pass  # TODO: apply filter

    # TODO: Handle dates

    # TODO: Add code to support device filters

    # TODO: Perform rsync


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a slice of Neurobooth data.")

    parser.add_argument(
        '--dest',
        type=str,
        required=True,
        help="Destination of the slice (on the local machine)."
    )
    parser.add_argument(
        '--source',
        action='append',
        type=str,
        required=False,
        help="Overwrite source directory defaults. Specify multiple times for multiple source directories."
    )
    parser.add_argument(
        '--start-date',
        type=datetime.date.fromisoformat,
        default=datetime.date.min,
        help="Beginning date of slice."
    )
    parser.add_argument(
        '--end-date',
        type=datetime.date.fromisoformat,
        default=datetime.date.max,
        help="Ending date of slice."
    )
    parser.add_argument(
        '--hdf5-only',
        action='store_true',
        help="Only include HDF5 files in the slice."
    )
    parser.add_argument(
        '--exclude',
        action='append',
        type=str,
        required=False,
        default=[],
        help="Exclude files with the given extension (including the .). Can specify multiple times"
    )

    args = parser.parse_args()
    validate_arguments(args)
    return args


def validate_arguments(args: argparse.Namespace) -> None:
    """Perform input validation checks on command line arguments."""
    # Check that the destination directory is valid
    args.dest = os.path.abspath(args.dest)
    check_valid_directory(args.dest)

    # Load default source directories if necessary, then check that each source directory is valid.
    if args.source is None:
        args.source = load_default_source_directories()
    else:
        args.source = [os.path.abspath(d) for d in args.source]
    for d in args.source:
        check_valid_directory(d)

    # Check that each extension exclusion starts with .
    for e in args.exclude:
        if not e.startswith('.'):
            raise argparse.ArgumentTypeError(f"{e} is not a valid file extension. Did you forget the leading .?")


def load_default_source_directories() -> List[str]:
    lines = resources.read_text(data, 'default_source_directories.txt').strip().splitlines(keepends=False)
    return [os.path.abspath(line) for line in lines]


def check_valid_directory(directory: str) -> None:
    if not os.path.isdir(directory):
        raise argparse.ArgumentTypeError(f"{directory} is not a valid directory.")


if __name__ == '__main__':
    main()
