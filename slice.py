#! /usr/bin/env python

"""
Script to create a slice of Neurobooth data at the specified location.
The slice can be specified based on date ranges, data types, and tasks.
Because of storage concerns, video data is not currently supported as part of a slice.
"""


import os
import argparse
import datetime

from settings import Settings


def main() -> None:
    args = parse_arguments()
    settings = Settings(args.config)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a slice of Neurobooth data.")

    parser.add_argument(
        '--dest',
        type=str,
        required=True,
        help="Destination of the slice (on the local machine)."
    )
    parser.add_argument(
        '--config',
        type=str,
        default='./config.json',
        help="Override package configuration file."
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

    args = parser.parse_args()

    # Further validate and perform additional processing on arguments
    args.dest = os.path.abspath(args.dest)
    if not os.path.isdir(args.dest):
        raise argparse.ArgumentTypeError(f"{args.dest} is not a valid directory.")

    args.config = os.path.abspath(args.config)
    if not os.path.isfile(args.config):
        raise argparse.ArgumentTypeError(f"{args.config} is not a valid file.")

    return args


if __name__ == '__main__':
    main()
