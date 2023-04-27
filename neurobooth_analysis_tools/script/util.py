import argparse
import os
from typing import Optional, List

from neurobooth_analysis_tools.data.files import default_source_directories


def check_valid_directory(parser: argparse.ArgumentParser, directory: str) -> None:
    """Raise a parser error if the given directory is not valid."""
    if not os.path.isdir(directory):
        parser.error(f"{directory} is not a valid directory.")


def validate_source_directories(parser: argparse.ArgumentParser, dirs: Optional[List[str]]) -> List[str]:
    """
    Ensure that each given source directory is valid.
    If no source directory was supplied, return the defaults.
    """
    if dirs is None or not dirs:
        return default_source_directories()

    dirs = [os.path.abspath(d) for d in dirs]
    for d in dirs:
        check_valid_directory(parser, d)
