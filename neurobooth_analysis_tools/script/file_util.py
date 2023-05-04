import argparse
import os
from typing import Optional, List

from neurobooth_analysis_tools.data.files import default_source_directories


def check_valid_directory(parser: argparse.ArgumentParser, directory: str) -> None:
    """Raise a parser error if the given directory is not valid."""
    if not os.path.isdir(directory):
        parser.error(f"{directory} is not a valid directory.")


def validate_directories(parser: argparse.ArgumentParser, dirs: List[str]) -> List[str]:
    """Ensure that each given source directory is valid and convert inputs to absolute paths."""
    dirs = [os.path.abspath(d) for d in dirs]
    for d in dirs:
        check_valid_directory(parser, d)
    return dirs


def validate_source_directories(parser: argparse.ArgumentParser, dirs: Optional[List[str]]) -> List[str]:
    """
    Ensure that each given source directory is valid.
    If no source directory was supplied, return the defaults.
    """
    if dirs is None or not dirs:
        return default_source_directories()
    return validate_directories(parser, dirs)


def sibling_directory(path: str, sibling_name: str, create: bool = False) -> str:
    """
    Replace the base directory with the provided new name.
    E.g., convert /X/Y/Z to /X/Y/W
    :param path: The directory to make a sibling for.
    :param sibling_name: The name of the sibling.
    :param create: Whether to create the sibling if it does not exist on the file system.
    :return: The sibling directory.
    """
    if os.path.isfile(path):
        raise ValueError(f'{dir} is not a directory.')

    root, basename = os.path.split(path)
    sibling = os.path.join(root, sibling_name)
    if create and not os.path.exists(sibling):
        os.mkdir(sibling)
    return sibling
