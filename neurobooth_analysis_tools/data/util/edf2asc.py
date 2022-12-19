"""
Wrapper for the command-line edf2asc utility to convert Eyelink EDF files into ASCII text.
"""


import os
import shutil
import subprocess


def check_edf2asc() -> None:
    """Check if the edf2asc CLI tool is installed."""
    if shutil.which('edf2asc') is None:
        raise Exception("The edf2asc command-line tool is not installed.")


def extract_events_ascii(edf_file: str, overwrite: bool = False) -> str:
    """Use the edf2asc CLI tool provided by SR Research to extract events from an EDF file."""
    base, _ = os.path.splitext(edf_file)
    asc_file = f'{base}.asc'

    if os.path.exists(asc_file):
        if overwrite:
            os.remove(asc_file)
        else:  # File already exists, and we do not want to overwrite -> just return without doing anything
            return asc_file

    subprocess.run(['edf2asc', '-nv', '-ns', edf_file], capture_output=True)
    return asc_file
