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


def run_edf2asc(edf_file: str, *args: str, suffix: str = '', rm: bool = True) -> str:
    """
    Run the edf2asc CLI tool provided by SR Research in a subprocess.

    :param edf_file: The EDF file to process.
    :param args: Optional arguments to pass to the CLI.
    :param suffix: Rename to the generated asc file to include this suffix before the extension.
    :param rm: If True, delete the file first if it exists.
    :return: The path to resulting file.
    """
    base, _ = os.path.splitext(edf_file)
    asc_file = f'{base}.asc'

    if rm and os.path.exists(asc_file):
        os.remove(asc_file)

    subprocess.run(['edf2asc', *args, edf_file], capture_output=True)

    if suffix:
        asc_file_renamed = f'{base}{suffix}.asc'
        os.replace(asc_file, asc_file_renamed)
        return asc_file_renamed

    return asc_file


def extract_events_ascii(edf_file: str, rm: bool) -> str:
    """Extract events from an EDF file."""
    return run_edf2asc(edf_file, '-nv', '-ns', suffix='_events', rm=rm)


def extract_href_ascii(edf_file: str, rm: bool) -> str:
    """Extract HREF position and resolution from an EDF file."""
    return run_edf2asc(edf_file, '-t', '-sh', '-s', suffix='_href', rm=rm)


def extract_gaze_ascii(edf_file: str, rm: bool) -> str:
    """Extract HREF position and resolution from an EDF file."""
    return run_edf2asc(edf_file, '-t', '-s', '-res', suffix='_gaze', rm=rm)
