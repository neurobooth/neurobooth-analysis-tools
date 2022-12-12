import os
import re
from functools import partial
from typing import NamedTuple, Tuple, List
from datetime import datetime

from data.types import NeuroboothTask, NeuroboothDevice

# Regex patterns
SUBJECT_YYYY_MM_DD = re.compile(r'(\d+)_(\d\d\d\d)[_-](\d\d)[_-](\d\d).*')
DATA_FILE_PATTERN = re.compile(r'(\d+)_(\d\d\d\d)-(\d\d)-(\d\d)_(\d\d)h-(\d\d)m-(\d\d)s_(.*)(\..+)')
NOTE_FILE_PATTERN = re.compile(r'(\d+)_(\d\d\d\d)-(\d\d)-(\d\d)-(.*)-notes\.txt')
TASK_DEVICE_PATTEN = re.compile(r'(' + r'|'.join([t.value for t in NeuroboothTask]) + r')_?(.*)')


class FilenameException(Exception):
    """Exception for unexpected Neurobooth file name pattern"""
    def __init__(self, *args):
        super(FilenameException, self).__init__(*args)


def is_valid_identifier(identifier: str, pattern: re.Pattern = SUBJECT_YYYY_MM_DD) -> bool:
    """Test if a string starts with a SUBJECT_YYYY-MM-DD pattern. (Both - and _ permitted between date fields.)"""
    matches = re.fullmatch(pattern, identifier)
    return matches is not None


def discover_session_directories(data_dirs: List[str]) -> Tuple[List[str], List[str]]:
    """Discover a list of Neurobooth sessions from within the given data directories."""
    sessions = []
    session_dirs = []
    for d in data_dirs:
        for session in os.listdir(d):
            session_path = os.path.join(d, session)
            if os.path.isdir(session_path) and is_valid_identifier(session):
                sessions.append(session)
                session_dirs.append(session_path)

    return sessions, session_dirs


class FileMetadata(NamedTuple):
    session_dir: str
    file_name: str
    subject_id: str
    datetime: datetime
    task: NeuroboothTask
    device: NeuroboothDevice
    device_info: str
    extension: str


def parse_files(subj_dir: str) -> List[FileMetadata]:
    """Parse the file names present within a Neurobooth session directory."""
    metadata = []
    for f in os.listdir(subj_dir):
        if not os.path.isfile(os.path.join(subj_dir, f)):
            continue
        if re.fullmatch(NOTE_FILE_PATTERN, f) is not None:  # Do not handle notes at the moment
            continue
        if is_csv(f) or is_xdf(f):  # Do not handle csv or xdf files at the moment
            continue
        if '_jittered' in f:  # Jitter files should be ignored
            continue
        metadata.append(parse_file(subj_dir, f))
    return metadata


def parse_file(subj_dir: str, file_name: str) -> FileMetadata:
    """Parse the information from a single Neurobooth data file name"""
    match = re.fullmatch(DATA_FILE_PATTERN, file_name)
    subj_id = match.group(1)
    year, month, day = int(match.group(2)), int(match.group(3)), int(match.group(4))
    hour, minute, second = int(match.group(5)), int(match.group(6)), int(match.group(7))
    dt = datetime(year, month, day, hour, minute, second)
    task_device_str = match.group(8)
    ext = match.group(9).lower()

    # Tease apart the task and device, which does not have a clear delimiter
    match = re.fullmatch(TASK_DEVICE_PATTEN, task_device_str)
    if match is None:
        raise FilenameException(f"No matching task definition found for {file_name}.")
    task = NeuroboothTask(match.group(1))
    device_str = match.group(2)
    device = file_str_to_device_enum(device_str, file_name)
    device_info = parse_device_info(device, device_str, file_name)

    return FileMetadata(
        session_dir=subj_dir,
        file_name=file_name,
        subject_id=subj_id,
        datetime=dt,
        task=task,
        device=device,
        device_info=device_info,
        extension=ext,
    )


def file_str_to_device_enum(device_str: str, file_name: str) -> NeuroboothDevice:
    """Based on a device substring in a file name, figure out the related device."""
    device_str = device_str.lower()

    if device_str == '' and (is_asc(file_name) or is_edf(file_name)):
        return NeuroboothDevice.EyeLink
    elif 'eyelink' in device_str:
        return NeuroboothDevice.EyeLink
    elif 'iphone' in device_str:
        return NeuroboothDevice.IPhone
    elif 'flir' in device_str:
        return NeuroboothDevice.FLIR
    elif 'intel' in device_str:
        return NeuroboothDevice.RealSense
    elif 'yeti' in device_str:
        return NeuroboothDevice.Yeti
    elif 'mbient' in device_str:
        return NeuroboothDevice.Mbient
    elif 'mouse' in device_str:
        return NeuroboothDevice.Mouse
    else:
        raise FilenameException(f"Unable to parse device from {file_name}.")


def parse_device_info(device: NeuroboothDevice, device_str: str, file_name: str) -> str:
    """Parse supplementary device information from the data file name."""
    if device == NeuroboothDevice.RealSense and is_bag(file_name):
        if 'intel1' in device_str:
            return 'RealSense 1'
        elif 'intel2' in device_str:
            return 'RealSense 2'
        elif 'intel3' in device_str:
            return 'RealSense 3'
        else:
            raise FilenameException(f"Unable to parse supplemental device information from {file_name}.")
    elif device == NeuroboothDevice.RealSense and is_hdf5(file_name):
        if 'Intel_D455_1' in device_str:
            return 'RealSense 1'
        elif 'Intel_D455_2' in device_str:
            return 'RealSense 2'
        elif 'Intel_D455_3' in device_str:
            return 'RealSense 3'
        else:
            raise FilenameException(f"Unable to parse supplemental device information from {file_name}.")
    elif device == NeuroboothDevice.Mbient:
        if '_BK_' in device_str:
            return 'Back'
        elif '_LF_' in device_str:
            return 'Left Foot'
        elif '_RF_' in device_str:
            return 'Right Foot'
        elif '_LH_' in device_str:
            return 'Left Hand'
        elif '_RH_' in device_str:
            return 'Right Hand'
        else:
            raise FilenameException(f"Unable to parse supplemental device information from {file_name}.")
    else:
        return ''


def has_extension(file: str, extension: str) -> bool:
    _, ext = os.path.splitext(file)
    return ext == extension


# Aliases for convenient file type checks
is_hdf5 = partial(has_extension, extension='.hdf5')
is_edf = partial(has_extension, extension='.util')
is_asc = partial(has_extension, extension='.asc')
is_bag = partial(has_extension, extension='.bag')
is_avi = partial(has_extension, extension='.avi')
is_mov = partial(has_extension, extension='.mov')
is_xdf = partial(has_extension, extension='.xdf')
is_txt = partial(has_extension, extension='.txt')
is_json = partial(has_extension, extension='.json')
is_csv = partial(has_extension, extension='.csv')
