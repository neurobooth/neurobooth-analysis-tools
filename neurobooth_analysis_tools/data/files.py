import os
import re
from functools import partial
from importlib import resources
from typing import NamedTuple, Tuple, List, Union, Optional
from datetime import datetime

from neurobooth_analysis_tools import data
from neurobooth_analysis_tools.data.types import NeuroboothTask, NeuroboothDevice

# Regex patterns
SUBJECT_YYYY_MM_DD = re.compile(r'(\d+)_(\d\d\d\d)[_-](\d\d)[_-](\d\d).*')
DATA_FILE_PATTERN = re.compile(r'(\d+)_(\d\d\d\d)-(\d\d)-(\d\d)_(\d\d)h-(\d\d)m-(\d\d)s_(.*)(\..+)')
NOTE_FILE_PATTERN = re.compile(r'(\d+)_(\d\d\d\d)-(\d\d)-(\d\d)-(.*)-notes\.txt')
TASK_DEVICE_PATTEN = re.compile(r'(' + r'|'.join([t.value for t in NeuroboothTask]) + r')_?(.*)')


class FilenameException(Exception):
    """Exception for unexpected Neurobooth file name pattern"""
    def __init__(self, *args):
        super(FilenameException, self).__init__(*args)


class FileMetadata(NamedTuple):
    session_path: str
    file_name: str
    subject_id: str
    datetime: datetime
    task: NeuroboothTask
    device: NeuroboothDevice
    device_info: str
    extension: str

    def _change_extension(self, new_extension: str) -> 'FileMetadata':
        """
        Create a similar metadata object with a different file extension.
        :param new_extension: The new file extension (including the ".").
        :return: A new FileMetadata object representing a simular file with a different extension.
        """
        base, _ = os.path.splitext(self.file_name)
        new_filename = base + new_extension
        return self._replace(
            file_name=new_filename,
            extension=new_extension,
        )


FILE_PATH = Union[str, FileMetadata]


def resolve_filename(file: FILE_PATH) -> str:
    """Load a neurobooth file and return its contents in a structured form."""
    if isinstance(file, FileMetadata):
        return os.path.join(file.session_path, file.file_name)
    elif isinstance(file, str):
        return file
    else:
        raise ValueError("Unsupported argument type.")


def is_valid_identifier(identifier: str, pattern: re.Pattern = SUBJECT_YYYY_MM_DD) -> bool:
    """Test if a string starts with a SUBJECT_YYYY-MM-DD pattern. (Both - and _ permitted between date fields.)"""
    matches = re.fullmatch(pattern, identifier)
    return matches is not None


def parse_session_id(identifier: str, pattern: re.Pattern = SUBJECT_YYYY_MM_DD) -> Tuple[str, datetime]:
    matches = re.fullmatch(pattern, identifier)
    if matches is None:
        raise ValueError(f'{identifier} did not match the expected format.')
    subject_num, year, month, day = matches.groups()

    date = datetime(int(year), int(month), int(day))
    return subject_num, date


def make_session_id_str(subject: Union[str, int], date: datetime) -> str:
    return f"{subject}_{date.strftime('%Y_%m_%d')}"


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


def parse_files(session_path: str) -> List[FileMetadata]:
    """Parse the file names present within a Neurobooth session directory."""
    metadata = []
    for f in os.listdir(session_path):
        if not os.path.isfile(os.path.join(session_path, f)):
            continue
        if re.fullmatch(NOTE_FILE_PATTERN, f) is not None:  # Do not handle notes at the moment
            continue
        if is_csv(f) or is_xdf(f) or is_log(f):  # Do not handle these files at the moment
            continue
        if '_jittered' in f:  # Jitter files should be ignored
            continue
        if is_tmp(f) or is_swp(f):  # Do not handle temporary files
            continue
        metadata.append(parse_file(session_path, f))
    return metadata


def parse_file(session_path: str, file_name: str) -> FileMetadata:
    """Parse the information from a single Neurobooth data file name"""
    match = re.fullmatch(DATA_FILE_PATTERN, file_name)
    if match is None:
        raise FilenameException(f"Could not parse {file_name}.")

    subj_id = match[1]
    year, month, day = int(match[2]), int(match[3]), int(match[4])
    hour, minute, second = int(match[5]), int(match[6]), int(match[7])
    dt = datetime(year, month, day, hour, minute, second)
    task_device_str = match[8]
    ext = match[9].lower()

    # Tease apart the task and device, which does not have a clear delimiter
    match = re.fullmatch(TASK_DEVICE_PATTEN, task_device_str)
    if match is None:
        raise FilenameException(f"No matching task definition found for {file_name}.")
    task = NeuroboothTask(match[1])
    device_str = match[2]
    device = file_str_to_device_enum(device_str, file_name)
    device_info = parse_device_info(device, device_str, file_name)

    return FileMetadata(
        session_path=session_path,
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

    if is_asc(file_name) or is_edf(file_name):
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
    if device == NeuroboothDevice.RealSense and is_video(file_name):
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
    return ext.lower() == extension.lower()


# Aliases for convenient file type checks
is_hdf5 = partial(has_extension, extension='.hdf5')
is_edf = partial(has_extension, extension='.edf')
is_asc = partial(has_extension, extension='.asc')
is_bag = partial(has_extension, extension='.bag')
is_avi = partial(has_extension, extension='.avi')
is_mov = partial(has_extension, extension='.mov')
is_xdf = partial(has_extension, extension='.xdf')
is_txt = partial(has_extension, extension='.txt')
is_json = partial(has_extension, extension='.json')
is_csv = partial(has_extension, extension='.csv')
is_log = partial(has_extension, extension='.log')
is_tmp = partial(has_extension, extension='.tmp')
is_swp = partial(has_extension, extension='.swp')


def is_video(file: str) -> bool:
    return is_bag(file) or is_avi(file) or is_mov(file)


def default_source_directories() -> List[str]:
    lines = resources.read_text(data, 'default_source_directories.txt').strip().splitlines(keepends=False)
    return [os.path.abspath(line) for line in lines]


def discover_associated_files(
        file: FileMetadata,
        extensions: Optional[List[str]] = None,
        use_device_info: bool = False,
) -> List[FILE_PATH]:
    """
    Get a list of data files all pertaining to the same session, task, and device.
    :param file: The file to compare to.
    :param extensions: Optional filter for file extensions. E.g., only return .json and .mov files.
    :param use_device_info: Whether to use the device_info field in comparisons.
        (E.g., to differentiate multiple devices of the same type).
    :return: A list of associated files matching the specified criteria.
    """
    def is_match(f: FileMetadata):
        match = (f.subject_id == file.subject_id)
        match &= (f.datetime == file.datetime)
        match &= (f.task == file.task)
        match &= (f.device == file.device)

        if use_device_info:
            match &= (f.device_info == file.device_info)

        if extensions is not None:
            match &= (f.extension in extensions)

        return match

    return [f for f in parse_files(file.session_path) if is_match(f)]
