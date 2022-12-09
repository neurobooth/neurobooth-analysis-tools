import os
import re
from functools import partial
from typing import NamedTuple, Tuple, List
from enum import Enum
from datetime import datetime


class NeuroboothTask(Enum):
    Ahh = 'ahh_obs_1'
    Lalala = 'lalala_obs_1'
    Mememe = 'mememe_obs_1'
    Gogogo = 'gogogo_obs_1'
    Pataka = 'pataka_obs_1'
    # Ahh_NoIMU = 'noIMU_ahh_obs_1'
    # Gogogo_NoIMU = 'noIMU_gogogo_obs_1'
    # CoordPause = 'coord_pause_obs_1'
    # CoordPause2 = 'coord_pause_obs_2'
    # TimingTest = 'timing_test_obs'
    # TimingTest2 = 'timing_test_obs_2'
    DSC = 'DSC_obs'
    MOT = 'MOT_obs_1'
    # Mouse = 'mouse_obs'
    # MouseDemo = 'mouse_demo_obs'
    Hevelius = 'hevelius_obs'
    # Intro = 'intro_sess_obs_1'
    # IntroOcculo = 'intro_occulo_obs_1'
    # IntroCog = 'intro_cog_obs_1'
    Calibration = 'calibration_obs_1'
    FixationNoTarget = 'fixation_no_target_obs_1'
    GazeHolding = 'gaze_holding_obs_1'
    # FixationTarget = 'fixation_target_obs_1'  # Alias for gaze holding
    # IntroSpeech = 'intro_speech_obs_1'
    # FingerNoseDemo = 'finger_nose_demo_obs_1'
    FingerNose = 'finger_nose_obs_1'
    FootTapping = 'foot_tapping_obs_1'
    SaccadesHoriz = 'saccades_horizontal_obs_1'
    SaccadesVert = 'saccades_vertical_obs_1'
    # SaccadesHorizTest = 'saccades_horizontal_obs_test'
    SmoothPursuit = 'pursuit_obs'
    # SmoothPursuitTest = 'pursuit_test_obs_1'
    AltHandMvmt = 'altern_hand_mov_obs_1'
    # Clapping = 'clapping_test'
    # Clapping2 = 'clapping_test_2'
    Passage = 'passage_obs_1'
    # PassageDemo = 'passage_demo_obs_1'
    # PassageTest = 'pursuit_test_obs_test'
    SitToStand = 'sit_to_stand_obs'


# Regex patterns
SUBJECT_YYYY_MM_DD = re.compile(r'(\d+)_(\d\d\d\d)[_-](\d\d)[_-](\d\d).*')
NEUROBOOTH_DATA_FILE = re.compile(r'.*')  # Fill out


def is_valid_identifier(identifier: str, pattern: re.Pattern = SUBJECT_YYYY_MM_DD) -> bool:
    matches = re.fullmatch(pattern, identifier)
    return matches is not None


def discover_session_directories(data_dirs: List[str]) -> Tuple[List[str], List[str]]:
    sessions = []
    session_dirs = []
    for d in data_dirs:
        for session in os.listdir(d):
            session_path = os.path.join(d, session)
            if os.path.isdir(session_path) and is_valid_identifier(session):
                sessions.append(session)
                session_dirs.append(session_path)

    return sessions, session_dirs


class NeuroboothFileMetadata(NamedTuple):
    full_name: str
    time: datetime
    task: NeuroboothTask
    device: str
    extension: str


def parse_files(subj_dir: str) -> List[NeuroboothFileMetadata]:
    metadata = []

    for f in os.listdir(subj_dir):
        if os.path.isfile(f):  # and is not a note.
            break  # Make regex pattern and extract info

    return metadata


def has_extension(file: str, extension: str) -> bool:
    _, ext = os.path.splitext(file)
    return ext == extension


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
