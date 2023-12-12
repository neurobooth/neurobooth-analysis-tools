"""
Shared data types.
"""

from enum import Enum, IntEnum


class NeuroboothTask(Enum):
    """Enumeration of valid Neurobooth Tasks"""
    Ahh = 'ahh_obs_1'
    LaLaLa = 'lalala_obs_1'
    MeMeMe = 'mememe_obs_1'
    GoGoGo = 'gogogo_obs_1'
    PaTaKa = 'pataka_obs_1'
    PassageReading = 'passage_obs_1'
    PassageDemo = 'passage_demo_obs_1'
    PassageTest = 'pursuit_test_obs_test'
    Ahh_NoIMU = 'noIMU_ahh_obs_1'
    Gogogo_NoIMU = 'noIMU_gogogo_obs_1'
    CoordPause = 'coord_pause_obs_1'
    CoordPause2 = 'coord_pause_obs_2'
    DSC = 'DSC_obs'
    MOT = 'MOT_obs_1'
    Mouse = 'mouse_obs'
    MouseDemo = 'mouse_demo_obs'
    Hevelius = 'hevelius_obs'
    Intro = 'intro_sess_obs_1'
    IntroOcculo = 'intro_occulo_obs_1'
    IntroCog = 'intro_cog_obs_1'
    Calibration = 'calibration_obs_1'
    FixationNoTarget = 'fixation_no_target_obs_1'
    GazeHolding = 'gaze_holding_obs_1'
    FixationTarget = 'fixation_target_obs_1'  # Alias for gaze holding
    IntroSpeech = 'intro_speech_obs_1'
    SaccadesHoriz = 'saccades_horizontal_obs_1'
    SaccadesVert = 'saccades_vertical_obs_1'
    SaccadesHorizTest = 'saccades_horizontal_obs_test'
    SmoothPursuit = 'pursuit_obs'
    SmoothPursuitTest = 'pursuit_test_obs_1'
    FingerNoseDemo = 'finger_nose_demo_obs_1'
    FingerNose = 'finger_nose_obs_1'
    FootTapping = 'foot_tapping_obs_1'
    AltHandMvmt = 'altern_hand_mov_obs_1'
    Clapping = 'clapping_test'
    Clapping2 = 'clapping_test_2'
    SitToStand = 'sit_to_stand_obs'
    TimingTest = 'timing_test_obs'
    TimingTest2 = 'timing_test_obs_2'
    DefaultTestTask = 'task_obs_1'


class NeuroboothDevice(IntEnum):
    """Enumeration of Neurobooth device types (e.g., IPhone, FLIR, EyeLink, etc...)"""
    IPhone = 0
    FLIR = 1
    RealSense = 2
    EyeLink = 3
    Yeti = 4
    Mbient = 5
    Mouse = 6


class DataException(Exception):
    """Exception for data-related errors."""
    def __init__(self, *args):
        super(DataException, self).__init__(*args)
