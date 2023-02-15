import numpy as np


def normalize_to_screen(gaze_pos: np.ndarray, screen_width: float, screen_height: float) -> np.ndarray:
    """
    Normalize a gaze position time-series so that it is centered at the origin, the larger dimension has a range of
    [-1, 1], and the smaller dimension has a smaller range that preserves the screen aspect ratio.
    :param gaze_pos: The (N x 2) gaze position time-series
    :param screen_width: The width of the screen in the same units as the position time-series
    :param screen_height: The height of the screen in the same units as the position time-series
    :return: The screen-normalized gaze position time-series.
    """
    gaze_pos = np.copy(gaze_pos)

    # Center the screen
    gaze_pos[:, 0] -= screen_width / 2
    gaze_pos[:, 1] -= screen_height / 2

    # Normalize so that the larger dimension is -1 to 1; preserve aspect ratio
    gaze_pos /= (max(screen_width, screen_height) / 2)

    return gaze_pos
