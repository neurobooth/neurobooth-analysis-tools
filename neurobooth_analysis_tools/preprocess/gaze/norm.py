import numpy as np
from typing import Union, NamedTuple


class ScreenProperties(NamedTuple):
    width_px: int
    height_px: int
    px_per_inch: float  # Screen pixel density. Can be used to figure out the physical size of the screen.


def normalize_px_to_screen(gaze_pos: np.ndarray, screen: ScreenProperties) -> np.ndarray:
    """
    Normalize a gaze position time-series so that it is centered at the origin, the larger dimension has a range of
    [-1, 1], and the smaller dimension has a smaller range that preserves the screen aspect ratio.
    :param gaze_pos: The (N x 2) gaze position time-series (in screen pixels)
    :param screen: Properties of the screen, namely its width and height (in pixels)
    :return: The screen-normalized gaze position time-series.
    """
    gaze_pos = np.copy(gaze_pos)

    # Center the screen
    gaze_pos[:, 0] -= screen.width_px / 2
    gaze_pos[:, 1] -= screen.height_px / 2

    # Normalize so that the larger dimension is -1 to 1; preserve aspect ratio
    norm_factor = max(screen.width_px, screen.height_px) / 2
    gaze_pos /= norm_factor

    return gaze_pos


def normalize_max_value(gaze_pos: np.ndarray) -> np.ndarray:
    """
    Normalize a gaze position time-series so that that maximum absolute value does not exceed 1.
    If the data are already within this range, no normalization occurs.
    :return: The normalized gaze position time-series.
    """
    norm_factor = np.max(np.abs(gaze_pos))
    norm_factor = max(norm_factor, 1)
    return gaze_pos / norm_factor


def pixel_to_dva(
        gaze_pos: np.ndarray,
        mm_to_target: Union[float, np.ndarray],
        screen: ScreenProperties,
        centered: bool = False,
) -> np.ndarray:
    """
    Convert a gaze time-series (in pixels) to degrees visual angle (dva).
    :param gaze_pos: The (N x 2) gaze position time-series (in screen pixels)
    :param mm_to_target: The distance between the screen and the bullseye target on the subject's forehead.
        Eyelink provides this in mm, so that is the assumed unit we use here.
    :param screen: Properties of the screen, namely its width and height (in pixels) and pixel density (in inches).
    :param centered: True if the position origin center of the screen.
        False if the position origin is the top-left of the screen. (Default: False)
    :return:  The (N x 2) gaze position time-series (in dva)
    """
    gaze_pos = np.copy(gaze_pos)

    if not centered:  # Center the screen if origin is top-left
        gaze_pos[:, 0] -= screen.width_px / 2
        gaze_pos[:, 1] -= screen.height_px / 2

    # Convert pixels to mm
    gaze_pos *= 25.4 / screen.px_per_inch

    # Convert to dva
    gaze_pos[:, 0] = np.arctan(gaze_pos[:, 0] / mm_to_target)
    gaze_pos[:, 1] = np.arctan(gaze_pos[:, 1] / mm_to_target)

    return gaze_pos


def normalize_dva_to_screen(
        gaze_pos: np.ndarray,
        mm_to_target: Union[float, np.ndarray],
        screen: ScreenProperties
) -> np.ndarray:
    """
    Normalize a gaze position time-series (in degrees visual angle) so that it is represented as a percentage of the
    screen half-size (with respect to the larger dimension).
    :param gaze_pos: The (N x 2) gaze position time-series (in dva; origin at the center of the screen)
    :param mm_to_target: The distance between the screen and the bullseye target on the subject's forehead.
        Eyelink provides this in mm, so that is the assumed unit we use here.
    :param screen: Properties of the screen, namely its width and height (in pixels) and pixel density (in inches).
    :return: The screen-normalized gaze position time-series.
    """
    # Normalize so that the larger dimension is -1 to 1; preserve aspect ratio
    norm_factor_px = max(screen.width_px, screen.height_px) / 2
    norm_factor_mm = norm_factor_px * (25.4 / screen.px_per_inch)  # Convert pixels to mm
    norm_factor = np.arctan(norm_factor_mm / mm_to_target)  # Convert to dva
    return gaze_pos / norm_factor
