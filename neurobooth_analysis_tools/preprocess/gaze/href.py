"""
Code for working with HREF coordinates from the EyeLink Portable Duo.
See section 4.4.2.2 of the EyeLink Portable Duo manual.
"""

import numpy as np
from typing import Tuple


# Mathematical constants
F = 15000  # Distance of the HREF plane from the eye, this constant is defined in the manual
F_SQ = F ** 2
DEG2RAD = np.pi / 180
RAD2DEG = 180 / np.pi


# Typing aliases
Point = Tuple[float, float]


def calc_eye_velocity(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate instantaneous eye velocity (dva/s) from HREF coordinates.
    See section 4.4.2.2 of the EyeLink Portable Duo manual for the calculations.
    :param x: The horizontal eye position in HREF coordinates.
    :param y: The vertical eye position in HREF coordinates.
    :param t: The timestamp for each sample (s). NOTE: edf2asc produces timestamps in ms, so first divide by 1000.
    :return: (x, y): Eye velocity time series (dva/s).
    """
    numerator = DEG2RAD * (F_SQ + np.square(x) + np.square(y))
    x_res = numerator / np.sqrt(F_SQ + np.square(y))
    y_res = numerator / np.sqrt(F_SQ + np.square(x))

    dx = np.gradient(x, t) / x_res
    dy = np.gradient(y, t) / y_res

    return dx, dy


def calc_rotation_angle(p1: Point, p2: Point) -> float:
    """
    Calculate the eye rotation angle (degrees) between two points in HREF coordinates.
    See section 4.4.2.2 of the EyeLink Portable Duo manual for the calculations.
    :param p1: A pair of (x, y) HREF coordinates.
    :param p2: Another pair of (x, y) HREF coordinates.
    :return: The visual angle (degrees) between the two coordinate pairs.
    """
    x1, y1 = p1
    x2, y2 = p2

    numerator = F_SQ + (x1 * x2) + (y1 * y2)
    denominator = (F_SQ + x1**2 + y1**2) * (F_SQ + x2**2 + y2**2)
    denominator = np.sqrt(denominator)

    return RAD2DEG * np.arccos(numerator / denominator)
