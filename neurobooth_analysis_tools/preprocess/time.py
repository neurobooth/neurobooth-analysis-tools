import numpy as np
from scipy.stats import linregress
from scipy.stats import t as t_dist
from typing import Union, Tuple


class SyncException(Exception):
    """Exception indicating a timestamp synchronization error."""
    def __init__(self, *args):
        super().__init__(*args)


def calc_timeseries_offset(
        ts1: np.ndarray,
        ts2: np.ndarray,
        slope_eps: float = 1e-3,
        confidence_interval: float = 0.95,
        intercept_eps: float = 5e-4,  # Corresponds to 0.5 ms if time-series are in seconds.
        return_half_width: bool = False,
) -> Union[float, Tuple[float, float]]:
    """
    Calculate the offset between two timestamp series using least-squares regression.
    Perform quality checks based on the slop and standard error of the intercept.
    :param ts1: The first timestamp series.
    :param ts2: The second timestamp series. Should have the same units and number of samples as ts1.
    :param slope_eps: Raise an exception if the slope between the series differs from 1 by more than this value.
    :param confidence_interval: Width of the confidence interval for the intercept. Defaults to 95% CI.
    :param intercept_eps: Raise an exception if the half-width of the CI of the intercept is larger than this value.
    :param return_half_width: Whether to return the half-width of the CI of the intercept.
    :return: The offset between ts1 and ts2. (I.e., the intercept of the regression line.)
        Optionally, the half-width of the CI of the intercept.
    """
    regression_result = linregress(ts1, ts2)

    if abs(1 - regression_result.slope) > slope_eps:
        raise SyncException("Slope between timestamp series is not near 1. Check for drift or outliers.")

    degrees_freedom = ts1.shape[0] - 2
    ci_half_width = abs(t_dist.ppf((1-confidence_interval)/2, degrees_freedom))
    ci_half_width *= regression_result.intercept_stderr
    if ci_half_width > intercept_eps:
        raise SyncException("The confidence window of the intercept between timestamp series is too large.")

    if return_half_width:
        return regression_result.intercept, ci_half_width
    else:
        return regression_result.intercept
