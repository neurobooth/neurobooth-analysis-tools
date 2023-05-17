import numpy as np
import pandas as pd


def align_marker_to_series(marker: np.ndarray, marker_ts: np.ndarray, series_ts: np.ndarray) -> np.ndarray:
    """Create a marker time-series with samples aligned to the supplied timestamps.

    :param marker: The series of marker values. Should be a 2D MxD array for D marker dimensions
    :param marker_ts: The timestamps of the marker series. Should be a 1D length M array.
    :param series_ts: The timestamps of the series to align the marker to. Should be a 1D length N array.
    :return: A new 2D NxD marker series that is temporally aligned to series_ts.
    """
    N = series_ts.shape[0]
    M, D = marker.shape
    aligned_marker = np.full((N, D), np.nan)
    if M < 1:
        return aligned_marker

    if series_ts[-1] > marker_ts[-1]:  # Ensure last marker value persists until the end of the new series
        marker_ts = np.r_[marker_ts, series_ts[-1]]

    for i, (begin, end) in enumerate(zip(marker_ts[:-1], marker_ts[1:])):
        mask = (series_ts >= begin) & (series_ts < end)
        if not mask.any():
            continue
        aligned_marker[mask, :] = marker[i, :]

    return aligned_marker


def align_marker(marker_df: pd.DataFrame, series_df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper for align_marker_to_series that uses Time_LSL to align the marker series.

    :param marker_df: A marker dataframe (obtained from neurobooth_analysis_tools.data.hdf5).
    :param series_df: A device dataframe (obtained from neurobooth_analysis_tools.data.hdf5).
    :return: A new marker dataframe that is temporally aligned with and has the same number of rows as series_df.
    """
    marker = marker_df[['MarkerX', 'MarkerY']].to_numpy()
    marker_ts = marker_df['Time_LSL'].to_numpy()
    series_ts = series_df['Time_LSL'].to_numpy()

    aligned_marker_df = pd.DataFrame(
        data=align_marker_to_series(marker, marker_ts, series_ts),
        columns=['MarkerX', 'MarkerY'],
    )
    aligned_marker_df['Time_LSL'] = series_ts
    return aligned_marker_df
