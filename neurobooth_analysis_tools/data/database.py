import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from typing import NamedTuple, List

from neurobooth_analysis_tools.data.sql.queries import QUERY_CLIN_DEMOG_SCALE


class DatabaseConnectionInfo(NamedTuple):
    """Structure containing all the information necessary to connect to a database."""
    user: str
    password: str
    host: str
    port: int
    dbname: str

    def postgresql_url(self):
        return f'postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}'


class SubjectInfo:
    """Load and provide easy access to Neuropheno subject demographics and clinical information."""

    def __init__(self, connection_info: DatabaseConnectionInfo):
        """Connect to the database and execute a query to download and cache subject information"""
        engine = create_engine(connection_info.postgresql_url())

        date_columns = ['visit_date', 'demog_date', 'clin_date', 'scale_date']

        with engine.connect() as connection:
            self.data = pd.read_sql_query(
                QUERY_CLIN_DEMOG_SCALE,
                connection,
                parse_dates={c: {} for c in date_columns},
            ).convert_dtypes()

    def align_to_input(self, subjects: np.ndarray, dates: np.ndarray) -> pd.DataFrame:
        """Align the loaded data table to the order of the input."""
        subj_date = pd.DataFrame.from_dict({
            'subject_id': subjects,
            'visit_date': dates
        })
        return pd.merge(subj_date, self.data, how='left', on=('subject_id', 'visit_date'))


def fuzzy_join_date(
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        hard_on: List[str],
        fuzzy_on_left: str,
        fuzzy_on_right: str,
        offset_column_name: str = 'Offset_Days',
        **kwargs,
) -> pd.DataFrame:
    """
    Do a fuzzy join based on two date columns, where the closest match between the dates is selected.

    :param left_df: The left dataframe in the join
    :param right_df: The right dataframe in the join
    :param hard_on: Any non-fuzzy join conditions
    :param fuzzy_on_left: The date column to be used in the left dataframe
    :param fuzzy_on_right:  The date column to be used in the right dataframe
    :param offset_column_name:  The name of the column that will contain the calculated date offset
    :param kwargs: Any kwargs that should be passed on to the join (e.g., 'how' to specify join type)
    :return: The joined dataframe, with an added column for the separation of the joined dates.
    """
    possible_matches = pd.merge(left_df, right_df, on=hard_on, **kwargs)

    # Calculate number of days (signed) between each date column in the fuzzy join
    possible_matches[offset_column_name] = possible_matches[fuzzy_on_right] - possible_matches[fuzzy_on_left]
    possible_matches[offset_column_name] /= np.timedelta64(1, 'D')  # Convert to days

    # Rank possible matches based on proximity
    possible_matches['__Rank__'] = possible_matches[offset_column_name].abs()
    possible_matches['__Rank__'] = possible_matches \
        .groupby([*hard_on, fuzzy_on_left])['__Rank__'] \
        .rank(method='min', na_option='bottom')

    # Only keep the best matches (rank 1)
    mask = possible_matches['__Rank__'] == 1
    possible_matches = possible_matches.loc[mask]
    return possible_matches.drop(columns='__Rank__')  # No longer needed
