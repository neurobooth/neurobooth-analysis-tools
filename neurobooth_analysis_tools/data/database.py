import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from typing import NamedTuple, List, Dict


class DatabaseConnectionInfo(NamedTuple):
    """Structure containing all the information necessary to connect to a database."""
    user: str
    password: str
    host: str
    port: int
    dbname: str

    def postgresql_url(self):
        return f'postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}'


class DatabaseConnection:
    """Load and provide easy access to Neuropheno subject demographics and clinical information."""
    session: pd.DataFrame
    demographic: pd.DataFrame
    clinical: pd.DataFrame
    scales: pd.DataFrame

    def __init__(self, connection_info: DatabaseConnectionInfo):
        """Connect to the database and execute a query to download and cache subject information"""
        self.connection_info = connection_info

    def download(self):
        """Download tables likely to be useful for analysis and fuzzy-join them with sessions by date."""
        tables = self.download_tables(
            'rc_visit_dates',
            'rc_demographic_clean',
            'rc_clinical_clean',
            'rc_ataxia_pd_scales_clean',
        )
        self.session = tables.pop('rc_visit_dates')
        session_view = self.session[['subject_id', 'redcap_event_name', 'neurobooth_visit_dates']]

        join_keys = {
            'rc_demographic_clean': 'end_time_demographic',
            'rc_clinical_clean': 'end_time_clinical',
            'rc_ataxia_pd_scales_clean': 'end_time_ataxia_pd_scales',
        }
        new_column_prefix = {
            'rc_demographic_clean': 'demographic',
            'rc_clinical_clean': 'clinical',
            'rc_ataxia_pd_scales_clean': 'scales',
        }

        tables = {
            name: fuzzy_join_date(
                session_view, table,
                hard_on=['subject_id'], fuzzy_on_left='neurobooth_visit_dates', fuzzy_on_right=join_keys[name],
                offset_column_name=f'{new_column_prefix[name]}_offset_days',
                how='left'
            )
            for name, table in tables.items()
        }
        self.demographic = tables['rc_demographic_clean']
        self.clinical = tables['rc_clinical_clean']
        self.scales = tables['rc_ataxia_pd_scales_clean']

    def download_tables(self, *table_names: str) -> Dict[str, pd.DataFrame]:
        """Download and return the specified tables from the database"""
        engine = create_engine(self.connection_info.postgresql_url())
        with engine.connect() as connection:
            return {
                table_name: pd.read_sql_table(table_name, connection).convert_dtypes()
                for table_name in table_names
            }


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
