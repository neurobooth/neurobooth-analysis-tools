import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from typing import NamedTuple

from data.sql.queries import QUERY_CLIN_DEMOG_SCALE


class DatabaseConnectionInfo(NamedTuple):
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