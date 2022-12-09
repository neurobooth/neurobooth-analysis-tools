import re
from datetime import datetime
from typing import Union


class PreprocessException(Exception):
    """Generic exception for issues occurring during data preprocessing."""
    def __init__(self, *args):
        super(PreprocessException, self).__init__(*args)


SUBJECT_YYYY_MM_DD = re.compile(r'(\d+)_(\d\d\d\d)_(\d\d)_(\d\d).*')


def is_valid_identifier(identifier: str, pattern: re.Pattern = SUBJECT_YYYY_MM_DD) -> bool:
    matches = re.fullmatch(pattern, identifier)
    return matches is not None


def parse_subject_date(identifier: str, pattern: re.Pattern = SUBJECT_YYYY_MM_DD) -> (str, datetime):
    matches = re.fullmatch(pattern, identifier)
    if matches is None:
        raise PreprocessException(f"{identifier} did not match the expected format.")
    subject_num, year, month, day = matches.groups()

    date = datetime(int(year), int(month), int(day))
    return subject_num, date


def make_identifier_string(subject: Union[str, int], date: datetime) -> str:
    return f"{subject}_{date.strftime('%Y_%m_%d')}"
