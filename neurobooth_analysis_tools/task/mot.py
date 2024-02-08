"""
Task-specific processing for multiple object tracking (MOT).
"""

import re
from typing import NamedTuple, List, Iterator, Optional, Tuple
from enum import Enum, auto
import numpy as np
import pandas as pd
from neurobooth_analysis_tools.data import hdf5


# Regex Patterns for extracting information from the marker time-series
TRIAL_START = re.compile(r'(.*)Trial_start_(.*)')
TRIAL_END = re.compile(r'(.*)Trial_end_(.*)')
N_TARGET = re.compile(r'number targets:(\d+)_(.*)')
CLICK = re.compile(r'Response_start_(.*)')


class MOTTrial(NamedTuple):
    """Structured representation of marker information for an MOT Trial"""
    practice: bool
    start_time: float
    animation_end_time: float
    end_time: float
    n_targets: int
    circle_paths: pd.DataFrame
    click_times: np.ndarray


def parse_markers(marker: hdf5.DataGroup) -> List[MOTTrial]:
    """Parse the marker time-series and return structured information for each MOT trial."""
    return list(parse_markers_iter(marker.time_series, marker.time_stamps))


class ParserError(Exception):
    pass


class ParserState(Enum):
    BEGIN = auto()
    N_TARGET = auto()
    ANIMATION = auto()
    CLICKS = auto()
    COMPLETE = auto()


class ParserContext:
    """Stores marker information regarding a trial during parsing. Also handles the parsing logic and state machine."""

    RETURN_TYPE = Optional[MOTTrial]

    def __init__(self):
        self.state = ParserState.BEGIN
        self.practice = None
        self.start_time = None
        self.animation_end_time = None
        self.n_targets = None
        self.circle_id = []
        self.circle_x = []
        self.circle_y = []
        self.circle_ts = []
        self.click_times = []

    def consume(self, marker: str, ts: float) -> RETURN_TYPE:
        """
        Parse a single marker string and advance the parser state.

        The general structure of the marker time-series should be:
        ...
        Trial_start (or PracticeTrial_start)
        number targets
        !V TARGET_POS (for C circles, we get C of these every animation update.)
        ...
        !V TARGET_POS
        Response_start (for each click)
        Trial_end (or PracticeTrial_end)
        ...

        :param marker: The marker string to parse.
        :param ts: The associated LSL timestamp.
        :return: If parsing of the trial is complete, the result will be an MOTTrial object.
            Otherwise, the result will be None.
        """
        match self.state:
            case ParserState.BEGIN:
                return self.consume_begin(marker, ts)
            case ParserState.N_TARGET:
                return self.consume_n_targets(marker, ts)
            case ParserState.ANIMATION:
                return self.consume_animation(marker, ts)
            case ParserState.CLICKS:
                return self.consume_clicks(marker, ts)
            case _:
                raise ParserError(f'Encountered unexpected MOT marker parsing state: {self.state}')

    def consume_begin(self, marker: str, ts: float) -> RETURN_TYPE:
        """
        Consume markers until we encounter a trial start marker.
        """
        match = re.match(TRIAL_START, marker)
        if match is None:
            return None

        self.practice = 'practice' in match[1].lower()
        self.start_time = ts
        self.state = ParserState.N_TARGET
        return None

    def consume_n_targets(self, marker: str, ts: float) -> RETURN_TYPE:
        """
        Match a number of targets marker.
        """
        match = re.match(N_TARGET, marker)
        if match is None:
            raise ParserError(f'Expected to find No. Target marker. Instead found: {marker}')

        self.n_targets = int(match[1])
        self.state = ParserState.ANIMATION
        return None

    def consume_animation(self, marker: str, ts: float) -> RETURN_TYPE:
        """
        Consume target position markers until we encounter response marker.
        """
        match = re.match(hdf5.MARKER_POS_PATTERN, marker)
        if match is None:
            self.state = ParserState.CLICKS
            return self.consume_clicks(marker, ts)

        self.circle_id.append(int(match[1][1:]))
        self.circle_x.append(int(match[2]))
        self.circle_y.append(int(match[3]))
        self.circle_ts.append(ts)
        self.animation_end_time = ts
        return None

    def consume_clicks(self, marker: str, ts: float) -> RETURN_TYPE:
        """
        Consume response markers until we encounter a trial end marker.
        """
        match = re.match(CLICK, marker)
        if match is None:
            return self.complete(marker, ts)

        self.click_times.append(ts)
        return None

    def complete(self, marker: str, ts: float) -> RETURN_TYPE:
        """
        Format the context into an MOTTrial object and set the state machine to a completed state.
        """
        match = re.match(TRIAL_END, marker)
        if match is None:
            raise ParserError(f'Expected to find trial end. Instead found: {marker}')

        self.state = ParserState.COMPLETE
        return MOTTrial(
            practice=self.practice,
            start_time=self.start_time,
            animation_end_time=self.animation_end_time,
            end_time=ts,
            n_targets=self.n_targets,
            circle_paths=pd.DataFrame.from_dict({
                'MarkerTgt': self.circle_id,
                'MarkerX': self.circle_x,
                'MarkerY': self.circle_x,
                'Time_LSL': self.circle_ts,
            }),
            click_times=np.array(self.click_times),
        )


def parse_markers_iter(markers: np.ndarray, timestamps: np.ndarray) -> Iterator[MOTTrial]:
    """
    Parse the marker time-series and return an iterator over structured information for each MOT trial.
    :param markers: The series of marker strings for the task.
    :param timestamps: The associated LSL timestamps of each marker string.
    :return: An iterator over MOTTrial objects that aggregate trial information from across many markers.
    """
    context = ParserContext()  # Each context encapsulates "running" information while parsing over many markers
    for marker, ts in zip(markers, timestamps):
        result = context.consume(marker, ts)
        if result is not None:  # Parsing of the trial is complete
            yield result
            context = ParserContext()  # Get a new context for the next trial

