"""
Code for parsing relevant events from ASCII files generated using edf2asc -nv -ns.
"""


from math import nan
from typing import NamedTuple, List, Literal
from functools import partial


class Span(NamedTuple):
    start: int
    end: int


class EyeEvents(NamedTuple):
    saccades: List[Span]
    fixations: List[Span]
    blinks: List[Span]


class EventData(NamedTuple):
    trial: Span
    left_eye: EyeEvents
    right_eye: EyeEvents


class ASCParseException(Exception):
    """Exception for issues occurring during ASC file parsing."""
    def __init__(self, *args):
        super(ASCParseException, self).__init__(*args)


def _parse_trial(file) -> Span:
    open = False
    start, end = nan, nan

    for line in file:
        msg = line.upper().split()
        if len(msg) == 0 or msg[0] != 'MSG':
            continue

        time = int(msg[1])
        if msg[2].lstrip('+-').isdigit():
            offset = int(msg[2])
            code = msg[3]
        else:
            offset = 0
            code = msg[2]

        if code == 'TASK_START':
            if open:
                raise ASCParseException("TASK_START improperly nested.")
            start = time - offset
            open = True
        elif code == 'TASK_END':
            if not open:
                raise ASCParseException("TASK_END occurred without TASK_START.")
            end = time - offset
            break

    return Span(start=start, end=end)


def _parse_eye_events_generic(file: List[str], eye: Literal['L', 'R'], end_code: str) -> List[Span]:
    events = []
    for line in file:
        msg = line.upper().split()
        if len(msg) == 0:
            continue

        # Don't bother parsing the event starts, as the event endings also include the start time
        if msg[0] == end_code and msg[1] == eye:
            events.append(Span(start=int(msg[2]), end=int(msg[3])))

    return events


_parse_saccades = partial(_parse_eye_events_generic, end_code='ESACC')
_parse_fixations = partial(_parse_eye_events_generic, end_code='EFIX')
_parse_blinks = partial(_parse_eye_events_generic, end_code='EBLINK')


def parse_asc_events(asc_file: str) -> EventData:
    """
    Parse events from an event-only EyeLink ASCII file.
    :param asc_file: The ASCII file to parse (generated with edf2asc -nv -ns)
    :return: The extracted eye events
    """
    with open(asc_file, 'r') as f:
        lines = f.readlines()

    # If files were larger, this block could be placed within the file context and passed the file object directly
    # instead of the lines. As is, reading the file once and iterating through RAM is probably more efficient.
    return EventData(
        trial=_parse_trial(lines),
        left_eye=EyeEvents(
            saccades=_parse_saccades(lines, 'L'),
            fixations=_parse_fixations(lines, 'L'),
            blinks=_parse_blinks(lines, 'L')
        ),
        right_eye=EyeEvents(
            saccades=_parse_saccades(lines, 'R'),
            fixations=_parse_fixations(lines, 'R'),
            blinks=_parse_blinks(lines, 'R')
        )
    )
