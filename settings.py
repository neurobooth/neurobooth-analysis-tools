"""
Functions to parse program configurations and secrets.
"""


import json
from typing import List


class Settings:
    raw_data: List[str]

    def __init__(self, config: str):
        with open(config, 'r') as f:
            self._config = json.load(f)

        self.raw_data = self._config['raw_data']
