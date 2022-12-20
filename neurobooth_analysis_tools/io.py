import os
from shutil import rmtree


def make_directory(path: str, clear=False) -> None:
    if os.path.exists(path):
        if clear:
            rmtree(path)
        else:
            return

    os.makedirs(path)
