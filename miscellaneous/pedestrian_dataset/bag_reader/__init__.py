"""
Thin adapter layer over the existing MCAPBagFileParser / BagFileParser.
Import from here rather than from the concrete files so the rest of the pipeline is insulated from the storage-format details.
"""

from .McapBagReader import MCAPBagFileParser, read_points as read_points_mcap
from .Db3BagReader  import BagFileParser,     read_points as read_points_db3

import os


def make_parser(bag_file: str):
    """
    Return the correct parser instance for "bag_file".
    """
    ext = os.path.splitext(bag_file)[1].lower()
    if ext == ".mcap":
        return MCAPBagFileParser(bag_file)
    elif ext == ".db3":
        return BagFileParser(bag_file)
    else:
        raise ValueError(f"Unsupported bag file extension: '{ext}'. Expected .mcap or .db3")


def read_points(cloud, field_names=None, skip_nans=False):
    """
    Unified read_points dispatcher (works with clouds decoded by either parser).
    """
    return read_points_mcap(cloud, field_names=field_names, skip_nans=skip_nans)