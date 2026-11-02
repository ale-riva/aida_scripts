"""
Conversions between ROS 2 nanosecond timestamps (int) and the pipeline's
canonical filename format: YYYYMMDD_HHMMSS_ns

Example
  1760858578_225748974  ->  "20251019_112258_225748974"
  "20251019_112258_225748974"  ->  1760858578225748974
"""

from __future__ import annotations

import datetime
import re


# ---------------------------------------------------------------------------
# Format spec
# ---------------------------------------------------------------------------
# Full timestamp as nanoseconds since Unix epoch (UTC).
# Split as: seconds part + nanoseconds sub-second part.
# String format: YYYYMMDD_HHMMSS_<nanoseconds-sub-second, zero-padded to 9 digits>
#
# The 9-digit nanosecond field is the "sub-second" remainder only (0–999999999),
# so the full integer value is recoverable losslessly.
# ---------------------------------------------------------------------------

_NS_PER_SEC = 1_000_000_000
_TS_RE = re.compile(r"^(\d{8})_(\d{6})_(\d{9})$")


def ns_to_filename(timestamp_ns: int) -> str:
    """
    Convert a ROS 2 nanosecond timestamp to the canonical filename string.

    Parameters
    timestamp_ns: nanoseconds since Unix epoch (e.g. from message.publish_time).

    Returns: str (e.g. "20251019_112258_225748974")
    """
    sec  = timestamp_ns // _NS_PER_SEC
    nsub = timestamp_ns %  _NS_PER_SEC          # sub-second nanoseconds

    dt = datetime.datetime.utcfromtimestamp(sec)
    date_part = dt.strftime("%Y%m%d")
    time_part = dt.strftime("%H%M%S")

    return f"{date_part}_{time_part}_{nsub:09d}"


def filename_to_ns(filename_stem: str) -> int:
    """
    Recover the original nanosecond timestamp from a filename stem.

    Parameters
    filename_stem: the stem portion of the filename (no extension) (e.g. "20251019_112258_225748974").

    Returns: int (Nanoseconds since Unix epoch).
    """
    m = _TS_RE.match(filename_stem)
    if not m:
        raise ValueError(
            f"Cannot parse timestamp from filename stem: '{filename_stem}'. "
            f"Expected format YYYYMMDD_HHMMSS_<9-digit ns>"
        )

    date_str, time_str, nsub_str = m.group(1), m.group(2), m.group(3)

    dt = datetime.datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
    sec = int(dt.replace(tzinfo=datetime.timezone.utc).timestamp())

    return sec * _NS_PER_SEC + int(nsub_str)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Use the starting timestamp from metadata.yaml as ground truth
    original_ns = 1_760_858_578_225_748_974
    stem = ns_to_filename(original_ns)
    recovered = filename_to_ns(stem)

    print(f"Original  : {original_ns}")
    print(f"Filename  : {stem}")
    print(f"Recovered : {recovered}")
    assert original_ns == recovered, "Round-trip failed!"
    print("Round-trip OK")