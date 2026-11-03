"""
Reads camera messages (CompressedImage or raw Image) from the BAG at
matched timestamps and saves them as .png files.

Output path: <output_base_dir>/<bag_stem>/samples/cameraN/<YYYYMMDD_HHMMSS_ns>.png

NOTE: cv_bridge is intentionally NOT used.
It was compiled against NumPy 1.x and segfaults under NumPy 2.x.
"""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np

from utils.timestamp_utils import ns_to_filename

logger = logging.getLogger(__name__)

_MS_TO_NS = 1_000_000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_camera_frame(
    parser,
    topic: str,
    timestamp_ns: int,
    out_path: str,
    tolerance_ms: float,
    compressed: bool = True,
    on_existing: str = "skip",
) -> bool:
    """
    Extract one camera frame from "parser" and write it to "out_path" as PNG.

    Parameters
    parser        : MCAPBagFileParser | BagFileParser
    topic         : ROS topic string for this camera
    timestamp_ns  : matched timestamp in nanoseconds
    out_path      : full path for the output .png file
    tolerance_ms  : fetch window half-width in milliseconds
    compressed    : True  -> sensor_msgs/CompressedImage
                    False -> sensor_msgs/Image
    on_existing   : "skip" | "overwrite" | "abort"

    Returns: bool  (True if written, False if skipped).
    """
    if os.path.exists(out_path):
        if on_existing == "skip":
            logger.debug("Skipping existing file: %s", out_path)
            return False
        elif on_existing == "abort":
            raise FileExistsError(f"Output file already exists: {out_path}")

    half_window_ns = int(tolerance_ms * _MS_TO_NS)

    messages = parser.get_messages(
        topics=topic,
        start_time=timestamp_ns - half_window_ns,
        end_time=timestamp_ns + half_window_ns,
    )

    if not messages:
        logger.warning("No message found for topic %s at ts=%d", topic, timestamp_ns)
        return False

    _, img_msg = messages[0]

    return write_camera_frame(img_msg, out_path, compressed, on_existing)


def write_camera_frame(img_msg, out_path: str, compressed: bool = True, on_existing: str = "skip") -> bool:
    """Decode an already-read camera message and save it as PNG."""
    if os.path.exists(out_path):
        if on_existing == "skip":
            logger.debug("Skipping existing file: %s", out_path)
            return False
        elif on_existing == "abort":
            raise FileExistsError(f"Output file already exists: {out_path}")

    cv_image = _decode_compressed(img_msg) if compressed else _decode_raw(img_msg)

    if cv_image is None:
        logger.warning("Failed to decode image for %s", out_path)
        return False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return bool(cv2.imwrite(out_path, cv_image))


def build_camera_output_path(output_base_dir: str, bag_stem: str, camera_key: str, timestamp_ns: int) -> str:
    """
    Return the canonical .png output path for a given camera sample.
    Example:  .../2025-10-19/samples/camera_0/20251019_112258_225748974.png
    """
    stem = ns_to_filename(timestamp_ns)
    return os.path.join(output_base_dir, bag_stem, "samples", camera_key, f"{stem}.png")


# ---------------------------------------------------------------------------
# Internal decoders
# ---------------------------------------------------------------------------

def _decode_compressed(msg) -> np.ndarray | None:
    """
    Decode a sensor_msgs/CompressedImage using cv2.imdecode.
    Works with JPEG and PNG compressed streams.
    """
    try:
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR)   # returns BGR uint8
        return image
    except Exception as e:
        logger.error("CompressedImage decode error: %s", e)
        return None


def _decode_raw(msg) -> np.ndarray | None:
    """
    Decode a sensor_msgs/Image.
    Supports bgr8, rgb8, mono8, and common bayer encodings.
    """
    try:
        encoding = msg.encoding.lower()
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)

        if encoding in ("bgr8", "rgb8"):
            image = buf.reshape((msg.height, msg.width, 3))
            if encoding == "rgb8":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image.copy()

        elif encoding == "mono8":
            image = buf.reshape((msg.height, msg.width))
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        elif encoding.startswith("bayer"):
            image = buf.reshape((msg.height, msg.width))
            bayer_codes = {
                "bayer_rggb8": cv2.COLOR_BayerRG2BGR,
                "bayer_bggr8": cv2.COLOR_BayerBG2BGR,
                "bayer_gbrg8": cv2.COLOR_BayerGB2BGR,
                "bayer_grbg8": cv2.COLOR_BayerGR2BGR,
            }
            code = bayer_codes.get(encoding, cv2.COLOR_BayerRG2BGR)
            return cv2.cvtColor(image, code)

        else:
            logger.warning("Unsupported image encoding '%s' - skipping", encoding)
            return None

    except Exception as e:
        logger.error("Raw Image decode error: %s", e)
        return None
