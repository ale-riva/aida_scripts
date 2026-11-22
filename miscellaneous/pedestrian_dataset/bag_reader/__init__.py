"""Read serialized bag messages through the MCAP and SQLite backends."""

import os

from rclpy.serialization import deserialize_message

from .Db3BagReader import BagFileParser, read_points as read_points_db3
from .McapBagReader import MCAPBagFileParser, read_points as read_points_mcap


def make_parser(bag_file: str):
    """Open the bag file with the parser for its storage extension."""
    extension = os.path.splitext(bag_file)[1].lower()
    if extension == ".mcap":
        return MCAPBagFileParser(bag_file)
    if extension == ".db3":
        return BagFileParser(bag_file)
    raise ValueError(f"Unsupported bag file extension: '{extension}'. Expected .mcap or .db3")


def read_points(cloud, field_names=None, skip_nans=False):
    """
    Unified read_points dispatcher (works with clouds decoded by either parser).
    """
    return read_points_mcap(cloud, field_names=field_names, skip_nans=skip_nans)


def iter_serialized_messages(parser, topics: list[str]):
    """Stream selected topics together without loading a bag into memory."""
    if hasattr(parser, "reader"):
        for schema, channel, record in parser.reader.iter_messages(topics=topics):
            yield channel.topic, record.log_time, schema.name, record.data
        return

    # Use row ID to keep equal recording timestamps in a stable order across passes.
    placeholders = ",".join("?" for _ in topics)
    rows = parser.conn.execute(
        "SELECT topics.name, messages.timestamp, topics.type, messages.data "
        "FROM messages JOIN topics ON messages.topic_id = topics.id "
        f"WHERE topics.name IN ({placeholders}) ORDER BY messages.timestamp, messages.rowid",
        tuple(topics),
    )
    yield from rows


def decode_message(parser, topic: str, type_name: str, payload):
    """Deserialize a payload with the bag backend's message type resolution."""
    if hasattr(parser, "reader"):
        message = parser._decode(topic, type_name, payload)
        if message is None:
            raise ValueError(f"Cannot decode message for topic {topic}")
        return message
    message_type = parser.topic_msg_message[topic]
    return deserialize_message(payload, message_type)
