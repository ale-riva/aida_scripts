"""
Thin adapter layer over the existing MCAPBagFileParser / BagFileParser.
Import from here rather than from the concrete files so the rest of the pipeline is insulated from the storage-format details.
"""

from .McapBagReader import MCAPBagFileParser, read_points as read_points_mcap
from .Db3BagReader  import BagFileParser,     read_points as read_points_db3

import os
from rclpy.serialization import deserialize_message


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


def iter_serialized_messages(parser, topics: list[str]):
    """Stream selected topics together without loading a bag into memory."""
    if hasattr(parser, "reader"):
        for schema, channel, record in parser.reader.iter_messages(topics=topics):
            yield channel.topic, record.log_time, schema.name, record.data
    else:
        placeholders = ",".join("?" for _ in topics)
        rows = parser.conn.execute(
            "SELECT topics.name,messages.timestamp,topics.type,messages.data "
            "FROM messages JOIN topics ON messages.topic_id=topics.id "
            f"WHERE topics.name IN ({placeholders}) ORDER BY messages.timestamp,messages.rowid",
            tuple(topics),
        )
        yield from rows


def decode_message(parser, topic: str, type_name: str, payload):
    if hasattr(parser, "reader"):
        message = parser._decode(topic, type_name, payload)
        if message is None:
            raise ValueError(f"Cannot decode message for topic {topic}")
        return message
    return deserialize_message(payload, parser.topic_msg_message[topic])
