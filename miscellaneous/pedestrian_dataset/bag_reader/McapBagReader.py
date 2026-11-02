from pathlib import Path
import struct
import math
from typing import (
    IO,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Tuple,
)
from mcap.reader import make_reader
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import rosbag2_py

from sensor_msgs.msg import PointCloud2, PointField

_DATATYPES = {}
_DATATYPES[PointField.INT8]    = ('b', 1)
_DATATYPES[PointField.UINT8]   = ('B', 1)
_DATATYPES[PointField.INT16]   = ('h', 2)
_DATATYPES[PointField.UINT16]  = ('H', 2)
_DATATYPES[PointField.INT32]   = ('i', 4)
_DATATYPES[PointField.UINT32]  = ('I', 4)
_DATATYPES[PointField.FLOAT32] = ('f', 4)
_DATATYPES[PointField.FLOAT64] = ('d', 8)
######################################################################################################

def _get_struct_fmt(is_bigendian, fields, field_names=None):
    fmt = '>' if is_bigendian else '<'

    offset = 0
    for field in (f for f in sorted(fields, key=lambda f: f.offset) if field_names is None or f.name in field_names):
        if offset < field.offset:
            fmt += 'x' * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print('Skipping unknown PointField datatype [%d]' % field.datatype, file=sys.stderr)
        else:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt    += field.count * datatype_fmt
            offset += field.count * datatype_length

    return fmt

def read_points(cloud, field_names=None, skip_nans=False, uvs=[]):
    """
    Read points from a L{sensor_msgs.PointCloud2} message.

    @param cloud: The point cloud to read from.
    @type  cloud: L{sensor_msgs.PointCloud2}
    @param field_names: The names of fields to read. If None, read all fields. [default: None]
    @type  field_names: iterable
    @param skip_nans: If True, then don't return any point with a NaN value.
    @type  skip_nans: bool [default: False]
    @param uvs: If specified, then only return the points at the given coordinates. [default: empty list]
    @type  uvs: iterable
    @return: Generator which yields a list of values for each point.
    @rtype:  generator
    """
    # assert isinstance(cloud, PointCloud2), 'cloud is not a sensor_msgs.msg.PointCloud2'
    fmt = _get_struct_fmt(cloud.is_bigendian, cloud.fields, field_names)
    width, height, point_step, row_step, data, isnan = cloud.width, cloud.height, cloud.point_step, cloud.row_step, cloud.data, math.isnan
    unpack_from = struct.Struct(fmt).unpack_from

    if skip_nans:
        if uvs:
            for u, v in uvs:
                p = unpack_from(data, (row_step * v) + (point_step * u))
                has_nan = False
                for pv in p:
                    if isnan(pv):
                        has_nan = True
                        break
                if not has_nan:
                    yield p
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    p = unpack_from(data, offset)
                    has_nan = False
                    for pv in p:
                        if isnan(pv):
                            has_nan = True
                            break
                    if not has_nan:
                        yield p
                    offset += point_step
    else:
        if uvs:
            for u, v in uvs:
                yield unpack_from(data, (row_step * v) + (point_step * u))
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    yield unpack_from(data, offset)
                    offset += point_step



#######################################################################################################
class MCAPBagFileParser:
    """
    Minimal MCAP analog of your sqlite-backed BagFileParser.

    Notes:
      * Pass the path to a single `.mcap` file (not a directory).
      * Timestamps are returned as MCAP log_time nanoseconds (int).
      * Message decoding uses ROS 2 CDR -> Python via rosidl/rclpy, same as your db3 parser.
    """

    def __init__(self, mcap_file: str):
        self._path = Path(mcap_file)
        if not self._path.exists():
            raise FileNotFoundError(f"No such file: {self._path}")
        if self._path.is_dir():
            raise ValueError(f"Expected an .mcap file, got directory: {self._path}")
        # open once and keep a seeking reader around (efficient, supports multiple iterations)
        self._fh = open(self._path, "rb")
        self.reader = make_reader(self._fh)

        # Public attributes to mirror your sqlite-based parser
        self.topic_type = {}         # {topic_name: "pkg/msg/Type"}
        self.topic_id = {}           # {topic_name: channel_id (int)}
        self.topic_msg_message = {}  # {topic_name: Python message class}

        # Prefer building topic/type map from the summary section if present
        try:
            summary = self.reader.get_summary()
        except Exception:
            summary = None

        # if summary is not None:
        #     # summary.channels: {channel_id -> Channel}; summary.schemas: {schema_id -> Schema}
        #     for ch_id, ch in summary.channels.items():
        #         topic = ch.topic
        #         self.topic_id[topic] = ch_id
        #         schema = summary.schemas.get(ch.schema_id)
        #         if schema and schema.name:
        #             type_str = schema.name  # e.g. "sensor_msgs/msg/PointCloud2"
        #             self.topic_type[topic] = type_str
        #             try:
        #                 self.topic_msg_message[topic] = get_message(type_str)
        #             except Exception as e:
        #                 print(f"An error occurred: {e}")
        #                 print("Message: " + type_str)

    def __del__(self):
        try:
            self._fh.close()
        except Exception:
            pass

    # ---------- Internal helpers ----------

    def _ensure_msg_type_cached(self, topic_name: str, schema_name: str | None):
        """
        Lazily populate self.topic_type and self.topic_msg_message if we discover a topic
        while iterating (useful when no summary section is present).
        """
        if schema_name and topic_name not in self.topic_msg_message:
            self.topic_type[topic_name] = schema_name
            try:
                self.topic_msg_message[topic_name] = get_message(schema_name)
            except Exception as e:
                print(f"An error occurred: {e}")
                print("Message: " + schema_name)

    def _decode(self, topic_name: str, schema_name: str | None, raw_bytes: bytes):
        self._ensure_msg_type_cached(topic_name, schema_name)
        msg_type = self.topic_msg_message.get(topic_name, None)
        if msg_type is None:
            # If we still can't resolve it, skip gracefully (mirrors your try/except stance)
            return None
        return deserialize_message(raw_bytes, msg_type)

    # ---------- Public API (mirrors your sqlite version) ----------

    # Return [(timestamp0, message0), (timestamp1, message1), ...]
    def get_messages(self,
                    topics: Optional[Iterable[str]] = None,
                    start_time: Optional[int] = None,
                    end_time: Optional[int] = None,
                    log_time_order: bool = True,
                    reverse: bool = False) -> List[Tuple[int, object]]:
        out: List[Tuple[int, object]] = []
        # iter_messages returns (schema, channel, message); message.data is raw CDR bytes; message.log_time is ns int
        for schema, channel, message in self.reader.iter_messages(topics=topics,start_time=start_time,end_time=end_time,log_time_order=log_time_order, reverse=reverse):
            decoded = self._decode(topics, getattr(schema, "name", None), message.data)
            if decoded is not None:
                out.append((message.publish_time, decoded))
        return out

    # Return [(timestampN, messageN)] for 1-indexed 'number' (like your LIMIT 1 OFFSET N-1)
    def get_message(self,
            topics: Optional[Iterable[str]] = None,
            start_time: Optional[int] = None,
            end_time: Optional[int] = None,
            log_time_order: bool = True,
            reverse: bool = False) -> List[List[int]]:
        
        result = []
        if topics is not None:
            
            # If single topic, return a flat list
            result = [msg.publish_time for _, _, msg in self.reader.iter_messages(topics=topics, start_time=start_time, 
                                                                            end_time=end_time, log_time_order=log_time_order, reverse=reverse)]
        
        return result

    # Return [ts0, ts1, ...]
    def get_message_timestamps(self,
                                topics: Optional[Iterable[str]] = None,
                                start_time: Optional[int] = None,
                                end_time: Optional[int] = None,
                                log_time_order: bool = True,
                                reverse: bool = False) -> List[List[int]]:
        
        result = []
        if topics is not None:
            
            # If single topic, return a flat list
            result = [msg.publish_time for _, _, msg in self.reader.iter_messages(topics=topics, start_time=start_time, 
                                                                            end_time=end_time, log_time_order=log_time_order, reverse=reverse)]
        
        return result

    # Given an iterable of timestamps, return [(timestamp, message), ...] in the same order
    def get_messages_based_on_ts(self, topic_name: str, ts_list: Iterable[int]) -> List[Tuple[int, object]]:
        ts_needed = {int(ts) for ts in ts_list}
        results: dict[int, object] = {}
        # Walk through messages once, capture the first occurrence of each requested timestamp
        for schema, channel, message in self.reader.iter_messages(topics=[topic_name]):
            if message.log_time in ts_needed and message.log_time not in results:
                decoded = self._decode(topic_name, getattr(schema, "name", None), message.data)
                if decoded is not None:
                    results[message.log_time] = decoded
                # Early exit if we’ve found everything
                if len(results) == len(ts_needed):
                    break

        # Preserve caller’s order; if any ts is missing, this will KeyError in your sqlite version.
        # To mirror that behavior without raising, we filter missing ones out.
        out: List[Tuple[int, object]] = []
        for ts in ts_list:
            ts = int(ts)
            if ts in results:
                out.append((ts, results[ts]))
        return out
