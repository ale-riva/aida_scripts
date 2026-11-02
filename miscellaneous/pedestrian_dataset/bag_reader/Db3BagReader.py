import sqlite3
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import matplotlib.pyplot as plt
import numpy as np

from typing import (
    Iterable,
    Optional,
)

#########################################################################################################################
## The code below is "ported" from 
# https://github.com/ros/common_msgs/tree/noetic-devel/sensor_msgs/src/sensor_msgs
import sys
from collections import namedtuple
import ctypes
import math
import struct
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
#########################################################################################################################
class BagFileParser():
    def __init__(self, bag_file):
        self.conn = sqlite3.connect(bag_file)
        self.cursor = self.conn.cursor()

        ## create a message type map
        topics_data = self.cursor.execute("SELECT id, name, type FROM topics").fetchall()
        self.topic_type = {name_of:type_of for id_of,name_of,type_of in topics_data}
        self.topic_id = {name_of:id_of for id_of,name_of,type_of in topics_data}
        self.topic_msg_message = {}
        for id_of,name_of,type_of in topics_data:
            try:
                self.topic_msg_message[name_of] = get_message(type_of)
            except Exception as e:
                print(f"An error occurred: {e}")
                print("Message: "+type_of)
        #self.topic_msg_message = {name_of:get_message(type_of) for id_of,name_of,type_of in topics_data}

    def __del__(self):
        self.conn.close()

    #TODO: check if new get_messages works w/o start_time and end_time
    # # Return [(timestamp0, message0), (timestamp1, message1), ...]
    # def get_messages(self, topic_name):
    #     topic_id = self.topic_id[topic_name]
    #     # Get from the db
    #     rows = self.cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = {}".format(topic_id)).fetchall()
    #     # Deserialise all and timestamp them
    #     return [ (timestamp,deserialize_message(data, self.topic_msg_message[topic_name])) for timestamp,data in rows]


    def get_message(self, topic_name, number):
        topic_id = self.topic_id[topic_name]
        # Get from the db
        rows = self.cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = {} LIMIT 1 OFFSET {};".format(topic_id,number-1)).fetchall()
        # Deserialise all and timestamp them
        return [ (timestamp,deserialize_message(data, self.topic_msg_message[topic_name])) for timestamp,data in rows]

  ## Adapted from "get_messages" to just return the list of timestamps (no pointclouds, no data in general)
    def get_message_timestamps(self, topic_name):
        topic_id = self.topic_id[topic_name]
        # Define the query
        query = "SELECT timestamp FROM messages WHERE topic_id = ?"
        # Execute the query. Mind that 
        #       - "execute(...).fetchall()" returns a list of tuples, independently from the number of elements
        #       - each tuple has the number of elements indicated in "SELECT" 
        rows = self.cursor.execute(query, (topic_id,)).fetchall()
        # Return a list with the timestamps
        return [tuple[0] for tuple in rows]


    ## Adapted from "get_messages" to return timestamps and data, based on the timestamps
    # TODO: check if new get_messages works w/o start_time and end_time
    def get_messages(self,
                    topics: Optional[Iterable[str]] = None,
                    start_time: Optional[int] = None,
                    end_time: Optional[int] = None):
        topic_id = self.topic_id[topics]

        # Define the query to get all messages within the timestamp range
        query = "SELECT timestamp, data FROM messages WHERE timestamp >= ? AND timestamp <= ? AND topic_id = ?"
        # Execute the query
        rows = self.cursor.execute(query, (int(start_time), int(end_time), topic_id)).fetchall()
        # Deserialise all and timestamp them
        return [ (timestamp,deserialize_message(data, self.topic_msg_message[topics])) for timestamp,data in rows]

# if __name__ == "__main__":

#         bag_file = '/home/riccardo/iac_code/bags/2023-12-15_16-04_lidar4/2023-12-15_16-05_lidar_0.db3'

#         parser = BagFileParser(bag_file)

#         msg = parser.get_messages("/luminar_front_points")[0][1]
#         # p_des_1 = [trajectory.points[i].positions[0] for i in range(len(trajectory.points))]
#         # t_des = [trajectory.points[i].time_from_start.sec + trajectory.points[i].time_from_start.nanosec*1e-9 for i in range(len(trajectory.points))]

#         for field in msg.fields:
#             print(f"Field Name: {field.name}")
#             print(f"Field Offset: {field.offset}")
#             print(f"Field Datatype: {field.datatype}")
#             print(f"Field Count: {field.count}")
#             print("-------------------")
        
#         pcl_np = np.array(list(read_points(msg)))
#         print(pcl_np[0])
        


