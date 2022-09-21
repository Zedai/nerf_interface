#!/usr/bin/env python
import rospy
from pickle import dumps, loads
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import socket
import matplotlib.pyplot as plt
import numpy as np
import time

global s

def listener():
    rospy.init_node('interface_nerf', anonymous=True)
    rospy.Subscriber("/quadrotor/pose", PoseStamped, callback)

    rospy.spin()

def callback(data):
    pub = rospy.Publisher('renderedView', Image, queue_size=10)

    x = data.pose.orientation.x
    y = data.pose.orientation.y
    z = data.pose.orientation.z
    w = data.pose.orientation.w
    r00 = 2*(x**2 + y**2)-1
    r01 = 2*(y*z - x*w)
    r02 = 2*(y*w + x*z)
    r10 = 2*(y*z + x*w)
    r11 = 2*(x**2 + z**2)-1
    r12 = 2*(z*w - x*y)
    r20 = 2*(y*w - x*z)
    r21 = 2*(z*w + x*y)
    r22 = 2*(x**2 + w**2)-1
    mat = [[r00, r01, r02, data.pose.position.x],
           [r10, r11, r12, data.pose.position.y],
           [r20, r21, r22, data.pose.position.z],
           [0, 0, 0, 1]]

    while not rospy.is_shutdown():
        s.send(dumps(mat))

        fragments = []
        while True:
            packet = s.recv(1000)
            if packet[-4:] == b'done':
                fragments.append(packet[:-4])
                break
            if not packet: break
            fragments.append(packet)

        final = loads(b"".join(fragments))
        pub.publish(bridge.cv2_to_imgmsg(final, encoding="rgb8"))
        break

if __name__ == '__main__':
    try:
        s = socket.socket()
        while True:
            try:
                s.connect(('128.238.39.126', 12348))
#                s.connect(('127.0.0.1', 12348))
                break
            except ConnectionRefusedError:
                print('NerF Simulator not Running')
                
        listener()
    except rospy.ROSInterruptException:
        s.close()
        pass
