#!/usr/bin/env python3

from training_algorithms.msg import RobotConfiguration
from training_algorithms.msg import CartesianPose
from sensor_msgs.msg import JointState
import rospy
import time


class ReturnJointState(object):
    def __init__(self):
        rospy.Subscriber('/iiwa/joint_states', JointState, self.joint_position_callback)
        # rospy.Subscriber('/iiwa/joint_states', JointState, self.joint_velocity_callback)
        self.joint_position = ()
        self.joint_velocity = ()

    def joint_position_callback(self, msg):
        # rospy.loginfo(msg.position)
        self.joint_position = msg.position

    def joint_velocity_callback(self, msg):
        self.joint_velocity = msg.velocity
    
    def check_self_collide_callback(self):
        pass

    def check_is_reached_goal_callback(self):
        pass

    
class ReturnCartesianPose:
    def __init__(self):
        rospy.Subscriber('/iiwa/CartesianPose', CartesianPose, self.cartesian_pose_callback)
        self.cartesian_pose = [0.0 for _ in range(6)]

    def cartesian_pose_callback(self, msg):
        # print("In cartesian pose sub ...")
        self.cartesian_pose[0] = msg.x
        self.cartesian_pose[1] = msg.y
        self.cartesian_pose[2] = msg.z
        self.cartesian_pose[3] = msg.a
        self.cartesian_pose[4] = msg.b
        self.cartesian_pose[5] = msg.c
 