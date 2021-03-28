#!/usr/bin/env python3

from training_algorithms.msg import RobotConfiguration
from iiwa_msgs.msg import JointPosition
from training_algorithms.msg import CartesianPose
import rospy

class PublishJointPosition(object):
    def __init__(self):
        self.joint_position_pub = rospy.Publisher("/iiwa/command/JointPosition", \
                                  JointPosition, queue_size=10)
        self.robot_configuration = JointPosition()
    
    def publish(self, configuration):
        # self.robot_configuration.header.seq = ''
        # self.robot_configuration.header.frame_id = ''
        self.robot_configuration.header.stamp = rospy.Time.now()
        self.robot_configuration.position.a1 = configuration[0]
        self.robot_configuration.position.a2 = configuration[1]
        self.robot_configuration.position.a3 = configuration[2]
        self.robot_configuration.position.a4 = configuration[3]
        self.robot_configuration.position.a5 = configuration[4]
        self.robot_configuration.position.a6 = configuration[5]
        self.robot_configuration.position.a7 = configuration[6]

        self.joint_position_pub.publish(self.robot_configuration)


class PublishCartesianPose(object):
    def __init__(self):
        self.cartesian_pose_pub = rospy.Publisher("/iiwa/CartesianPose", \
                                  CartesianPose, queue_size=10)
        self.cartesian_pose = CartesianPose()
        # rospy.init_node('cartesian_pose_pub', anonymous=True)

    def publish(self, current_ee_pose):
        # self.cartesian_pose.header.seq = ''
        # self.cartesian_pose.header.frame_id = ''
        self.cartesian_pose.header.stamp = rospy.Time.now()
        self.cartesian_pose.x = current_ee_pose[0]
        self.cartesian_pose.y = current_ee_pose[1]
        self.cartesian_pose.z = current_ee_pose[2]
        self.cartesian_pose.a = current_ee_pose[3]
        self.cartesian_pose.b = current_ee_pose[4]
        self.cartesian_pose.c = current_ee_pose[5]

        self.cartesian_pose_pub.publish(self.cartesian_pose)
    