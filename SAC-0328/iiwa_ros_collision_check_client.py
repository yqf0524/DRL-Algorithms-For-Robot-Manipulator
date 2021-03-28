#!/usr/bin/env python3

import rospy
import time
from sensor_msgs.msg import JointState
from iiwa_msgs.msg import JointPosition


class DataStream(object):
    def __init__(self):
        rospy.Subscriber('/iiwa/joint_states', JointState, self.joint_position_callback)
        # rospy.Subscriber('/iiwa/joint_states', JointState, self.joint_velocity_callback)
        rospy.init_node('joint_states_listener', anonymous=True)
        self.joint_position = ()
        self.joint_velocity = ()

    def joint_position_callback(self, msg):
        # rospy.loginfo(msg.position[0])
        self.joint_position = msg.position
        joint_posi = msg.position
        time.sleep(1)
        print(data.joint_position)

    def joint_velocity_callback(self, msg):
        self.joint_velocity = msg.velocity
    
    def check_self_collide_callback(self):
        pass

    def check_is_reached_goal_callback(self):
        pass

    

# if __name__ == '__main__':
#     data = DataStream()
#     rospy.spin()