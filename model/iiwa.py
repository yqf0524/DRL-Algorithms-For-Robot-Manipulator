import klampt as k
from klampt import WorldModel
from klampt import vis
import numpy as np
import sympy


class iiwa(object):
    """
    This class contains all iiwa needed parameters and functions.

    All of dynamic parameters of iiwa refer to the paper "Dynamic Identification of
    the KUKA LBR iiwa Robot With Retrieval of Physical Parameters Using Global Optimization".
    Link: https://ieeexplore.ieee.org/document/9112185

    The functions include forward kinematics, inverse kinematics, forward dynamics,
    inverse dynamics, kinematic sensitivity, choose configuration and simulation
    of environment disturbances.
    """
    def __init__(self):
        self.position_limit = np.rad2deg(np.array([168, 118, 168, 118, 168, 118, 173]))
        self.velocity_limit = np.rad2deg(np.array([85, 85, 100, 75, 130, 135, 135]))
        self.effort_limit = np.array([320, 320, 176, 176, 110, 40, 40])
        self.link_length = np.array([0.36, 0.42, 0.4, 0.126], dtype=np.float32)
        self.DH_params = np.array([
            [0.0, 0.36, 0.0, -np.pi / 2],
            [0.0, 0.00, 0.0, np.pi / 2],
            [0.0, 0.42, 0.0, -np.pi / 2],
            [0.0, 0.00, 0.0, np.pi / 2],
            [0.0, 0.40, 0.0, -np.pi / 2],
            [0.0, 0.00, 0.0, np.pi / 2],
            [0.0, 0.126, 0.0, 0.0]], dtype=np.float32)
        self.link_mass = np.array([])
        self.world = WorldModel()
        self.world.loadElement('./iiwa14.urdf')
        self.robot = self.world.robot(0)

    def forward_kinematic(self, joint_position, link_length=None):
        """
        Calculate the forward kinematic to get current pose of iiwa

        :param joint_position: current joint position
        :param link_length: current link length with disturbance
        :return: current pose
        """
        pose = np.eye(4)  # same shape with homogeneous matrix
        theta = joint_position + self.DH_params[:, 1]
        length = link_length if link_length else self.DH_params[:, 2]
        offset = self.DH_params[:, 3]
        alpha = self.DH_params[:, 4]
        for i in range(len(theta)):
            pose = np.matmul(pose, self._homo_matrix(theta, length, offset, alpha))
        return pose

    def _homo_matrix(self, theta, length, offset, alpha):
        """
        Help to calculate forward kinematics.
        Units of input parameters: radian, mm, mm, radian

        :param theta: Rotation about z by an angle theta ("joint angle")
        :param length: Translation along x by a distance a ("link length")
        :param offset: Translation along z by a distance d ("link offset")
        :param alpha: Rotation about x by an angle alpha ("link twist")

        :return: homogeneous matrix
        """
        Tr = np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), length*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), length*np.sin(theta)],
            [0,             np.sin(alpha),                np.cos(alpha),               offset],
            [0,             0,                            0,                           1]], dtype=np.float32)
        return Tr

    def robot(self):
        pass