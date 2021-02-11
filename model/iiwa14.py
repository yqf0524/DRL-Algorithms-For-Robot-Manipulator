from klampt.model import collide
from klampt import math
from klampt import WorldModel
from klampt import vis
import numpy as np
import sympy


class Model:
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
        vis.add("iiwa", self.robot)
        self.init_configuration = np.zeros(7)

    def so3_to_rpy(self, so3_orientation):
        """
              [a11,a12,a13]
        so3 = [a21,a22,a23]
              [a31,a32,a33]
              
        Klamp’t represents the matrix as a list
        [a11, a21, a31, a12, a22, a32, a13, a23, a33].

        :param so3_orientation: orientation of end effector
        :return: rpy corresponse to (X, Y, Z)
        """
        so3 = np.reshape(so3_orientation, (1, -1), order="F").squeeze(axis=0)
        so3 = list(so3)
        rpy = math.so3.rpy(so3)
        return rpy

    def check_collision(self, configuration):
        """
        Using klampt.model.collide.WorldCollider(world, ignore=[]) to check collision of robot model

        :param configuration: the configuration that must be checked
        :return (bool): False -> no collision, True -> collision occurred
        """
        collision = False
        collide_test = collide.WorldCollider(self.world)
        for (i, j) in collide_test.robotSelfCollisions(0):
            pass
        return collision

    def display_robot(self, configuration):
        """
        Display robot model through klampt.vis

        :param configuration: current configuration of robot
        :return: nothing
        """
        config = list(configuration)
        config = [0] + config + [0, 0]
        self.robot.setConfig(config)
        vis.show()


class KinematicModel(Model):
    """
    Subclass of Model.
    """

    def __init__(self):
        super(KinematicModel, self).__init__()
        self.current_configuration = self.init_configuration
        self.current_ee_se3 = None
        self.current_ee_so3 = None
        self.current_ee_position = None
        self.current_ee_rpy = None
        self.update_kinematic()

    def forward_kinematic(self, configuration, link_length_noise=None):
        """
        Calculate the forward kinematic to get current pose of iiwa

        :param configuration: current joint position
        :param link_length_noise: current link length disturbance (7, 1)
        :return: current pose
        """
        ee_se3 = np.eye(4)
        theta = configuration + self.DH_params[:, 1]
        length = np.transpose(self.DH_params[:, 2]) + link_length_noise
        offset = np.transpose(self.DH_params[:, 3])
        alpha = np.transpose(self.DH_params[:, 4])
        for i in range(len(theta)):
            ee_se3 = np.matmul(ee_se3, self._homo_matrix(theta, length, offset, alpha))
        return ee_se3

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
            [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), length * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), length * np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), offset],
            [0, 0, 0, 1]], dtype=np.float32)
        return Tr

    def update_kinematic(self):
        self.current_ee_se3 = self.forward_kinematic(
            self.current_configuration, self.link_gaussian_noise(0, 2, 8))
        self.current_ee_so3 = self.current_ee_se3[0:3, 0:3]
        self.current_ee_position = self.current_ee_se3[3, 0:3]
        self.current_ee_rpy = self.so3_to_rpy(self.current_ee_so3)

    def link_gaussian_noise(self, mean, var, shape):
        return np.random.normal(mean, var, shape)


class DynamicModel(Model):
    def __init__(self):
        super(DynamicModel, self).__init__()
        self.effort_limit = np.array([320, 320, 176, 176, 110, 40, 40])
