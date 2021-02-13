from klampt.model import collide
from klampt import WorldModel
from klampt import vis
from scipy.spatial.transform import Rotation as R
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
        self.joint_limit = np.rad2deg(np.array([168, 118, 168, 118, 168, 118, 173]))
        self.velocity_scale = 0.1
        self.velocity_limit = self.velocity_scale * \
                              np.rad2deg(np.array([85, 85, 100, 75, 130, 135, 135]))
        self.DH_params = np.array([
            [0.0, 0.36, 0.0, -np.pi / 2],
            [0.0, 0.00, 0.0, np.pi / 2],
            [0.0, 0.42, 0.0, np.pi / 2],
            [0.0, 0.00, 0.0, -np.pi / 2],
            [0.0, 0.40, 0.0, -np.pi / 2],
            [0.0, 0.00, 0.0, np.pi / 2],
            [0.0, 0.126, 0.0, 0.0]], dtype=np.float32)
        self.link_mass = np.array([])
        self.world = WorldModel()
        self.world.loadElement('model/iiwa14.urdf')
        self.robot = self.world.robot(0)
        vis.add("iiwa", self.robot)

    def so3_to_rpy(self, so3):
        """
        Converts an SO3 rotation matrix to rpy angles

        :param so3: rotation matrix
        :return: list of rpy angles
        """
        r = R.from_matrix(so3)
        rpy = r.as_euler("xyz")
        return rpy

    def check_joint_limit(self, configuration):
        in_joint_limit = (np.abs(configuration) <= self.joint_limit).all()
        if not in_joint_limit:
            return False, np.clip(configuration, -self.joint_limit, self.joint_limit)
        return True, configuration

    def clip_velocity(self, action):
        """
        Set collection frequency of data to 100 HZ. Approximate
        current action as current velocity

        :param action: action will be taken based on current observation
        :return: the clipped action
        """
        clip_action = np.clip(action, -self.velocity_limit, self.velocity_limit)
        return clip_action

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
        self.current_configuration = np.deg2rad(
            [0, 60, 0, -61.9587, 0, 60, 0])
        # [-120, -60, 78, 62, -83, -100, 148])
        self.current_ee_position = None
        self.current_ee_rpy = None
        self.update_kinematic()

    def forward_kinematic(self, configuration, link_length_noise=None):
        """
        Calculate the forward kinematic to get current pose of iiwa

        :param configuration: current joint position
        :param link_length_noise: current link length disturbance (1, 7)
        :return: current se3
        """
        ee_se3 = np.eye(4)
        theta = configuration  # + np.transpose(self.DH_params[:, 0])
        length = self.DH_params[:, 1] + link_length_noise
        offset = [0, 0, 0, 0, 0, 0, 0]  # np.transpose(self.DH_params[:, 2])
        alpha = self.DH_params[:, 3]
        for i in range(len(configuration)):
            ee_se3 = np.matmul(
                ee_se3, self._homo_matrix(theta[i], length[i], offset[i], alpha[i]))
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
        tr = np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha),
             np.sin(theta) * np.sin(alpha), length * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(alpha),
             -np.cos(theta) * np.sin(alpha), length * np.sin(theta)],
            [0.0, np.sin(alpha), np.cos(alpha), offset],
            [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        return tr

    def update_kinematic(self):
        current_ee_se3 = self.forward_kinematic(
            self.current_configuration, self.link_noise(0, 0.0002, 7))
        current_ee_so3 = current_ee_se3[0:3, 0:3]
        self.current_ee_position = current_ee_se3[0:3, 3]
        self.current_ee_rpy = self.so3_to_rpy(current_ee_so3)

    def link_noise(self, loc, scale, size):
        """
        Create a random noise to simulate environment influence and montage error

        :param scale: float or array_like of floats
                Mean ("centre") of the distribution.
        :param loc: float or array_like of floats
                Standard deviation (spread or "width") of the distribution. Must be
                non-negative.
        :param size: int or tuple of ints, optional，corresponds to noise target
        :return: a scaled noise array
        """
        # noise = [0, 0.02 mm] = 0.0002 m
        noise = np.random.normal(loc, scale, size)
        return noise


class DynamicModel(Model):
    def __init__(self):
        super(DynamicModel, self).__init__()
        self.effort_limit = np.array([320, 320, 176, 176, 110, 40, 40])
