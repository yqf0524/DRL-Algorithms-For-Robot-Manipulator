from scipy.spatial.transform import Rotation as R
import numpy as np
import sympy


class KinematicModel(object):
    """
    This class contains all iiwa needed parameters and functions.
    All of kinematic parameters of iiwa refer to KUKA technical data.
    The important functions include forward kinematics, kinematic sensitivity, 
    
    of environment disturbances.
    """
    def __init__(self):
        self.joint_limit = np.array([168, 118, 168, 118, 168, 118, 173])
        self.velocity_scale = 0.1 * 0.02  # 0.01 sec
        self.velocity_limit = self.velocity_scale * np.array([85, 85, 100, 75, 130, 135, 135])
        self.DH_params = np.array([
            [0.0, 0.36, 0.0, -np.pi/2],
            [0.0, 0.00, 0.0, np.pi/2],
            [0.0, 0.42, 0.0, -np.pi/2],
            [0.0, 0.00, 0.0, np.pi/2],
            [0.0, 0.40, 0.0, -np.pi/2],
            [0.0, 0.00, 0.0, np.pi/2],
            [0.0, 0.126, 0.0, 0.0]], dtype=np.float32)

        self.init_configuration = np.array([30, 60, 0, 60, 0, 60, 0])

        self.start_ee_se3 = self.forward_kinematic(self.init_configuration)
        self.start_ee_position = self.start_ee_se3[0:3, 3]
        self.start_ee_rpy = self.so3_to_rpy(self.start_ee_se3[0:3, 0:3])
        self.start_ee_pose = np.concatenate((self.start_ee_position, 
                                             self.start_ee_rpy), axis=0)

        self.current_configuration = self.init_configuration
        self.current_ee_position = np.zeros(3)
        self.current_ee_rpy = np.zeros(3)
        self.current_ee_pose = np.zeros(6)

        self.target_ee_position = np.zeros(3)
        self.target_ee_rpy = np.zeros(3)
        self.target_ee_pose = np.zeros(6)

        self.tol_position = 1e-4  # 0.0001 meter = 0.1 mm, Euclidean Distance
        self.tol_orientation = 2e-3  # approximate 0.115 grad, winkel

        self.position_error_norm = 0
        self.rpy_error_norm = 0
        self.is_achieved = False
        self.is_collide = False
        self.in_workspace = True
        self.update_kinematic()

    def forward_kinematic(self, configuration, link_length_noise=None):
        """
        Calculate the forward kinematic to get current pose of iiwa
        :param configuration: current joint position
        :param link_length_noise: current link length disturbance (1, 7)
        :return: current se3
        """
        ee_se3 = np.eye(4)
        theta = np.deg2rad(configuration)  # + np.transpose(self.DH_params[:, 0])
        length = self.DH_params[:, 1]
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
        :param length: Translation along z by a distance d ("link length")
        :param offset: Translation along x by a distance a ("link offset")
        :param alpha: Rotation about x by an angle alpha ("link twist")
        :return: homogeneous matrix
        """
        tr = np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha),
             np.sin(theta) * np.sin(alpha), offset * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(alpha),
             -np.cos(theta) * np.sin(alpha), offset * np.sin(theta)],
            [0.0, np.sin(alpha), np.cos(alpha), length],
            [0.0, 0.0, 0.0, 1.0]])
        return tr

    def set_init_configuration(self, init_configuration):
        """
        This function is called to set the init configuration of robot manipulator.
        :param configuration: The user given init configuration.
        """
        self.init_configuration = init_configuration
        self.update_kinematic()

    def set_target_ee_pose(self, position, orientation):
        """
        This function is called to set the target pose of robot manipulator.
        :param position: target position of endeffector.
        :param orientation: target orientation of endeffector, in rpy
        :return no return value
        """
        self.target_ee_position = position
        self.target_ee_rpy = orientation
        self.target_ee_pose = np.concatenate((self.target_ee_position, 
                              self.target_ee_rpy), axis=0)

    def so3_to_rpy(self, so3):
        """
        Converts an SO3 rotation matrix to rpy angles
        :param so3: rotation matrix
        :return: list of rpy angles
        """
        r = R.from_matrix(so3)
        rpy = r.as_euler("xyz")
        return rpy

    def clip_joint_position(self, configuration):
        """
        Clip joint positions and check whether they are out of joint limit or not.
        :param configuration: current robot configuration or joint positions
        :return in_joint_limit: bool value, 
                new_config: the robot configuration after np.clip.
        """
        new_config = np.clip(configuration, -self.joint_limit, self.joint_limit)
        in_joint_limit = (new_config == configuration).all()
        return in_joint_limit, new_config

    def clip_joint_velocity(self, action):
        """
        Set collection frequency of data to 100 HZ. Approximate
        current action as current velocity
        :param action: action will be taken based on current observation
        :return: the clipped action
        """
        action = action
        clip_action = np.clip(action, -self.velocity_limit, self.velocity_limit)
        return clip_action

    def collision_check(self):
        """
        Using moveit! Robot_states to check collision. This method will be 
        wrapped into a service server. For a given robot configuration it 
        returns a bool value.
        :return bool value. False -> no collision, True -> collision occurred
        """
        # configuration: the configuration that will be checked
        self.is_collide = False
        #/TODO
        return self.is_collide

    def is_in_workspace(self):
        """
        Check whether cartesian position of pose in custom defined workspace.
        :param ee_position, the position of endeffector.
        :return in_workspace, a bool value
        """
        ee_position = self.current_ee_position
        self.in_workspace = ((0.3 <= ee_position[0] and ee_position[0] <= 1.0) and \
                            (-0.5 <= ee_position[1] and ee_position[1] <= 0.5) and \
                            (0.01 <= ee_position[2] and ee_position[2] <= 0.7))
    
    def is_achieved_goal(self):
        # Computer the norm of position_error.
        position_error_norm = np.linalg.norm(self.current_ee_position - 
                                             self.target_ee_position)
        # Finding the minimum rpy errors. e.g.: -pi = pi in cartesian space.
        rpy_error = np.min((np.abs(self.current_ee_rpy - self.target_ee_rpy), \
                    np.abs(self.current_ee_rpy + self.target_ee_rpy)), axis=0)
        # Computer the norm of rpy_error.
        rpy_error_norm = np.linalg.norm(rpy_error)
        # Checking whether the error meets the accuracy requirements
        self.is_achieved = position_error_norm <= self.tol_position \
                           and rpy_error_norm <= self.tol_orientation

        self.position_error_norm = position_error_norm
        self.rpy_error_norm = rpy_error_norm

    def update_kinematic(self):
        """
        This function is called to update the robot kinematic parameters using 
        forward_kinematic function. Three parameters will be updated: 
        # current_ee_position: degree in x-, y-, z-axises.
        # current_ee_orientation: radian in roll-pitch-yaw.
        # current_ee_pose: combine both: [position, orientation].
        """
        current_ee_se3 = self.forward_kinematic(self.current_configuration)
        current_ee_so3 = current_ee_se3[0:3, 0:3]
        self.current_ee_position = current_ee_se3[0:3, 3]
        self.current_ee_rpy = self.so3_to_rpy(current_ee_so3)
        self.current_ee_pose = np.concatenate((self.current_ee_position,
                               self.current_ee_rpy), axis=0)
        self.is_achieved_goal()
        self.is_in_workspace()
        self.collision_check()

    def display_robot(self, configuration):
        """
        Display robot model through klampt.vis
        :param configuration: current configuration of robot
        :return: nothing
        """
        pass






















class DynamicModel(object):
    def __init__(self):
        super(DynamicModel, self).__init__()
        self.effort_limit = np.array([320, 320, 176, 176, 110, 40, 40])
