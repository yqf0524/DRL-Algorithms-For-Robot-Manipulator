from iiwa_model import KinematicModel
import numpy as np


class PositionControl(object):
    """
    The custom environment for deep reinforcement learning of KUKA iiwa LBR 14.
    """
    def __init__(self):
        self.iiwa = KinematicModel()
        self.reward_range = (-100, 100)
        self.action_space = len(self.iiwa.init_configuration)
        self.action_space_high = self.iiwa.velocity_limit  # 100 hz
        self.action_space_low = -self.iiwa.velocity_limit  # 100 hz
        self.observation_space = 7 + 6 + 6  # (config, cur_pose, tar_pose)
        self.observation_space_high = self.iiwa.joint_limit
        self.observation_space_low = -self.iiwa.joint_limit
        self.xyz_error_norm_recorder = self.iiwa.xyz_error_norm
        self.min_xyz_error = self.iiwa.xyz_error_norm
        self.coord_xyz = self.iiwa.current_ee_xyz

    def reset(self):
        """
        Resets the environment to initial state and returns an initial observation.
        :return: observation (object): the initial observation.
        """
        self.iiwa.current_configuration = self.iiwa.init_configuration
        self.iiwa.update_kinematic()
        self.xyz_error_norm_recorder = self.iiwa.xyz_error_norm
        self.min_xyz_error = self.iiwa.xyz_error_norm
        self.coord_xyz = self.iiwa.current_ee_xyz
        observation = np.concatenate((np.divide(self.iiwa.current_configuration, 
                      self.observation_space_high), self.iiwa.current_ee_pose, 
                      self.iiwa.target_ee_pose), axis=0)
        return observation

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        :param action: an action provided by the agent
        :return:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step()
                         calls will return undefined results
            info (str): contains auxiliary diagnostic information (helpful for debugging,
                         and sometimes learning)
        """
        next_state = self.iiwa.current_configuration + action
        in_joint_limit, next_state = self.iiwa.clip_joint_position(next_state)
        self.iiwa.current_configuration = next_state
        self.iiwa.update_kinematic()
        next_observation = np.concatenate((np.divide(self.iiwa.current_configuration, 
                           self.observation_space_high), self.iiwa.current_ee_pose, 
                           self.iiwa.target_ee_pose), axis=0)
        if self.iiwa.xyz_error_norm < self.min_xyz_error:
            self.min_xyz_error = self.iiwa.xyz_error_norm
            self.coord_xyz = self.iiwa.current_ee_xyz

        reward = self.compute_reward(self.iiwa.xyz_error_norm, \
                                     self.iiwa.rpy_error_norm)
        done = self.iiwa.is_achieved
        info = "Aha!: Everything's fine!"
        # The following codes will check whether the next state is something wrong or not.
        # If next_state is not in joint_limit, give him a punishment (-100).
        if not in_joint_limit:
            reward += self.reward_range[0]
            done = True
            info = "Warning: next_state was out of the joint limit. "
        # If next_state is not in in_workspace, give him a higher punishment (-100) and 
        # finish the current learnning process.
        if not self.iiwa.in_workspace:
            reward += self.reward_range[0]  # -100
            done = True
            info = "Error: the position of TCP was out of the workspace."
        # If next_state causes a collision with itself or other objects, give him a higher 
        # punishment (-100) and finish the current learnning process.
        if self.iiwa.is_collide:
            reward += self.reward_range[0] * 1.5
            done = True
            info = "Fatal error: Oops! A collision has occurred."

        return reward, next_observation, done, info

    def compute_reward(self, xyz_error_norm, rpy_error_norm):
        """
        Based on current position_error and current orientation error calculate the absolute
        error. The sum of both negative errors represent the reward of applied action.
        :param xyz_error_norm: current position_error of Endeffector
        :param rpy_error_norm: current orientation_error of Endeffector
        :param is_achieved: Whether iiwa has achieved its goal position
        :return: total reward value
        """
        # The inverse of log(xyz_error_norm) sum negative rpy_error_norm 
        # represent the reward (punishment) of the current action.
        # reward = -xyz_error_norm * np.exp(xyz_error_norm) - \
        #          self.iiwa.tol_rpy_norm * np.power(rpy_error_norm, np.e)
        reward = 0
        done_rpy = rpy_error_norm <= self.iiwa.tol_rpy_norm
        done_xyz = xyz_error_norm <= self.xyz_error_norm_recorder
        if done_rpy:
            reward += self.iiwa.tol_rpy_norm / rpy_error_norm
        else:
            reward -= rpy_error_norm / self.iiwa.tol_rpy_norm * 0.2
        # If the robot move towards to target, give him a positive reward.
        # Then remember this distance for next iteration.
        if done_xyz:
            # reward += self.xyz_error_norm_recorder / xyz_error_norm
            self.xyz_error_norm_recorder = xyz_error_norm
            reward +=  np.power(np.log(xyz_error_norm), 2) * 2 
            # self.xyz_error_norm_recorder /= 1.05  # after 156 times, < tol_xyz_norm
        else:
            reward -= xyz_error_norm / self.xyz_error_norm_recorder
        # If the robot reaches its goal position, give him an extra high positive reward.
        if self.iiwa.is_achieved:
            reward += self.reward_range[1] * 5  # +100
        return reward

    def close(self):
        """
        Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    def render(self):
        """
        Renders the environment supported by klampt.vis.
        :return: no return
        """
        pass

    def seed(self, seed=None):
        """
        Sets the seed for this env's random number generator(s).
        :param seed:
        :return:
            list<bigint>: Returns the list of seeds used in this env's random
            number generators. The first value in the list should be the
            "main" seed, or the value which a reproducer should pass to
            'seed'. Often, the main seed equals the provided 'seed', but
            this won't be true if seed=None, for example.
        """
        pass

    def class_name(self):
        return str(self.__class__)

    def z_score_norm(self, obs, axis=0):
        obs_mean = np.mean(obs, axis=axis)
        obs_std = np.min(obs, axis=axis)
        obs_z_score_norm = np.divide(obs - obs_mean, obs_std)
        return obs_z_score_norm






class VelocityControl(object):
    def __init__(self):
        pass

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self):
        pass

    def compute_reward(self, ee_position, ee_rpy):
        pass


class ForceControl(object):
    def __init__(self):
        pass

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self):
        pass

    def compute_reward(self, ee_position, ee_rpy):
        pass
