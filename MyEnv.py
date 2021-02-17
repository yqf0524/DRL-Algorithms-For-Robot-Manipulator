import numpy as np
from model.iiwa14 import KinematicModel
from model.iiwa14 import DynamicModel


class MainEnv:
    """
    The main iiwa environment class.

    Three subclasses are:

        PositionControl,
        VelocityControl,
        ForceControl

    The main API methods that users of this class are:

        reset
        step
        render
        close
        seed

    And set the following attributes:

        action_space: The Space object corresponding to valid actions
            For continuous action space action_space means elements of valid action
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    """
    reward_range = (-float('inf'), float('inf'))
    # Set these in ALL subclasses
    action_space = None
    action_space_high = None
    action_space_low = None
    observation_space = None
    observation_space_high = None
    observation_space_low = None

    def reset(self):
        """
        Resets the environment to an initial state and returns an initial
        observation.

        :return: observation (object): the initial observation.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def render(self):
        """
        Renders the environment supported by klampt.vis.

        :return: no return
        """
        raise NotImplementedError

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    def compute_reward(self, achieved_goal, desired_goal):
        """
        Based on current position and target position calculate the
        distance. Negative distance become reward of applied action.

        :param achieved_goal: TCP position of next observation
        :param desired_goal: Target position of TCP
        :return: Negative reward value
        """
        raise NotImplementedError

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
        return

    def class_name(self):
        return str(self.__class__)


class PositionControl(MainEnv):
    def __init__(self):
        self.iiwa = KinematicModel()
        self.action_space = len(self.iiwa.current_configuration)
        self.action_space_high = np.deg2rad(self.iiwa.velocity_limit / 100)  # 100 hz
        self.action_space_low = np.deg2rad(-self.iiwa.velocity_limit / 100)  # 100 hz
        self.observation_space = len(self.iiwa.current_configuration)
        self.observation_space_high = np.deg2rad(self.iiwa.joint_limit)
        self.observation_space_low = np.deg2rad(-self.iiwa.joint_limit)
        self.current_action = np.zeros(7)
        self.target_position = np.zeros(3)
        self.start_position = self.iiwa.current_ee_position
        self.target_rpy = self.iiwa.current_ee_rpy
        self.tol_position = 1e-4  # 0.0001 meter = 0.1 mm, Euclidean Distance
        self.tol_orientation = 2e-3  # approximate 0.115, Euclidean Distance
        self.is_done_counter = 0

    def reset(self):
        self.iiwa.current_configuration = self.iiwa.start_configuration
        current_state = self.iiwa.current_configuration
        self.iiwa.update_kinematic()
        self.start_position = self.iiwa.current_ee_position
        return current_state

    def step(self, action):
        action = self.iiwa.clip_velocity(action)
        next_state = self.iiwa.current_configuration + action
        # collision = self.iiwa.check_collision(next_state)
        in_joint_limit, next_state = self.iiwa.clip_joint_position(next_state)
        info = "Everything is fine."
        # if collision:
        #     reward = -100.0
        #     done = True
        #     info = "Oops, a collision has occurred."
        #     return next_state, reward, done, info
        self.iiwa.current_configuration = next_state
        self.iiwa.update_kinematic()
        ee_rpy = self.iiwa.current_ee_rpy
        ee_position = self.iiwa.current_ee_position
        reward, done = self.compute_reward(ee_rpy, ee_position)
        if not in_joint_limit:
            self.is_done_counter += 1
            print(self.is_done_counter)
            reward += -1

        return next_state, reward, done, info

    def render(self):
        config = self.iiwa.current_configuration
        self.iiwa.display_robot(config)

    def compute_reward(self, ee_rpy, ee_position):
        rpy_error = np.linalg.norm(ee_rpy - self.target_rpy)
        position_error = np.linalg.norm(ee_position - self.target_position)
        reward = -(rpy_error + position_error)
        is_done = position_error <= self.tol_position  # rpy_error <= self.tol_orientation
        if is_done:
            reward = -rpy_error / self.tol_orientation + \
                     position_error / self.tol_position
        return reward, is_done


class VelocityControl(MainEnv):
    def __init__(self):
        pass

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self):
        pass

    def compute_reward(self, achieved_goal, desired_goal):
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return -distance


class ForceControl(MainEnv):
    def __init__(self):
        pass

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self):
        pass

    def compute_reward(self, achieved_goal, desired_goal):
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return -distance


class KUKAiiwa:
    subclasses = {
        "PositionControl": PositionControl,
        "VelocityControl": VelocityControl,
        "ForceControl": ForceControl}

    def __new__(cls, control_type: str):
        try:
            return cls.subclasses[control_type]()
        except TypeError as T:
            # print(T.__doc__)
            print("No class name: ", control_type)
            print("subclasses: PositionControl, VelocityControl, ForceControl.")
