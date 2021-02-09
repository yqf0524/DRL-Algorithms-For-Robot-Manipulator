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
        self.link_mass = np.array([])

    def forward_kinematic(self, ):
        pass
