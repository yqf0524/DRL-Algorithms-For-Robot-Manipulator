import numpy as np


class OrnsteinUhlenbeckNoise:
    """
    A Ornstein Uhlenbeck action noise, this is designed to aproximate brownian motion with friction.

    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

    :param dim: (tuple) the dimension of the noise
    :param mu: (float) the mean of the noise
    :param theta: (float) the rate of mean reversion, affect converge
    :param sigma: (float) the scale of the noise, affect random
    :param dt: (float) the timestep for the noise
    """

    def __init__(self, dim, mu=0, theta=0.15, sigma=0.2, dt=1.0):
        self.dim = dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt

        self.X = np.ones(self.dim) * self.mu

    def reset(self):
        self.X = np.ones(self.dim) * self.mu

    def __call__(self):
        drift = self.theta * (self.mu - self.X) * self.dt
        random = self.sigma * self._delta_wiener()

        self.X = self.X + drift + random

        return self.X

    def _delta_wiener(self):
        return np.sqrt(self.dt) * np.random.randn(self.dim)