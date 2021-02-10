import numpy as np


def gaussian_noise(self, mean, var, shape):
    return np.random.normal(mean, var, shape)