import klampt as k
from klampt import vis
from klampt import math
import time
import numpy as np
import traceback
from TD3PG.OU_process import OUNoise

arr = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
re = np.reshape(arr, (1, -1), order="F").squeeze()
print(arr)
print(type(re))
arr = list(re)
arr = [0].extend(arr)
print([0,0,arr,0])
print(type(arr))
print(type(np.random.normal(0,2,8)))

# arr = math.so3.matrix(arr)
# print(arr)
# arr = math.so3.identity()
# print(arr)
# rpy = math.so3.rpy(arr)
# print(type(rpy))

# print(np.array([6]))
# world = k.WorldModel()
# world.loadElement('./iiwa14.urdf')
# robot = world.robot(0)
# vis.add('robot', robot)
# vis.createWindow("display robot")
#
# print(robot.numDrivers())
#
# link_ee = robot.link(8)
# print(link_ee.getLocalDirection([0,0,0]))
# print(link_ee.getWorldPosition([0,0,0]))
# robot.setConfig([0,1,1,1,1,1,1,1,0,0])
# print(link_ee.getLocalDirection([0,0,0]))



# t0 = time.time()
# q0 = list(np.zeros(10))
# q = list(np.ones(10))
# link_ee = world.link(8)
# print(link_ee.getLocalPosition([0,0,0]))
#
# world.setConfig(q)
# vis.show()
if __name__ == '__main__':
    ou = OUNoise(7)
    states = []
    for i in range(1000):
        states.append(ou.noise())
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()