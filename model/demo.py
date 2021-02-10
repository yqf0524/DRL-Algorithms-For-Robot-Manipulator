import klampt as k
from klampt import vis
import time
import numpy as np

world = k.WorldModel()
world.loadElement('./iiwa14.urdf')
robot = world.robot(0)
vis.add('robot', robot)
vis.createWindow("display robot")

print(robot.numDrivers())


# t0 = time.time()
# q0 = list(np.zeros(10))
# q = list(np.ones(10))
# link_ee = world.link(8)
# print(link_ee.getLocalPosition([0,0,0]))
#
# world.setConfig(q)
# vis.show()

