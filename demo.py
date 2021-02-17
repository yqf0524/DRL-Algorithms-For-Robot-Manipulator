
import numpy as np
import time
import matplotlib.pyplot as plt



# arr = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
# re = np.reshape(arr, (1, -1), order="F").squeeze()
# print(arr)
# print(type(re))
# arr = list(re)
# arr = [0].extend(arr)
# print([0, 0, arr, 0])
# print(type(arr))
# print(type(np.random.normal(0, 2, 8)))

"""
class fruit:
    print("fruit class")


class apple(fruit):
    print("apple class")

    def _fruit(self):
        print("call apple")


class orange(fruit):
    print("orange class")

    def _fruit(self):
        print("call orange")


class choose:
    subclasses = {"apple": apple,
                  "orange": orange}

    def __new__(cls, subclass: str):
        if subclass in cls.subclasses.keys():
            return cls.subclasses[subclass]()
        else:
            print("No class name: ", subclass)
"""

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

"""
class Fruit(object):
    def __init__(self):
        print("Fruit class")
        pass

    def print_color(self):
        pass


class Apple(Fruit):
    def __init__(self):
        super(Apple, self).__init__()
        print("Apple class")
        pass

    def print_color(self):
        print("apple is in red")


class Orange(Fruit):
    def __init__(self):
        print("Orange class")
        pass

    def print_color(self):
        print("orange is in orange")


class FruitFactory(object):
    fruits = {"apple": Apple, "orange": Orange}

    def __new__(cls, name):
        if name in cls.fruits.keys():
            return cls.fruits[name]()
        else:
            return Fruit()
"""


if __name__ == '__main__':

    plt.ion()
    plt.figure(0)
    dt = 0.01
    counter = 0

    while True:
        t_now = dt * counter

        y0 = np.sin(t_now)
        y1 = np.cos(t_now)

        plt.plot(t_now, y0, '.')
        plt.draw()
        time.sleep(0.01)

    # Parameter = namedtuple("Parameter", ["theta", "mu", "dt", "sigma"])
    # pars = [
    #     # # theta
    #     # Parameter(1, 0, 1, 1),
    #     # Parameter(1e-2, 0, 1, 1),
    #     # # mu
    #     # Parameter(1, 0, 1, 1),
    #     # Parameter(1, 10, 1, 1),
    #     # # dt
    #     # Parameter(1, 0, 1, 1),
    #     # Parameter(1, 0, 1e-2, 1),
    #     # # sigma
    #     # Parameter(1, 0, 1, 1),
    #     # Parameter(1, 0, 1, 1e-2),
    #     # DDPG
    #     Parameter(0.15, 0, 1, 0.2),
    #     Parameter(0.15, 0, 1e-2, 0.5),
    # ]
    # t = np.linspace(0, 100, 100)
    # ys = []
    # for par in pars:
    #     noise = OrnsteinUhlenbeckNoise(
    #         1, mu=par.mu, dt=par.dt, theta=par.theta, sigma=par.sigma
    #     )
    #     ys.append([noise() for _ in t])
    #
    # import matplotlib.pyplot as plt
    #
    # fig, ax = plt.subplots(nrows=len(ys), sharex=True)
    # fig.suptitle(
    #     "$X_{n+1} = X_n + \\theta (\mu - X_n)\Delta t + \sigma \Delta W_n$", fontsize=16
    # )
    # for i, y in enumerate(ys):
    #     ax[i].set_title(
    #         f"$\\theta={pars[i].theta:.2f}, \mu={pars[i].mu:.2f}, \Delta t={pars[i].dt:.2f}, \sigma={pars[i].sigma:.2f}$"
    #     )
    #     ax[i].plot(t, y)
    #
    # fig.subplots_adjust(top=0.85, hspace=0.9)
    # plt.show()

    # ret = np.maximum([-2, -1], [-1.5, -2])
    # print(ret)
    # ret = (np.array([1,1,1]) == np.array([1,1,2])).all()
    # print(ret)

    # arr = spatialmath.SE3.Ry(30, "deg")
    # print(type(arr))
    # print(arr.shape)
    # print(arr.t.shape)
    # eul = spatialmath.SO3.rpy(arr, 'deg')
    # print(type(eul))
    # print(eul)
    # print(np.zeros([3, 1]))
    # robot = rtb.models.DH.Panda()
    # print(robot)

    # r = R.from_euler('x', 90, degrees=True)
    # m = r.as_matrix()
    # print(m)
    # print(r.as_euler("xyz", degrees=True))

    # config = np.ones(7)
    # print(config)
    # robot = K()
    # se3 = robot.forward_kinematic(config, np.zeros(7))
    # print(se3)

    # a = np.array([2, 3, 34])
    # b = np.array([1, 2, 3])
    # c = np.array([1, 2, 3])
    # print(a)
    # ret1 = (b < a).all()
    # ret2 = (a == b).all()
    # ret3 = (a == c).any()
    # print(ret1, ret2, ret3)

    # arr = np.eye(4)
    # print(arr[2, 0:3].shape)

    # noise = OUNoise(2)
    # for i in range(100):
    #     print(noise.noise())
    #     time.sleep(1)

    # for i in range(100):
    #     n = np.random.normal(0,1)
    #     print(n)
    #     time.sleep(1)

    # fruit1 = FruitFactory("apple")
    # # fruit2 = FruitFactory("orange")
    # ret = np.clip([-2, 1], [-1.5, 0], [0, 0.5])
    # print(ret)

    # ou = OUNoise(7)
    # states = []
    # for i in range(1000):
    #     states.append(ou.noise())
    # import matplotlib.pyplot as plt
    #
    # plt.plot(states)
    # plt.show()

    # ee_se3 = np.eye(4)
    # tr = np.array([
    #     [2, 0, 0, 0],
    #     [0, 2, 0, 0],
    #     [0, 0, 2, 0],
    #     [0, 0, 0, 2]
    # ])
    # ret = np.matmul(ee_se3, tr)
    # print(ret)
    # print(link_gaussian_noise(0,1,7).shape)
    # print(tr[0, :].shape)

    # so3 = np.eye(3)
    # so3 = np.reshape(so3, (1, -1), order="F").squeeze(axis=0)
    # so3 = list(so3)
    # rpy = math.so3.rpy(so3)
    # print(rpy)
