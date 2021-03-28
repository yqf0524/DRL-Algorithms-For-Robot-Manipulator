#!/usr/bin/env python3


import pandas as pd

df = pd.read_excel("training_algorithms/scripts/data/path_cartesian_space_forward.xlsx", sheet_name='path_1', header=0, index_col=0, engine='openpyxl')
arr = df.values
print(int(arr[0]))









# import matplotlib.pyplot as plt
# from iiwa_ros_data_process_in_DRL import DataProcessing
# from iiwa_ros_plotter_in_DRL import Plotter1

# plot = Plotter1(1, 2, 3)
# dp = DataProcessing('forward')
# plot.save_pose_figure('SAC_training_backward_plots')

# plt.pause(10)

















# import numpy as np
# import pandas as pd
# a = np.array([1,2,3,4,5,6,7])[np.newaxis, :]
# a = np.concatenate((a,-a),axis=0)
# a_df = pd.DataFrame(a)
# writer = pd.ExcelWriter('training_algorithms/scripts/test.xlsx')
# a_df.to_excel(writer, sheet_name='config_'+str(1))
# writer.save()



# r_df = pd.read_excel('training_algorithms/scripts/test.xlsx', sheet_name='config_1')
# print(r_df)







# import numpy as np
# import torch as T

# t1 = T.Tensor([[1,2,3,4],[2,3,4,5]])
# print(t1)
# print(t1.view(-1))



















# from iiwa_ros_plotter_in_DRL  import Plotter1, Plotter2
# import matplotlib.pyplot as plt
# from matplotlib import animation
# from matplotlib.patches import Rectangle

# import numpy as np




# figure = plt.figure('Test figure')

# cur_x = []
# cur_y = []
# cur_z = []

# ims_xyz = []
# ims_rpy = []
# xy_graphic = figure.add_subplot(1, 2, 1)
# zy_graphic = figure.add_subplot(1, 2, 2)



# def _xyz_figure(figure):
#         figure.suptitle('Real time xyz while training')
#         with plt.style.context(['science', 'ieee']):
#             xy_graphic.set_title('Position in x-y axis')
#             xy_graphic.set_xlabel('x-axis/[m]')
#             xy_graphic.set_ylabel('y-axis/[m]')
#             xy_graphic.set_xlim((0.3, 1.1))
#             xy_graphic.set_ylim((-0.6, 0.6))
#             xy_graphic.grid()
#             # (0.4 <= x <= 1.0) && (-0.5 <= y <= 0.5)
#             xy_graphic.add_patch(Rectangle((0.4, -0.5), 0.6, 1,
#                             alpha=1, color='y',ls='-.', lw=1.5, fc='none'))

#             zy_graphic.set_title('Position in z-y axis')
#             zy_graphic.set_xlabel('z-axis/[m]')
#             zy_graphic.set_ylabel('y-axis/[m]')
#             zy_graphic.set_xlim((-0.1, 0.8))
#             zy_graphic.set_ylim((-0.6, 0.6))
#             zy_graphic.grid()
#             # (0.01 <= z <= 0.7) && (-0.5 <= y <= 0.5)
#             zy_graphic.add_patch(Rectangle((0.0, -0.5), 0.7, 1,
#                             alpha=1, color='y',ls='-.', lw=1.5, fc='none'))
#             plt.draw()



# def xyz_figure(current_ee_xyz):
#     cur_x.append(current_ee_xyz[0])
#     cur_y.append(current_ee_xyz[1])
#     cur_z.append(current_ee_xyz[2])
#     # xyz figure subplots
#     im1 = xy_graphic.plot(cur_x, cur_y, '-ob')
#     im2 = zy_graphic.plot(cur_z, cur_y, '-ob')
#     ims_xyz.append(im1)
#     ims_rpy.append(im2)


# _xyz_figure(figure)

# for i in range(50):
#     xyz = np.random.random(size=3)
#     xyz_figure(xyz)
#     plt.pause(0.1)


# ani = animation.ArtistAnimation(figure, ims_xyz, interval=100, repeat_delay=50)
# ani.save("test_xyz.gif", writer='pillow')
# ani = animation.ArtistAnimation(figure, ims_rpy, interval=100, repeat_delay=50)
# ani.save("test_rpy.gif", writer='pillow')


# plt.pause(5)



# t = np.arange(100)
# y = np.random.normal(size=len(t))
# fig = plt.figure('Test figure')
# time_consume_graphic = fig.gca()

# plt.ion()

# def plot(time_consume_graphic):
#     with plt.style.context(['science', 'ieee']):
#         time_consume_graphic.set_title('Total time comsume of training process')
#         time_consume_graphic.set_xlabel('iteration')
#         time_consume_graphic.set_ylabel('total time comsumes/[sec]')
#         time_consume_graphic.grid()
#         plt.draw()


# for i in range(20):
#     if i == 10:
#         plt.cla()
#         plot(time_consume_graphic)
#         y = np.random.normal(size=len(t))

#     im = time_consume_graphic.plot(t[0:i], y[0:i], '-r')
#     ims.append(im)
#     plt.pause(0.1)

# ani = animation.ArtistAnimation(fig, ims, interval=200, repeat_delay=50)
# ani.save("test.gif", writer='pillow')
        
# plt.pause(5)






















# import numpy as np
# import matplotlib.pyplot as plt

# # t = np.arange(0.0001,0.8, 0.0004)
# # y = np.log2(t)

# # with plt.style.context(['science']):
# #     plt.plot(t, y)
# #     plt.show()

# error = 0.206
# for i in range(1000):
#     error /= 1.05
#     if error <= 0.0001:
#         print(i)
#         break





















# from datetime import datetime

# print(datetime.now())
# print(datetime.date(datetime.now()))
# print(datetime.time(datetime.now()))

# time = str(datetime.now())

# print(time)
















# from multiprocessing import Pipe, Queue
# import numpy as np

# conn1, conn2 = Pipe()
# queue = Queue(maxsize=4)

# arr = np.array([1,2,3])
# arr = "arra"
# conn1.send(arr)
# print(arr is "arr")
# print(conn2.recv() == "arr")



# conn1.send(arr)
# conn1.send(2)
# conn1.send(3)
# conn1.send(4)
# conn1.send(5)

# for i in range(4):
#     if conn2.recv() == "equal":
#         print('test success.')
#     print(conn2.recv())

























# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# import numpy as np

# figure = plt.figure(4)
# ax = figure.gca()
# # ax.add_patch
# plt.ion()
# def _score_figure(figure):
#     ax = figure.add_subplot(2, 1, 1)
#     ax.set_title("Obtained score per iteration")
#     ax.set_xlabel("iteration")
#     ax.set_ylabel("score")
#     ax.grid()
#     # ax.plot(0, 0, '-or')
#     ax.add_patch(Rectangle((0,0),0.1,0.1,alpha=1, color='y',ls='-.', lw=1.5, fc='none'))
#     plt.draw()

#     figure.savefig('example.png', dpi=1200)

# _score_figure(figure)


# y = []
# x = []
# for i in range(100):
#     rand = np.random.normal(size=(1))
#     y.append(rand[0])
#     x.append(i)
#     ax.plot(x, y, '-.ob')
#     plt.pause(20)
# plt.pause(2)
# _score_figure(figure)
# plt.pause(6)









# from multiprocessing import Queue
# import time

# queue = Queue()

# # iteration, score, step_num, time_consume, is_achieved
# a = [10, 200, 400, 100, False]
# print(a)
# queue.get()
# queue.put(a)
# time.sleep(3)
# a_get = queue.get()
# print(a_get)
















# import matplotlib.pyplot as plt
# import numpy as np
# import time

# start_pose = [0.6, 0.3, 0.25, -3.14, 0, 3.14]
# target_position = [0.55, 0.2, 0.2]

# plt.ion()
# fig = plt.figure(1)
# fig.suptitle('Cartesian pose while training')
# xy_graphic = fig.add_subplot(2, 2, 1)
# xy_graphic.set_title('Pose in x-y axis')
# xy_graphic.set_xlabel('x-axis')
# xy_graphic.set_ylabel('y-axis')
# xy_graphic.set_xlim((0.4, 0.9))
# xy_graphic.set_ylim((-0.5, 0.5))
# xy_graphic.grid()
# plt.draw()

# for i in range(100):
#     x = np.random.uniform(0.5, 0.8)
#     y = np.random.uniform(-0.5, 0.5)
#     z = np.random.uniform(0, 0.7)
#     a = np.random.uniform(-3.14, 3.14)
#     b = np.random.uniform(-3.14, 3.14)
#     c = np.random.uniform(-3.14, 3.14)
    
#     # plt.clf()  # clear everything on the figure.

#     xy_graphic.plot(x, y, '.b', target_position[0], target_position[1], '.r')

#     # zy_graphic = fig.add_subplot(2, 2, 2)
#     # zy_graphic.set_title('Position in x-y axis')
#     # zy_graphic.set_xlabel('z-axis')
#     # zy_graphic.set_ylabel('y-axis')
#     # zy_graphic.set_xlim((-0.1, 0.8))
#     # zy_graphic.set_ylim((-0.5, 0.5))
#     # zy_graphic.grid()
#     # zy_graphic.plot(z, y, '.b', target_position[2], target_position[1], '.r')

#     # rp_graphic = fig.add_subplot(2, 2, 3)
#     # rp_graphic.set_title('Orientation in roll-pitch axis')
#     # rp_graphic.set_xlabel('roll-axis')
#     # rp_graphic.set_ylabel('pitch-axis')
#     # rp_graphic.set_xlim((-3.5, 3.5))
#     # rp_graphic.set_ylim((-3.5, 3.5))
#     # rp_graphic.grid()
#     # rp_graphic.plot(a, b, '.b', start_pose[3], start_pose[4], '.r')

#     # yp_graphic = fig.add_subplot(2, 2, 4)
#     # yp_graphic.set_title('Orientation in roll-pitch axis')
#     # yp_graphic.set_xlabel('yaw-axis')
#     # yp_graphic.set_ylabel('pitch')
#     # yp_graphic.set_xlim((-3.5, 3.5))
#     # yp_graphic.set_ylim((-3.5, 3.5))
#     # yp_graphic.grid()
#     # yp_graphic.plot(c, b, '.b', start_pose[5], start_pose[4], '.r')
#     plt.pause(0.01)























# import numpy as np
# from multiprocessing import Pipe


# A = np.array([1,2,3,4])
# B = np.array([2,1,3,2])
# mse = ((A - B) ** 2).mean(axis=0)
# print(mse)

# conn1, conn2 = Pipe()

# conn1.send(True)

# if conn2.recv():
#     print("Received signal from conn1.")














# from datetime import datetime

# print(datetime.now())
# print(datetime.date(datetime.now()))
# print(datetime.time(datetime.now()))


















# import numpy as np

# def z_score_norm(obs, axis=0):
#     obs_mean = np.mean(obs, axis=axis)
#     print(obs_mean)
#     obs_std = np.std(obs, axis=axis)
#     print(obs_std)
#     obs_norm = np.divide(obs - obs_mean, obs_std)
#     print(obs_norm)
#     return obs_norm


# obs = np.array([[-1,0,3], [-20,0,20]])
# print(obs)
# z_score_norm(obs[0], axis=0)

# print(np.divide(obs, [2,2,2]))

# print(np.random.normal(scale=0.02, size=(10, 7)))












# from iiwa_DRL_environment import PositionControl
# from iiwa_ros_plotter_in_DRL import Plotter
# import numpy as np
# import rospy
# import time


# if __name__ == '__main__':
#     env = PositionControl()
    
#     # Set target_position and target_pose for DRL training.
#     target_ee_position = env.iiwa.start_ee_position + np.array([0.15, 0.1, -0.1])
#     target_ee_rpy = env.iiwa.start_ee_rpy
#     env.iiwa.set_target_ee_pose(target_ee_position, target_ee_rpy)
#     print(env.iiwa.start_ee_pose)
#     print(env.iiwa.target_ee_pose)

#     plot = Plotter(env.iiwa.target_ee_pose)

#     while not rospy.is_shutdown():
#         # conn2 receives the current catesian pose
#         current_pose = np.random.random(6)
#         plot.pose_figure(current_pose)
#     plt.ioff()











# import numpy as np


# arr1 = np.array([1,1,1])
# arr2 = np.array([2,3,2])
# arr = np.concatenate((arr1, arr2))
# print(arr)

# s = 'good: fine'
# ss = s.split(':')
# print(ss)

# init_configuration = [1]
# if init_configuration is not None:
#     print('pass')

# a1 = np.array([1,2,-3])
# a2 = np.array([-2,1,1])
# print(np.linalg.norm(a1))
# print(np.linalg.norm(np.abs(a1)))

# print(a2-a1)
# print(a1+a2)

# m1 = np.min((np.abs(a2-a1), np.abs(a1+a2)),axis=0)
# print(m1)











# from multiprocessing import Process, Pipe
# import time

# def send1(conn):
#     for i in range(100):
#         conn.send(i)
#         print('p1 send data ...',i)
#         time.sleep(1)

# def recv1(conn):
#     for i in range(100):
#         print('p2 recv data ...', conn.recv())
#         time.sleep(0.5)

# def send2(conn):
#     for i in range(100):
#         conn.send([1,2,-4,-5])
#         print('p3 send data ...')
#         time.sleep(1)

# def recv2(conn):
#     for i in range(100):
#         print('p4 recv data ...', conn.recv())
#         time.sleep(1.5)

# if __name__ == '__main__':
#     conn1, conn2 = Pipe()
#     # p1 = Process(target=send1, args=(conn1,))
#     # p2 = Process(target=recv1, args=(conn2,))
#     p3 = Process(target=send1, args=(conn1,))
#     p4 = Process(target=recv2, args=(conn2,))
#     # p1.start()
#     # p2.start()
#     p3.start()
#     p4.start()

#     for i in range(100):
#         time.sleep(2)
#         print('waiting ...\n')
























# import numpy as np

# num = 0
# num1 = 0
# iter = 100
# for i in range(iter):
#     noise = np.random.normal(scale=1, size=1)
#     if noise <= 1:
#         num += 1
#     if noise >= 0.2:
#         num1 += 1
#     print(noise)
    

# print(num/iter)
# print(num1/iter)











# import matplotlib.pyplot as plt
# import numpy as np
# import time

# start_pose = [0.6, 0.3, 0.25, -3.14, 0, 3.14]
# target_position = [0.55, 0.2, 0.2]

# plt.ion()
# plt.figure(1)
# for i in range(100):
#     x = np.random.uniform(0.5, 0.8)
#     y = np.random.uniform(-0.5, 0.5)
#     z = np.random.uniform(0, 0.7)
#     a = np.random.uniform(-3.14, 3.14)
#     b = np.random.uniform(-3.14, 3.14)
#     c = np.random.uniform(-3.14, 3.14)
    
#     # plt.clf()  # clear everything on the figure.
#     plt.suptitle('Cartesian pose while training')
#     xy_graphic = plt.subplot(2, 2, 1)
#     xy_graphic.set_title('Pose in x-y axis')
#     xy_graphic.set_xlabel('x-axis')
#     xy_graphic.set_ylabel('y-axis')
#     xy_graphic.set_xlim((0.4, 0.9))
#     xy_graphic.set_ylim((-0.5, 0.5))
#     plt.grid()
#     plt.plot(x, y, '.b', target_position[0], target_position[1], '.r')

#     zy_graphic = plt.subplot(2, 2, 2)
#     zy_graphic.set_title('Position in x-y axis')
#     zy_graphic.set_xlabel('z-axis')
#     zy_graphic.set_ylabel('y-axis')
#     zy_graphic.set_xlim((-0.1, 0.8))
#     zy_graphic.set_ylim((-0.5, 0.5))
#     plt.grid()
#     plt.plot(z, y, '.b', target_position[2], target_position[1], '.r')

#     rp_graphic = plt.subplot(2, 2, 3)
#     rp_graphic.set_title('Orientation in roll-pitch axis')
#     rp_graphic.set_xlabel('roll-axis')
#     rp_graphic.set_ylabel('pitch-axis')
#     rp_graphic.set_xlim((-3.5, 3.5))
#     rp_graphic.set_ylim((-3.5, 3.5))
#     plt.grid()
#     plt.plot(a, b, '.b', start_pose[3], start_pose[4], '.r')

#     yp_graphic = plt.subplot(2, 2, 4)
#     yp_graphic.set_title('Orientation in roll-pitch axis')
#     yp_graphic.set_xlabel('yaw-axis')
#     yp_graphic.set_ylabel('pitch')
#     yp_graphic.set_xlim((-3.5, 3.5))
#     yp_graphic.set_ylim((-3.5, 3.5))
#     plt.grid()
#     plt.draw()
#     plt.plot(c, b, '.b', start_pose[5], start_pose[4], '.r')
#     plt.pause(0.01)
# plt.ioff()
# plt.show()












# import multiprocessing as mp
# import time

# import matplotlib.pyplot as plt
# import numpy as np

# # Fixing random state for reproducibility
# np.random.seed(19680801)

# class ProcessPlotter(object):
#     def __init__(self):
#         self.x = []
#         self.y = []

#     def terminate(self):
#         plt.close('all')

#     def call_back(self):
#         while self.pipe.poll():
#             command = self.pipe.recv()
#             if command is None:
#                 self.terminate()
#                 return False
#             else:
#                 self.x.append(command[0])
#                 self.y.append(command[1])
#                 self.ax.plot(self.x, self.y, 'ro')
#         self.fig.canvas.draw()
#         return True

#     def __call__(self, pipe):
#         print('starting plotter...')

#         self.pipe = pipe
#         self.fig, self.ax = plt.subplots()
#         timer = self.fig.canvas.new_timer(interval=1000)
#         timer.add_callback(self.call_back)
#         timer.start()

#         print('...done')
#         plt.xlim(-1,1)
#         plt.ylim(-1,1)
#         plt.show()
    
# class NBPlot(object):
#     def __init__(self):
#         self.plot_pipe, plotter_pipe = mp.Pipe()
#         self.plotter = ProcessPlotter()
#         self.plot_process = mp.Process(
#             target=self.plotter, args=(plotter_pipe,), daemon=True)
#         self.plot_process.start()

#     def plot(self, finished=False):
#         send = self.plot_pipe.send
#         if finished:
#             send(None)
#         else:
#             data = np.random.random(2)
#             send(data)


# def main():
#     pl = NBPlot()
#     for ii in range(10):
#         plt.clf()
#         pl.plot()
#         time.sleep(1)
#     pl.plot(finished=True)


# if __name__ == '__main__':
#     if plt.get_backend() == "MacOSX":
#         mp.set_start_method("forkserver")
#     main()





















# from iiwa_RL_environment import PositionControl
# from TD3_Algorithm import Agent as TD3Agent
# from SAC_Algorithm import Agent as SACAgent
# from training_algorithms.msg import RobotConfiguration
# from training_algorithms.msg import CartesianPose
# from iiwa_ros_return_joint_states import DataStream
# from iiwa_ros_control_by_DRL import JointControl
# from iiwa_ros_control_by_DRL import DisplayPose
# import rospy
# import numpy as np
# import time


# def spin_thread():
#     rospy.spin()


# if __name__ == '__main__':
    
#     data = DataStream()
#     # add_thread = threading.Thread(target=spin_thread)
#     # add_thread.start
#     # data.joint_position

#     env = PositionControl()
    
#     # The nodes used in training process.
#     joint_control = JointControl()
#     pose = DisplayPose()
#     init_joint_position = env.iiwa.start_configuration
#     joint_control.publish_joint_position(init_joint_position)

#     env.target_position = np.array([0.13, 0.1, -0.55])

#     print(env.iiwa.start_ee_position)
#     print(env.iiwa.start_ee_rpy)
#     print(env.target_position)
#     print(env.target_rpy)








# from TD3_Algorithm import OrnsteinUhlenbeckNoise
# import time
# import rospy
# from multiprocessing import Process
# import matplotlib.pyplot as plt 


# class PlotFigure:
#     def __init__(self):
#         self.noise = OrnsteinUhlenbeckNoise(7)
#         self.noise_array = [0]
#         self.time = [0]

#     def generate_noise(self,noise):
#         print("start:")
#         while not rospy.is_shutdown():
#             self.noise_array.append(max(noise()))
#             # print(noise.X[0])
#             time.sleep(0.01)
#         print("end program.")

#     def plot(self):
#         plt.ion()
#         plt.figure(1)
#         start_time = time.time()
#         while not rospy.is_shutdown():
#             plt.clf()
#             self.time.append(time.time()-start_time)
#             plt.plot(self.time,self.noise_array,'-r')
#             plt.draw()
#             time.sleep(0.01)


# if __name__ == '__main__':

#     plot = PlotFigure()

#     p1 = Process(target=plot.generate_noise)
#     p2 = Process(target=plot.plot)
#     p1.start()
#     p2.start()










# from iiwa_RL_environment import PositionControl
# from TD3_Algorithm import Agent as TD3Agent
# from SAC_Algorithm import Agent as SACAgent
# from iiwa_ros_return_joint_states import DataStream
# from iiwa_ros_control_by_DRL import JointControl
# from iiwa_ros_control_by_DRL import DisplayPose
# from training_algorithms.msg import RobotConfiguration
# from training_algorithms.msg import CartesianPose
# import rospy
# import numpy as np
# import time


# if __name__ == '__main__':
    
#     # add_thread = threading.Thread(target=spin_thread)
#     # add_thread.start

#     data = DataStream()
#     # data.joint_position
#     # The nodes used in training process.
#     # joint_control = JointControl()
#     # pose = DisplayPose()
#     # init_joint_position = env.iiwa.start_configuration
#     # joint_control.publish_joint_position(init_joint_position)

#     # env = PositionControl()
#     # env.target_position = np.array([0.13, 0.1, -0.55])

#     # rospy.init_node() init only once.
#     joint_position_pub = rospy.Publisher("/iiwa/command/JointPosition", \
#                                     RobotConfiguration, queue_size=10)

#     robot_configuration = RobotConfiguration()
































# from iiwa_RL_environment import PositionControl
# import numpy as np


# env = PositionControl()
# configuration = env.iiwa.start_configuration
# se4 = env.iiwa.forward_kinematic(configuration)

# print(se4)

# """
#  position: 
#     x: 0.695885426245
#     y: 0.020348680666
#     z: 0.224422364775

# """



















# from iiwa_ros_return_joint_states import DataStream
# import rospy
# import threading
# import time

# def spin():
#     rospy.spin()


# if __name__ == '__main__':
#     data = DataStream()

#     add_thread = threading.Thread(target=spin)
#     add_thread.start
#     print("*******")
#     while True:
#         print("in while")
#         print(data.joint_position)
#         time.sleep(0.5)
