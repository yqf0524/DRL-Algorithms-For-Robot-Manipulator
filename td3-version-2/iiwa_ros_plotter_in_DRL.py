#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime
import numpy as np
import os
import time

class Plotter1(object):
    def __init__(self, num1, num2, dir_name):
        self.dir_name = dir_name
        plt.ion()
        # Create figures
        self.fig1 = plt.figure(num1, figsize=[10.4,4.8])
        self.fig1.suptitle('Real time xyz while training')
        self.xy_graphic = self.fig1.add_subplot(1, 2, 1)
        self.zy_graphic = self.fig1.add_subplot(1, 2, 2)

        self.fig2 = plt.figure(num2, figsize=[10.4,4.8])
        self.fig2.suptitle('Real time rpy while training')
        self.rp_graphic = self.fig2.add_subplot(1, 2, 1)
        self.yp_graphic = self.fig2.add_subplot(1, 2, 2)

        self.tar_x = 0
        self.tar_y = 0
        self.tar_z = 0
        self.tar_roll = 0
        self.tar_pitch = 0
        self.tar_yaw = 0
        # display pose figure templates.
        self._xyz_figure()
        self._rpy_figure()

    def target_cartesian_pose(self, target_cartesian_pose):
        # target cartesian pose
        self.tar_x = target_cartesian_pose[0]
        self.tar_y = target_cartesian_pose[1]
        self.tar_z = target_cartesian_pose[2]
        self.tar_roll = target_cartesian_pose[3]
        self.tar_pitch = target_cartesian_pose[4]
        self.tar_yaw = target_cartesian_pose[5]


    def xyz_figure(self, path_xyz):
        self.xy_graphic.cla()
        self.zy_graphic.cla()
        self._xyz_figure()
        # xyz figure subplots
        self.xy_graphic.plot(path_xyz[0], path_xyz[1], '-b', linewidth=1)
        self.zy_graphic.plot(path_xyz[2], path_xyz[1], '-b', linewidth=1)
        self.xy_graphic.plot(self.tar_x, self.tar_y, 'or')
        self.zy_graphic.plot(self.tar_z, self.tar_y, 'or')

    def rpy_figure(self, path_rpy):
        self.rp_graphic.cla()
        self.yp_graphic.cla()
        self._rpy_figure()
        # rpy figure subplots
        self.rp_graphic.plot(path_rpy[0], path_rpy[1], '-b', linewidth=1)
        self.yp_graphic.plot(path_rpy[2], path_rpy[1], '-b', linewidth=1)
        self.rp_graphic.plot(self.tar_roll, self.tar_pitch, 'or')
        self.yp_graphic.plot(self.tar_yaw, self.tar_pitch, 'or')

    def _xyz_figure(self):
        with plt.style.context(['science', 'ieee']):
            self.xy_graphic.set_title('Position in x-y axis')
            self.xy_graphic.set_xlabel('x-axis/[m]')
            self.xy_graphic.set_ylabel('y-axis/[m]')
            self.xy_graphic.set_xlim((0.3, 1.1))
            self.xy_graphic.set_ylim((-0.6, 0.6))
            self.xy_graphic.grid()
            # (0.4 <= x <= 1.0) && (-0.5 <= y <= 0.5)
            self.xy_graphic.add_patch(Rectangle((0.4, -0.5), 0.6, 1,
                            alpha=1, color='y',ls='-.', lw=1.5, fc='none'))

            self.zy_graphic.set_title('Position in z-y axis')
            self.zy_graphic.set_xlabel('z-axis/[m]')
            self.zy_graphic.set_ylabel('y-axis/[m]')
            self.zy_graphic.set_xlim((-0.1, 0.8))
            self.zy_graphic.set_ylim((-0.6, 0.6))
            self.zy_graphic.grid()
            # (0.01 <= z <= 0.7) && (-0.5 <= y <= 0.5)
            self.zy_graphic.add_patch(Rectangle((0.0, -0.5), 0.7, 1,
                            alpha=1, color='y',ls='-.', lw=1.5, fc='none'))
            plt.draw()
                            
    def _rpy_figure(self):
        with plt.style.context(['science', 'ieee']):
            self.rp_graphic.set_title('Orientation in roll-pitch axis')
            self.rp_graphic.set_xlabel('roll-axis/[rad]')
            self.rp_graphic.set_ylabel('pitch-axis/[rad]')
            self.rp_graphic.set_xlim((-3.5, 3.5))
            self.rp_graphic.set_ylim((-3.5, 3.5))
            self.rp_graphic.grid()
            # (-pi <= roll <= pi) && (-pi <= pitch <= pi)
            self.rp_graphic.add_patch(Rectangle((-np.pi, -np.pi), 2 * np.pi, 2 * np.pi,
                            alpha=1, color='y',ls='-.', lw=1.5, fc='none'))

            self.yp_graphic.set_title('Orientation in yaw-pitch axis')
            self.yp_graphic.set_xlabel('yaw-axis/[rad]')
            self.yp_graphic.set_ylabel('pitch-axis/[rad]')
            self.yp_graphic.set_xlim((-3.5, 3.5))
            self.yp_graphic.set_ylim((-3.5, 3.5))
            self.yp_graphic.grid()
            # (-pi <= yaw <= pi) &7 (-pi <= pitch <= pi)
            self.yp_graphic.add_patch(Rectangle((-np.pi, -np.pi), 2 * np.pi, 2 * np.pi,
                            alpha=1, color='y',ls='-.', lw=1.5, fc='none'))
            plt.draw()

    def save_pose_figure(self):
        date = str(datetime.date(datetime.now()))
        time = str(datetime.time(datetime.now()))
        checkpoint_dir = "src/training_algorithms/scripts/" + self.dir_name + "/pose_figures"
        figure_name_xyz = 'xyz_path_' + date + '_' + time
        figure_name_rpy = 'rpy_path_' + date + '_' + time
        checkpoint_file_xyz = os.path.join(checkpoint_dir, figure_name_xyz)
        checkpoint_file_rpy = os.path.join(checkpoint_dir, figure_name_rpy)
        print("Saving pose figure ...")
        self.fig1.savefig(checkpoint_file_xyz + '.png', dpi=300)
        self.fig1.savefig(checkpoint_file_xyz + '.pdf')
        self.fig2.savefig(checkpoint_file_rpy + '.png', dpi=300)
        self.fig2.savefig(checkpoint_file_rpy + '.pdf')


class Plotter2(object):
    def __init__(self, num1, num2, dir_name):
        self.dir_name = dir_name
        plt.ion()
        # Create figures
        self.fig1 = plt.figure(num1, figsize=[12, 7.2])
        self.fig2 = plt.figure(num2, figsize=[12, 5.6])
        # display pose figure templates.
        self._learning_curve_figure(self.fig1)
        self._time_consume_figure(self.fig2)

    def learning_curve_figure(self, iteration, score, step_num):
        self.score_graphic.plot(iteration, score, '-r', linewidth=1)
        self.step_num_graphic.plot(iteration, step_num, '-r', linewidth=1)

    def _learning_curve_figure(self, figure):
        figure.suptitle('Learning curves of training process')
        with plt.style.context(['science', 'ieee']):
            self.score_graphic = figure.add_subplot(2, 1, 1)
            self.score_graphic.set_title('Obtained score per iteration')
            self.score_graphic.set_xlabel('iteration')
            self.score_graphic.set_ylabel('score')
            self.score_graphic.grid()

            self.step_num_graphic = figure.add_subplot(2, 1, 2)
            self.step_num_graphic.set_title('Learning step number per iteration')
            self.step_num_graphic.set_xlabel('iteration')
            self.step_num_graphic.set_ylabel('step number')
            self.step_num_graphic.grid()
            plt.draw()

    def save_learning_curve_figure(self):
        date = str(datetime.date(datetime.now()))
        time = str(datetime.time(datetime.now()))
        checkpoint_dir = "src/training_algorithms/scripts/" + self.dir_name + "/learning_curve_figures"
        figure_name = 'learning_curve_figure_' + date + '_' + time
        checkpoint_file = os.path.join(checkpoint_dir, figure_name)
        print("Saving learning_curve_figure ...")
        self.fig1.savefig(checkpoint_file + '.png', dpi=300)
        self.fig1.savefig(checkpoint_file + '.pdf')

    def time_consume_figure(self, iteration, time_consume):
        self.time_consume_graphic.plot(iteration, time_consume, '-r', linewidth=1)

    def _time_consume_figure(self, figure):
        with plt.style.context(['science', 'ieee']):
            self.time_consume_graphic = figure.gca()
            self.time_consume_graphic.set_title('Total time comsume of training process')
            self.time_consume_graphic.set_xlabel('iteration')
            self.time_consume_graphic.set_ylabel('total time comsumes/[sec]')
            self.time_consume_graphic.grid()
            self.time_consume_graphic.plot(0, 0, '-r')
            plt.draw()
    
    def save_time_consume_figure(self):
        date = str(datetime.date(datetime.now()))
        time = str(datetime.time(datetime.now()))
        checkpoint_dir = "src/training_algorithms/scripts/" + self.dir_name + "/time_consume_figures"
        figure_name = 'time_consume_figure_' + date + '_' + time
        checkpoint_file = os.path.join(checkpoint_dir, figure_name)
        print("Saving total_time_consumes_figure...")
        self.fig2.savefig(checkpoint_file + '.png', dpi=300)
        self.fig2.savefig(checkpoint_file + '.pdf')

# plot2 = Plotter1(1,2,'TD3_training_foreward_plots')
# x = [[0.4, 0.7, 0.5], [0.4, 0.3, 0.1], [0.5, 0.2, 0.1]]

# plot2.xyz_figure(x)


# plot = Plotter2(1,2,'TD3_training_foreward_plots')
# plt.pause(10)
