#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime
import numpy as np
import os
import time

class Plotter1(object):
    def __init__(self):
        plt.ion()
        # Create figures
        self.fig1 = plt.figure(1, figsize=[10.4, 4.8])
        self.fig1.suptitle('Real time xyz while training')
        self.xy_graphic = self.fig1.add_subplot(1, 2, 1)
        self.zy_graphic = self.fig1.add_subplot(1, 2, 2)
        # figure 2
        self.fig2 = plt.figure(2, figsize=[10.4, 4.8])
        self.fig2.suptitle('Real time rpy while training')
        self.rp_graphic = self.fig2.add_subplot(1, 2, 1)
        self.yp_graphic = self.fig2.add_subplot(1, 2, 2)
        # figure 3
        self.fig3 = plt.figure(5, figsize=[9.6, 6.4])
        # create attributes
        self.start_x = 0
        self.start_y = 0
        self.start_z = 0
        self.tar_x = 0
        self.tar_y = 0
        self.tar_z = 0
        self.tar_roll = 0
        self.tar_pitch = 0
        self.tar_yaw = 0
        # display pose figure templates.
        self._xyz_figure()
        self._rpy_figure()
        self._xyz_3d_figure()

    def start_target_cartesian_pose(self, start_cartesian_pose, target_cartesian_pose):
        # start cartesian pose
        self.start_x = start_cartesian_pose[0]
        self.start_y = start_cartesian_pose[1]
        self.start_z = start_cartesian_pose[2]
        self.start_roll = start_cartesian_pose[3]
        self.start_pitch = start_cartesian_pose[4]
        self.start_yaw = start_cartesian_pose[5]
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
        self.xy_graphic.plot(self.start_x, self.start_y, '.g')
        self.xy_graphic.plot(self.tar_x, self.tar_y, '.r')
        self.zy_graphic.plot(self.start_z, self.start_y, '.g')
        self.zy_graphic.plot(self.tar_z, self.tar_y, '.r')

        self.xy_graphic.plot(path_xyz[0], path_xyz[1], '-b', linewidth=1)
        self.zy_graphic.plot(path_xyz[2], path_xyz[1], '-b', linewidth=1)

    def rpy_figure(self, path_rpy):
        self.rp_graphic.cla()
        self.yp_graphic.cla()
        self._rpy_figure()
        # rpy figure subplots
        self.rp_graphic.plot(self.start_roll, self.start_pitch, '.g')
        self.rp_graphic.plot(self.tar_roll, self.tar_pitch, '.r')
        self.yp_graphic.plot(self.start_yaw, self.start_pitch, '.g')
        self.yp_graphic.plot(self.tar_yaw, self.tar_pitch, '.r')

        self.rp_graphic.plot(path_rpy[0], path_rpy[1], '.b', markersize=0.3)
        self.yp_graphic.plot(path_rpy[2], path_rpy[1], '.b', markersize=0.3)
        
    def xyz_3d_figure(self, path_xyz):
        self.ax.cla()
        self._xyz_3d_figure()
        self.ax.plot(self.start_x, self.start_y, self.start_z, '.g')
        self.ax.plot(self.tar_x, self.tar_y, self.tar_z, '.r')
        self.ax.plot(path_xyz[0], path_xyz[1], path_xyz[2], '-b', linewidth=1)

    def _xyz_figure(self):
        with plt.style.context(['science', 'ieee']):
            self.xy_graphic.set_title('Position in x-y axis')
            self.xy_graphic.set_xlabel('x-axis/[m]')
            self.xy_graphic.set_ylabel('y-axis/[m]')
            self.xy_graphic.set_xlim((-0.1, 1.1))
            self.xy_graphic.set_ylim((-0.6, 0.6))
            self.xy_graphic.grid()
            # (0.4 <= x <= 1.0) && (-0.5 <= y <= 0.5)
            self.xy_graphic.add_patch(Rectangle((0.0, -0.5), 1, 1,
                            alpha=1, color='y',ls='-.', lw=1.5, fc='none'))

            self.zy_graphic.set_title('Position in z-y axis')
            self.zy_graphic.set_xlabel('z-axis/[m]')
            self.zy_graphic.set_ylabel('y-axis/[m]')
            self.zy_graphic.set_xlim((-0.1, 1.1))
            self.zy_graphic.set_ylim((-0.6, 0.6))
            self.zy_graphic.grid()
            # (0.01 <= z <= 0.7) && (-0.5 <= y <= 0.5)
            self.zy_graphic.add_patch(Rectangle((0.0, -0.5), 1, 1,
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

    def _xyz_3d_figure(self):
        with plt.style.context(['science', 'ieee']):
            self.ax = self.fig3.gca(projection='3d')
            self.ax.set_title('Real time 3D figure of xyz while training')
            self.ax.set_xlabel('x-axis/[m]')
            self.ax.set_ylabel('y-axis/[m]')
            self.ax.set_zlabel('z-axis/[m]')
            self.ax.set_xlim((0.0, 1.0))
            self.ax.set_ylim((-0.5, 0.5))
            self.ax.set_zlim((0.0, 1.0))
            self.ax.grid()
            plt.draw()

    def save_pose_figure(self, dir_name):
        date = str(datetime.date(datetime.now()))
        time = str(datetime.time(datetime.now()))
        checkpoint_dir = "src/training_algorithms/scripts/" + dir_name + "/path_figures"
        figure_name_xyz = 'path_xyz_' + date + '_' + time
        figure_name_rpy = 'path_rpy_' + date + '_' + time
        figure_name_xyz_3d = 'path_xyz_3d_' + date + '_' + time
        checkpoint_file_xyz = os.path.join(checkpoint_dir, figure_name_xyz)
        checkpoint_file_rpy = os.path.join(checkpoint_dir, figure_name_rpy)
        checkpoint_file_xyz_3d = os.path.join(checkpoint_dir, figure_name_xyz_3d)
        print("Saving pose figure to " + dir_name + " ...")
        self.fig1.savefig(checkpoint_file_xyz + '.png', dpi=600)
        self.fig1.savefig(checkpoint_file_xyz + '.pdf', dpi=600)
        self.fig2.savefig(checkpoint_file_rpy + '.png', dpi=600)
        self.fig2.savefig(checkpoint_file_rpy + '.pdf', dpi=600)
        self.fig3.savefig(checkpoint_file_xyz_3d + '.png', dpi=600)
        self.fig3.savefig(checkpoint_file_xyz_3d + '.pdf', dpi=600)


class Plotter2(object):
    def __init__(self):
        plt.ion()
        # Create figures
        self.fig1 = plt.figure(3, figsize=[12, 8.8])
        self.fig1.suptitle('Learning curves of training process')
        self.score_graphic = self.fig1.add_subplot(2, 1, 1)
        self.step_num_graphic = self.fig1.add_subplot(2, 1, 2)
        self.fig2 = plt.figure(4, figsize=[12, 5.6])
        self.error_graphic = self.fig2.gca()
        # display learning curve figure templates.
        self._learning_curve_figure()
        self._error_figure(self.fig2)

    def learning_curve_figure(self, iteration, avg_score, step_num):
        self.score_graphic.cla()
        self.step_num_graphic.cla()
        self._learning_curve_figure()
        self.score_graphic.plot(iteration, avg_score, '-r', linewidth=0.5)
        self.step_num_graphic.plot(iteration, step_num, '-r', linewidth=0.5)
        self.score_graphic.grid()
        self.step_num_graphic.grid()

    def error_figure(self, iteration, error_norm):
        self.error_graphic.cla()
        self._learning_curve_figure()
        self.error_graphic.plot(iteration, error_norm, '-r', linewidth=0.5)
        self.error_graphic.grid()

    def _learning_curve_figure(self):
        with plt.style.context(['science', 'ieee']):
            self.score_graphic.set_title('Obtained score per iteration')
            self.score_graphic.set_xlabel('iteration')
            self.score_graphic.set_ylabel('avg score')
            self.score_graphic.grid()

            self.step_num_graphic.set_title('Learning step number per iteration')
            self.step_num_graphic.set_xlabel('iteration')
            self.step_num_graphic.set_ylabel('step number')
            self.step_num_graphic.grid()
            plt.draw()

    def _error_figure(self, figure):
        with plt.style.context(['science', 'ieee']):
            self.error_graphic.set_title('The distance between TCP and goal position')
            self.error_graphic.set_xlabel('iteration')
            self.error_graphic.set_ylabel('error/[mm]')
            self.error_graphic.grid()
            plt.draw()
    
    def save_learning_curve_figure(self, dir_name):
        date = str(datetime.date(datetime.now()))
        time = str(datetime.time(datetime.now()))
        checkpoint_dir = "src/training_algorithms/scripts/" + dir_name + "/learning_curve_figures"
        figure_name = date + '_' + time
        checkpoint_file = os.path.join(checkpoint_dir, figure_name)
        print("Saving learning_curve_figure to " + dir_name + " ...")
        self.fig1.savefig(checkpoint_file + '.png', dpi=600)
        self.fig1.savefig(checkpoint_file + '.pdf')

    def save_error_figure(self, dir_name):
        date = str(datetime.date(datetime.now()))
        time = str(datetime.time(datetime.now()))
        checkpoint_dir = "src/training_algorithms/scripts/" + dir_name + "/error_figures"
        figure_name = date + '_' + time
        checkpoint_file = os.path.join(checkpoint_dir, figure_name)
        print("Saving error_figure to " + dir_name + " ...")
        self.fig2.savefig(checkpoint_file + '.png', dpi=600)
        self.fig2.savefig(checkpoint_file + '.pdf')

