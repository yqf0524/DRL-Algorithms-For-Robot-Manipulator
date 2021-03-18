#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime
import numpy as np
import os


class Plotter1(object):
    def __init__(self):
        plt.ion()
        # Create figures
        self.fig1 = plt.figure(1)
        # display pose figure templates.
        self._pose_figure(self.fig1)

        self.cur_x = []
        self.cur_y = []
        self.cur_z = []
        self.cur_roll = []
        self.cur_pitch = []
        self.cur_yaw = []

    def pose_figure(self, current_cartesian_pose):
        self.cur_x.append(current_cartesian_pose[0])
        self.cur_y.append(current_cartesian_pose[1])
        self.cur_z.append(current_cartesian_pose[2])
        self.cur_roll.append(current_cartesian_pose[3])
        self.cur_pitch.append(current_cartesian_pose[4])
        self.cur_yaw.append(current_cartesian_pose[5])
        # Pose figure subplots
        self.xy_graphic.plot(self.cur_x, self.cur_y, '-ob')
        self.zy_graphic.plot(self.cur_z, self.cur_y, '-ob')
        self.rp_graphic.plot(self.cur_roll, self.cur_pitch, '-ob')
        self.yp_graphic.plot(self.cur_yaw, self.cur_pitch, '-ob')

    def set_target_cartesian_pose(self, target_cartesian_pose, training_success):
        if training_success:
            self.save_pose_figure()
        # target cartesian pose
        tar_x = target_cartesian_pose[0]
        tar_y = target_cartesian_pose[1]
        tar_z = target_cartesian_pose[2]
        tar_roll = target_cartesian_pose[3]
        tar_pitch = target_cartesian_pose[4]
        tar_yaw = target_cartesian_pose[5]
        self.fig1.clf()
        self._pose_figure(self.fig1)
        self.xy_graphic.plot(tar_x, tar_y, 'or')
        self.zy_graphic.plot(tar_z, tar_y, 'or')
        self.rp_graphic.plot(tar_roll, tar_pitch, 'or')
        self.yp_graphic.plot(tar_yaw, tar_pitch, 'or')
        plt.draw()

    def _pose_figure(self, figure):
        figure.suptitle('Cartesian pose while training')
        self.xy_graphic = figure.add_subplot(2, 2, 1)
        self.xy_graphic.set_title('Position in x-y axis')
        self.xy_graphic.set_xlabel('x-axis')
        self.xy_graphic.set_ylabel('y-axis')
        self.xy_graphic.set_xlim((0.3, 1.1))
        self.xy_graphic.set_ylim((-0.6, 0.6))
        self.xy_graphic.grid()
        # (0.4 <= x <= 1.0) && (-0.5 <= y <= 0.5)
        self.xy_graphic.add_patch(Rectangle((0.4, -0.5), 0.6, 1,
                        alpha=1, color='y',ls='-.', lw=1.5, fc='none'))

        self.zy_graphic = figure.add_subplot(2, 2, 2)
        self.zy_graphic.set_title('Position in z-y axis')
        self.zy_graphic.set_xlabel('z-axis')
        self.zy_graphic.set_ylabel('y-axis')
        self.zy_graphic.set_xlim((-0.1, 0.8))
        self.zy_graphic.set_ylim((-0.6, 0.6))
        self.zy_graphic.grid()
        # (0.01 <= z <= 0.7) && (-0.5 <= y <= 0.5)
        self.zy_graphic.add_patch(Rectangle((0.0, -0.5), 0.7, 1,
                        alpha=1, color='y',ls='-.', lw=1.5, fc='none'))

        self.rp_graphic = figure.add_subplot(2, 2, 3)
        self.rp_graphic.set_title('Orientation in roll-pitch axis')
        self.rp_graphic.set_xlabel('roll-axis')
        self.rp_graphic.set_ylabel('pitch-axis')
        self.rp_graphic.set_xlim((-3.5, 3.5))
        self.rp_graphic.set_ylim((-3.5, 3.5))
        self.rp_graphic.grid()
        # (-pi <= roll <= pi) && (-pi <= pitch <= pi)
        self.rp_graphic.add_patch(Rectangle((-np.pi, -np.pi), 2 * np.pi, 2 * np.pi,
                        alpha=1, color='y',ls='-.', lw=1.5, fc='none'))

        self.yp_graphic = figure.add_subplot(2, 2, 4)
        self.yp_graphic.set_title('Orientation in yaw-pitch axis')
        self.yp_graphic.set_xlabel('yaw-axis')
        self.yp_graphic.set_ylabel('pitch-axis')
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
        checkpoint_dir = "src/training_algorithms/scripts/TD3_training_go_plots/pose_figures"
        figure_name = 'pose_path_' + date + '_' + time
        checkpoint_file = os.path.join(checkpoint_dir, figure_name + '.png')
        self.fig1.savefig(checkpoint_file, dpi=600)
        print("Saving pose figure ...")


class Plotter2(object):
    def __init__(self):
        plt.ion()
        # Create figures
        self.fig2 = plt.figure(2)
        self.fig3 = plt.figure(3)
        # display pose figure templates.
        self._learning_curve_figure(self.fig2)
        self._time_consume_figure(self.fig3)

    def learning_curve_figure(self, iteration, score, step_num, is_achieved):
        # if is_achieved:
        self.score_graphic.plot(iteration, score, '-r')
        self.step_num_graphic.plot(iteration, step_num, '-r')
        # else:
        #     self.score_graphic.plot(iteration, score, '-b')
        #     self.step_num_graphic.plot(iteration, step_num, '-b')

    def _learning_curve_figure(self, figure):
        figure.suptitle('Learning curves of training process')
        with plt.style.context(['science', 'ieee']):
            self.score_graphic = figure.add_subplot(2, 1, 1)
            self.score_graphic.set_title('Obtained score per iteration')
            self.score_graphic.set_xlabel('iteration')
            self.score_graphic.set_ylabel('score')
            self.score_graphic.grid()
            self.score_graphic.plot(0, 0, '-r')

            self.step_num_graphic = figure.add_subplot(2, 1, 2)
            self.step_num_graphic.set_title('Learning step number per iteration')
            self.step_num_graphic.set_xlabel('iteration')
            self.step_num_graphic.set_ylabel('step number')
            self.step_num_graphic.grid()
            self.step_num_graphic.plot(0, 0, '-r')
            plt.draw()

    def save_learning_curve_figure(self):
        date = str(datetime.date(datetime.now()))
        time = str(datetime.time(datetime.now()))
        checkpoint_dir = "src/training_algorithms/scripts/TD3_training_go_plots/learning_curve_figures"
        figure_name = 'learning_curve_figure_' + date + '_' + time
        checkpoint_file = os.path.join(checkpoint_dir, figure_name + '.png')
        self.fig2.savefig(checkpoint_file, dpi=600)
        print("Saving learning curve figure ...")

    def time_consume_figure(self, iteration, time_consume, is_achieved):
        # if is_achieved:
        self.time_consume_graphic.plot(iteration, time_consume, '-r')
        # else:
            # self.time_consume_graphic.plot(iteration, time_consume, '-b')

    def _time_consume_figure(self, figure):
        plt.ion()
        with plt.style.context(['science', 'ieee']):
            self.time_consume_graphic = figure.gca()
            self.time_consume_graphic.set_title('Total time comsume of training process')
            self.time_consume_graphic.set_xlabel('iteration')
            self.time_consume_graphic.set_ylabel('total time comsumes')
            self.time_consume_graphic.grid()
            self.time_consume_graphic.plot(0, 0, '-r')
            plt.draw()
    
    def save_time_consume_figure(self):
        date = str(datetime.date(datetime.now()))
        time = str(datetime.time(datetime.now()))
        checkpoint_dir = "src/training_algorithms/scripts/TD3_training_go_plots/time_consume_figures"
        figure_name = 'time_consume_figure_' + date + '_' + time
        checkpoint_file = os.path.join(checkpoint_dir, figure_name + '.png')
        self.fig3.savefig(checkpoint_file, dpi=600)
        print("Saving total time comuses figure ...")

# plot1 = Plotter1()
# plt.pause(2)
# plot1.set_target_cartesian_pose([0.5,0,0.4,0,0,0], False)
# plt.pause(1)
# plot1.pose_figure([0.6,0.2,0.4,1,2,1])
# plt.pause(1)
# plot1.pose_figure([0.2,0.3,0.2,2,1,2])

# plt.pause(10)