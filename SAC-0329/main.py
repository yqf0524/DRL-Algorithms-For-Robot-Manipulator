#!/usr/bin/env python3

# import training env and DRL algorithms
from iiwa_DRL_environment import PositionControl
from TD3_Algorithm import Agent as TD3Agent
from SAC_Algorithm import Agent as SACAgent
# import msgs, publishers and subscribers
from iiwa_ros_publisher_in_DRL import PublishJointPosition
from iiwa_ros_subscriber_in_DRL import ReturnJointState
# import plotter for training process
from iiwa_ros_plotter_in_DRL import Plotter1
from iiwa_ros_plotter_in_DRL import Plotter2
# import data processing class
from iiwa_ros_data_process_in_DRL import DataProcessing
# import necessary python packages
from multiprocessing import Process, Pipe, Queue
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rospy
import time


def callback_spin():
    print("Process 1 started ...")
    rospy.spin()
    print("Process 1 terminated ...")


if __name__ == '__main__':
    # Create init_node for DRL.
    rospy.init_node('ROS_node_in_DRL', anonymous=True)
    # Create publisher and subscriber for training process.
    joint_position_pub = PublishJointPosition()
    joint_state_sub = ReturnJointState()
    # plot path figure. 'SAC_training_forward_plots', 'SAC_training_backward_plots'
    plot1 = Plotter1()
    # plot learning curve.
    plot2 = Plotter2()
    plt.pause(1)
    # create obj of DataProcessing
    dp_forward = DataProcessing('forward')
    dp_backward = DataProcessing('backward')
    # initialize the training environment
    env = PositionControl()
    init_configuration = np.array([20, 60, 0, 60, 0, 60, 0])
    env.iiwa.set_init_configuration(init_configuration)
    dp_forward.add_config_df(init_configuration)
    # Set target_pose for DRL training.
    move_distance_xyz = np.array([0.15, -0.1, -0.1], dtype=np.float32)
    target_ee_xyz_forward = (env.iiwa.current_ee_xyz + move_distance_xyz).astype(np.float32)
    target_ee_xyz_backward = (env.iiwa.current_ee_xyz).astype(np.float32)
    target_ee_rpy = (env.iiwa.current_ee_rpy).astype(np.float32)

    print('Target_xyz_forward:', target_ee_xyz_forward)
    print('Target_xyz_backward:', target_ee_xyz_backward)
    print('Target_rpy:', target_ee_rpy)
    
    # move robot_manipulator to initial joint position.
    print("\nInitialize robot configutation ...\n")
    joint_position_pub.publish(np.deg2rad(env.iiwa.init_configuration))

    # # Initialize agents of TD3 algorithms.
    # td3_agent_forward = TD3Agent(alpha=0.001, beta=0.001, tau=0.005, env=env, 
    #                      input_dims=env.observation_space, n_actions = 7, layer1_size=256, 
    #                      layer2_size=256, layer3_size=256, layer4_size=256, layer5_size=256, 
    #                      layer6_size=256, checkpoint_dir="training_algorithms/scripts/TD3_model_forward")
    # td3_agent_backward = TD3Agent(alpha=0.001, beta=0.001, tau=0.005, env=env, 
    #                      input_dims=env.observation_space, n_actions = 7, layer1_size=256, 
    #                      layer2_size=256, layer3_size=256, layer4_size=256, layer5_size=256, 
    #                      layer6_size=256, checkpoint_dir="training_algorithms/scripts/TD3_model_backward")

    # Initialize agents of SAC algorithms.
    sac_agent_forward = SACAgent(alpha=0.0003, beta=0.0003, tau=0.005, env=env, input_dims=env.observation_space,
                        n_actions=7, layer1_size=256, layer2_size=256, layer3_size=256, layer4_size=256, 
                        layer5_size=256, layer6_size=256, checkpoint_dir="src/training_algorithms/scripts/SAC_model_forward")
    sac_agent_backward = SACAgent(alpha=0.0003, beta=0.0003, tau=0.005, env=env, input_dims=env.observation_space,
                         n_actions=7, layer1_size=256, layer2_size=256, layer3_size=256, layer4_size=256, 
                         layer5_size=256, layer6_size=256, checkpoint_dir="src/training_algorithms/scripts/SAC_model_backward")

    epochs = 8000
    steps_per_epoch = 3e3
    load_checkpoint = False
    training_algorithms = True

    train_move_forward_start_config = env.iiwa.current_configuration
    train_move_forward = True
    iter_history_forward = [-1]
    avg_score_history_forward = [0]
    training_steps_history_forward = [0]
    xyz_error_history_forward = [env.iiwa.xyz_error_norm]
    rpy_error_history_forward = [env.iiwa.rpy_error_norm]
    max_avg_score_success_forward = -100
    iter_forward = 0

    train_move_backward_start_config = env.iiwa.current_configuration
    train_move_backward = False
    iter_history_backward = [-1]
    avg_score_history_backward = [0]
    training_steps_history_backward = [0]
    xyz_error_history_backward = [env.iiwa.xyz_error_norm]
    rpy_error_history_backward = [env.iiwa.rpy_error_norm]
    max_avg_score_success_backward = -100
    iter_backward = 0
    
    if load_checkpoint:
        sac_agent_forward.load_models()
        sac_agent_backward.load_models()

    time_sleep_move_to_start_config = 2  # wait for 2 secs
    traing_start_time = time.time()
    for i in range(epochs):
        if rospy.is_shutdown():
            break
        done = False
        score = 0
        step_counter = 0
        time_consume = 0
        path_xyz = [[],[],[]]
        path_rpy = [[],[],[]]

        # Training robotic arm go to target pose.
        if train_move_forward:
            dp_forward.path_joint_space_df = pd.DataFrame(columns=dp_forward.joint_index)
            env.iiwa.set_init_configuration(train_move_forward_start_config)
            observation = env.reset()
            # agent = td3_agent_forward
            agent = sac_agent_forward
            dp = dp_forward
            # Set target_pose for DRL training.
            env.iiwa.set_target_ee_pose(target_ee_xyz_forward, target_ee_rpy)
            # plot start and target pose on figure
            plot1.start_target_cartesian_pose(env.iiwa.current_ee_pose, env.iiwa.target_ee_pose)
        else:  # train_move_backward = True
            dp_backward.path_joint_space_df = pd.DataFrame(columns=dp_forward.joint_index)
            env.iiwa.set_init_configuration(train_move_backward_start_config)
            observation = env.reset()
            # agent = td3_agent_backward
            agent = sac_agent_backward
            dp = dp_backward
            # Set target_pose for DRL training.
            env.iiwa.set_target_ee_pose(target_ee_xyz_backward, target_ee_rpy)
            # plot start and target pose on figure
            plot1.start_target_cartesian_pose(env.iiwa.current_ee_pose, env.iiwa.target_ee_pose)
        # If training move_forward is not success, then move back to start config.
        joint_position_pub.publish(np.deg2rad(env.iiwa.init_configuration))
        dp.add_path_joint_space(env.iiwa.current_configuration, step_counter)

        print("Robot manipulator move to init configuration ...")
        # Waiting until it reachs start configuration.
        # time.sleep(time_sleep_move_to_start_config)
        print('Start training ...')
        while not done and not rospy.is_shutdown():
            # start training algorithms
            action = agent.choose_action(observation)
            reward, observation_, done, info = env.step(action)
            if training_algorithms:
                agent.remember(observation, action, reward, observation_, done)
                agent.learn()
            score += reward
            observation = observation_
            # Publish control signal of next joint position to topic
            joint_position_pub.publish(np.deg2rad(env.iiwa.current_configuration))
            dp.add_path_joint_space(env.iiwa.current_configuration, step_counter)
            path_xyz[0].append(env.iiwa.current_ee_pose[0])
            path_xyz[1].append(env.iiwa.current_ee_pose[1])
            path_xyz[2].append(env.iiwa.current_ee_pose[2])
            path_rpy[0].append(env.iiwa.current_ee_pose[3])
            path_rpy[1].append(env.iiwa.current_ee_pose[4])
            path_rpy[2].append(env.iiwa.current_ee_pose[5])
            # next learning step
            step_counter += 1

            if done or step_counter == steps_per_epoch:
                done = True
                print(info)
                print('xyz:', env.coord_xyz, 'xyz_norm: %.8f' % env.min_xyz_error, 'rpy_norm: %.8f' % env.iiwa.rpy_error_norm)
                plot1.xyz_figure(path_xyz)
                plot1.rpy_figure(path_rpy)
                plot1.xyz_3d_figure(path_xyz)
                if train_move_forward:
                    # iteration, score, step_num, time_recorder, is_achieved
                    iter_history_forward.append(iter_forward)
                    avg_score = score / step_counter
                    avg_score_history_forward.append(avg_score)
                    training_steps_history_forward.append(step_counter)
                    xyz_error_history_forward.append(env.min_xyz_error)
                    rpy_error_history_forward.append(env.iiwa.rpy_error_norm)
                    forward_success = env.iiwa.is_achieved
                    if not rospy.is_shutdown():
                        plot2.learning_curve_figure(iter_history_forward, avg_score_history_forward, 
                                                    training_steps_history_forward)
                        plot2.error_figure(iter_history_forward, xyz_error_history_forward, rpy_error_history_forward)
                        plt.pause(0.01)
                    if forward_success:
                        plot1.save_pose_figure('SAC_training_forward_plots')
                        dp_forward.save_path_cartesian_space(path_xyz, path_rpy)
                        dp_forward.save_path_joint_space()
                        dp_backward.add_config_df(env.iiwa.current_configuration)
                        train_move_forward = False
                        train_move_backward = True
                        train_move_backward_start_config = env.iiwa.current_configuration
                        if avg_score >= max_avg_score_success_forward: agent.save_models()  # save agent models
                            
                    print('Epoch:', iter_forward, '| move_forward:', forward_success, 
                          '| score_forward: %.3f' % score, '| training_step:', step_counter, '\n')
                    iter_forward += 1
                    i = iter_forward
                    if iter_forward % 500 == 0:
                        plot2.save_learning_curve_figure('SAC_training_forward_plots')
                        plot2.save_error_figure('SAC_training_forward_plots')
                        iter_history_forward = []
                        avg_score_history_forward = []
                        training_steps_history_forward = []
                        xyz_error_history_forward = []
                        rpy_error_history_forward = []
                else:  # Training train_move_backward = True
                    # iteration, score, step_num, time_recorder, is_achieved
                    iter_history_backward.append(iter_backward)
                    avg_score = score / step_counter
                    avg_score_history_backward.append(avg_score)
                    training_steps_history_backward.append(step_counter)
                    xyz_error_history_backward.append(env.min_xyz_error)
                    rpy_error_history_backward.append(env.iiwa.rpy_error_norm)
                    backward_success = env.iiwa.is_achieved
                    if not rospy.is_shutdown():
                        plot2.learning_curve_figure(iter_history_backward, avg_score_history_backward, 
                                                    training_steps_history_backward)
                        plot2.error_figure(iter_history_backward, xyz_error_history_backward, rpy_error_history_backward)
                        plt.pause(0.01)
                    if backward_success:
                        plot1.save_pose_figure('SAC_training_backward_plots')
                        dp_backward.save_path_cartesian_space(path_xyz)
                        dp_backward.save_path_joint_space()
                        env.iiwa.set_init_configuration(env.iiwa.current_configuration)
                        dp_forward.add_config_df(env.iiwa.current_configuration)
                        train_move_forward = True
                        train_move_backward = False
                        train_move_forward_start_config = env.iiwa.current_configuration
                        if avg_score >= max_avg_score_success_backward:
                            agent.save_models()  # save agent models
                    print('Epoch:', iter_backward, '| move_backward:', backward_success, 
                          '| score_backward: %.3f' % score, '| training_step:', step_counter, '\n')
                    iter_backward += 1
                    i = iter_backward
                    if iter_backward % 500 == 0:
                        plot2.save_learning_curve_figure('SAC_training_backward_plots')
                        plot2.save_error_figure('SAC_training_backward_plots')
                        iter_history_backward = []
                        avg_score_history_backward = []
                        training_steps_history_backward = []
                        xyz_error_history_backward = []
                        rpy_error_history_backward = []
    # Saving learning curve forward
    plot2.learning_curve_figure(iter_history_forward, avg_score_history_forward, 
                                training_steps_history_forward)
    plot2.error_figure(iter_history_forward, xyz_error_history_forward)
    plt.pause(0.01)
    plot2.save_learning_curve_figure('SAC_training_forward_plots')
    plot2.save_error_figure('SAC_training_forward_plots')
    # Saving learning curve backward
    plot2.learning_curve_figure(iter_history_backward, avg_score_history_backward, 
                            training_steps_history_backward)
    plot2.error_figure(iter_history_backward, xyz_error_history_backward)
    plt.pause(0.01)
    plot2.save_learning_curve_figure('SAC_training_backward_plots')
    plot2.save_error_figure('SAC_training_backward_plots')
    # Saving config data
    dp_forward.save_config()
    dp_backward.save_config()
    # Compute the total training time
    traing_end_time = time.time()
    total_time_consume = traing_end_time - traing_start_time
    hours = total_time_consume // (60 * 60)
    minutes = (total_time_consume - hours * 3600) // 60
    seconds = total_time_consume - hours * 3600 - minutes * 60
    print("Programm ended! Training_time_consumes: %dh:%dm:%.4fs"% (hours, minutes, seconds), '\n')
        


