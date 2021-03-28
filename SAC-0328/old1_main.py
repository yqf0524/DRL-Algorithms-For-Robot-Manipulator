#!/usr/bin/env python3

# import training env and DRL algorithms
from iiwa_DRL_environment import PositionControl
from TD3_Algorithm import Agent as TD3Agent
from SAC_Algorithm import Agent as SACAgent
# import msgs, publishers and subscribers
from training_algorithms.msg import RobotConfiguration
from training_algorithms.msg import CartesianPose
from iiwa_ros_publisher_in_DRL import PublishJointPosition
from iiwa_ros_publisher_in_DRL import PublishCartesianPose 
from iiwa_ros_subscriber_in_DRL import ReturnJointState
from iiwa_ros_subscriber_in_DRL import ReturnCartesianPose
# import plotter for training process
from iiwa_ros_plotter_in_DRL import Plotter1
from iiwa_ros_plotter_in_DRL import Plotter2
# import necessary python packages
from multiprocessing import Process, Pipe, Queue
import matplotlib.pyplot as plt
import numpy as np
import rospy
import time


def callback_spin():
    print("Process 1 started ...")
    rospy.spin()
    print("Process 1 terminated ...")


if __name__ == '__main__':
    # Create init_node for DRL.
    rospy.init_node('ROS_node_in_DRL', anonymous=True)
    # Create publishers for training process.
    joint_position_pub = PublishJointPosition()
    cartesian_pose_pub = PublishCartesianPose()
    # Create subscribers for training process.
    joint_state_sub = ReturnJointState()
    cartesian_pose_sub = ReturnCartesianPose()
    # plot training processes
    plot1 = Plotter1(1, 2, 'TD3_training_forward_plots')
    plot2 = Plotter2(3, 'TD3_training_forward_plots')
    # plot3 = Plotter1(4, 5, 'TD3_training_backward_plots')
    # plot4 = Plotter2(6, 'TD3_training_backward_plots')
    plt.pause(1)
    # initialize the training environment
    env = PositionControl()
    init_configuration = np.array([20, 60, 0, 60, 0, 60, 0])
    env.iiwa.set_init_configuration(init_configuration)
    # Set target_position and target_pose for DRL training.
    move_distance_xyz = np.array([0.15, -0.1, -0.1])
    target_ee_xyz = env.iiwa.start_ee_xyz + move_distance_xyz
    target_ee_rpy = env.iiwa.start_ee_rpy
    env.iiwa.set_target_ee_pose(target_ee_xyz, target_ee_rpy)
    print('Start pose:', env.iiwa.start_ee_pose)
    print('Target pose:', env.iiwa.target_ee_pose)
    # move robot_manipulator to initial joint position.
    print("\nInitialize robot configutation ...\n")
    joint_position_pub.publish(np.deg2rad(env.iiwa.init_configuration))

    plot1.target_cartesian_pose(env.iiwa.target_ee_pose)
    # plot3.target_cartesian_pose(env.iiwa.start_ee_pose)
    # plt.pause(0.1)

    # Initialize agents of TD3 algorithms.
    td3_agent_forward = TD3Agent(alpha=0.001, beta=0.001, tau=0.005, env=env, 
                         input_dims=env.observation_space, n_actions = 7, layer1_size=256, 
                         layer2_size=256, layer3_size=256, layer4_size=256, layer5_size=256, 
                         layer6_size=256, checkpoint_dir="training_algorithms/scripts/TD3_model_forward")
    td3_agent_backward = TD3Agent(alpha=0.001, beta=0.001, tau=0.005, env=env, 
                         input_dims=env.observation_space, n_actions = 7, layer1_size=256, 
                         layer2_size=256, layer3_size=256, layer4_size=256, layer5_size=256, 
                         layer6_size=256, checkpoint_dir="training_algorithms/scripts/TD3_model_backward")

    # Initialize agents of SAC algorithms.
    sac_agent_forward = SACAgent(alpha=0.0003, beta=0.0003, tau=0.005, env=env, input_dims=env.observation_space,
                        n_actions=7, layer1_size=512, layer2_size=256, layer3_size=256, layer4_size=256, 
                        layer5_size=256, layer6_size=256, checkpoint_dir="training_algorithms/scripts/TD3_model_forward")
    sac_agent_backward = SACAgent(alpha=0.0003, beta=0.0003, tau=0.005, env=env, input_dims=env.observation_space,
                         n_actions=7, layer1_size=512, layer2_size=256, layer3_size=256, layer4_size=256, 
                         layer5_size=256, layer6_size=256, checkpoint_dir="training_algorithms/scripts/TD3_model_forward")

    epochs = 2000
    steps_per_epoch = 4e3
    train_td3_move_forward_start_config = np.multiply(env.reset()[0:7], env.observation_space_high)
    train_td3_move_backward_start_config = []
    train_td3_move_forward = True
    train_td3_move_backward = False
    time_consumes_history_forward = []
    time_consumes_history_backward = []
    score_history_forward = []
    score_history_backward = []
    training_steps_history_forward = []
    training_steps_history_backward = []
    iter_history_forward = []
    iter_history_backward = []
    iter_forward = 0
    iter_backward = 0
    load_checkpoint = False
    training_algorithms = True
    
    if load_checkpoint:
        td3_agent_go.load_models()
        td3_agent_back.load_models()

    time_sleep_move_to_start_config = 2  # wait for 2 secs
    traing_start_time = time.time()
    for i in range(epochs):
        if rospy.is_shutdown():
            break
        observation = None
        agent = None
        done = False
        score = 0
        step_counter = 0
        time_consume = 0
        path_xyz = [[],[],[]]
        path_rpy = [[],[],[]]
        # Training robotic arm go to target pose.
        if train_td3_move_forward:
            env.iiwa.init_configuration = train_td3_move_forward_start_config
            observation = env.reset()
            # agent = td3_agent_forward
            agent = sac_agent_forward
        else:  # train_td3_move_backward = True
            env.iiwa.init_configuration = train_td3_move_backward_start_config
            observation = env.reset()
            # agent = td3_agent_backward
            agent = sac_agent_backward
        print("Robot manipulator move to init configuration ...\n")
        # If training move_forward is not success, then move back to start.
        joint_position_pub.publish(np.deg2rad(env.iiwa.init_configuration))
        # Waiting until it reachs start configuration.
        # time.sleep(time_sleep_move_to_start_config)
        print('Start training ...\n')
        start_time = time.time()
        while not done and step_counter < steps_per_epoch and not rospy.is_shutdown():
            
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
            # Publish cartesian pose
            cartesian_pose_pub.publish(env.iiwa.current_ee_pose)
            path_xyz[0].append(env.iiwa.current_ee_pose[0])
            path_xyz[1].append(env.iiwa.current_ee_pose[1])
            path_xyz[2].append(env.iiwa.current_ee_pose[2])
            path_rpy[0].append(env.iiwa.current_ee_pose[3])
            path_rpy[1].append(env.iiwa.current_ee_pose[4])
            path_rpy[2].append(env.iiwa.current_ee_pose[5])
            # next learning step
            step_counter += 1

            if done:
                time_consume = time.time() - start_time
                print(info)
                print('xyz:', env.iiwa.current_ee_xyz, 'xyz_norm:', env.iiwa.xyz_error_norm)
                if train_td3_move_forward:
                    # iteration, score, step_num, time_recorder, is_achieved
                    score_history_forward.append(score)
                    training_steps_history_forward.append(step_counter)
                    time_consumes_history_forward.append(time_consume)
                    iter_history_forward.append(iter_forward)
                    forward_success = env.iiwa.is_achieved
                    plot1.xyz_figure(path_xyz)
                    plot1.rpy_figure(path_rpy)
                    if not rospy.is_shutdown():
                        plot2.learning_curve_figure(iter_history_forward, score_history_forward, 
                                                    training_steps_history_forward)
                        # plot2.time_consume_figure(iter_history_forward, time_consumes_history_forward)
                        plt.pause(0.05)
                    if forward_success:
                        plot1.save_pose_figure()
                        td3_agent_forward = False
                        td3_agent_backward = True
                        train_td3_move_backward_start_config = env.iiwa.current_configuration
                        if score >= max(score_history_forward):
                            agent.save_models()  # save agent models
                    print('Epoch:', iter_forward, '| move_forward:', forward_success, 
                          '| score_forward: %.3f' % score, '| training_step:', step_counter, '\n')
                    iter_forward += 1
                else:  # Training train_td3_move_backward = True
                    # iteration, score, step_num, time_recorder, is_achieved
                    score_history_backward.append(score)
                    training_steps_history_backward.append(step_counter)
                    time_consumes_history_backward.append(time_consume)
                    iter_history_backward.append(iter_forward)
                    backward_success = env.iiwa.is_achieved
                    # plot3.xyz_figure(path_xyz)
                    # plot3.rpy_figure(path_rpy)
                    if not rospy.is_shutdown():
                        plot4.learning_curve_figure(iter_history_backward, score_history_backward, 
                                                    training_steps_history_backward)
                        # plot4.time_consume_figure(iter_history_backward, time_consumes_history_backward)
                        plt.pause(0.05)
                    if backward_success:
                        # plot3.save_pose_figure()
                        td3_agent_forward = True
                        td3_agent_backward = False
                        train_td3_move_forward_start_config = env.iiwa.current_configuration
                        if score >= max(score_history_backward):
                            agent.save_models()  # save agent models
                    print('Epoch:', iter_backward, '| move_backward:', backward_success, 
                          '| score_backward: %.3f' % score, '| training_step:', step_counter, '\n')
                    iter_backward += 1

    plot2.save_learning_curve_figure()
    # plot2.save_time_consume_figure()
    plot4.save_learning_curve_figure()
    # plot4.save_time_consume_figure()
    traing_end_time = time.time()
    total_time_consume = traing_end_time - traing_start_time
    hours = total_time_consume // (60 * 60)
    minutes = (total_time_consume - hours * 3600) // 60
    seconds = total_time_consume - hours * 3600 - minutes * 60
    print("Programm ended! Training_time_consumes: %dh:%dm:%.4fs"% (hours, minutes, seconds), '\n')
        


