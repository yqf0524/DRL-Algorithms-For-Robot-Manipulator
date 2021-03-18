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

    # p1 = Process(target=callback_spin)
    # p1.start()

    # plot1 = Plotter1()
    plot2 = Plotter2()

    # initialize training environment
    env = PositionControl()
    env.iiwa.set_init_configuration(np.array([10, 45, 0, 60, 0, 60, 0]))
    # Set target_position and target_pose for DRL training.
    move_distance_xyz = np.array([0.15, 0.1, -0.1])
    target_ee_position = env.iiwa.start_ee_position + move_distance_xyz
    target_ee_rpy = env.iiwa.start_ee_rpy
    env.iiwa.set_target_ee_pose(target_ee_position, target_ee_rpy)
    print('Start pose:', env.iiwa.start_ee_pose)
    print('Target pose:', env.iiwa.target_ee_pose)


    # move robot_manipulator to initial joint position.
    print("\nInitialize robot configutation ...\n")
    init_configuration = env.iiwa.init_configuration
    joint_position_pub.publish(np.deg2rad(init_configuration))

    # Initialize agents of TD3 algorithms.
    td3_agent_go = TD3Agent(alpha=0.001, beta=0.001, tau=0.005, env=env, 
                         input_dims=env.observation_space, n_actions = 7, layer1_size=128, 
                         layer2_size=256, layer3_size=256, layer4_size=256, layer5_size=256, 
                         layer6_size=128, checkpoint_dir="src/training_algorithms/scripts/TD3_model_go")
    td3_agent_back = TD3Agent(alpha=0.001, beta=0.001, tau=0.005, env=env, 
                         input_dims=env.observation_space, n_actions = 7, layer1_size=128, 
                         layer2_size=256, layer3_size=256, layer4_size=256, layer5_size=256, 
                         layer6_size=128, checkpoint_dir="src/training_algorithms/scripts/TD3_model_back")
    
    max_iteration = 20000
    episode_per_iter = 1e4
    time_consumes_td3_go = []
    time_consumes_td3_back = []
    train_td3_move_foreward_success = False
    train_td3_move_foreward_start_config = env.reset()
    train_td3_move_backward_success = True
    train_td3_move_backward_start_config = []
    score_history_go = []
    score_history_back = []
    load_checkpoint = True
    training_algorithms = True
    
    # if load_checkpoint:
    #     td3_agent_go.load_models()
    #     td3_agent_back.load_models()

    # Print the bad training infos.
    info_arr = ["Warning", "Error", "Fatal error"]
    traing_start_time = time.time()
    time_sleep_move_to_start_config = 2  # wait for 2 secs
    iteration_arr = []
    score_arr = []
    step_num_arr = []
    time_consume_arr = []
    is_achieved_arr = []

    for i in range(max_iteration):
        if rospy.is_shutdown():
            break
        # Training robotic arm go to target pose.
        if train_td3_move_backward_success:
            env.iiwa.init_configuration = train_td3_move_foreward_start_config
            # If training move_foreward is not success, then move back to start.
            print("Robot manipulator move back to move_foreward_start_config ...\n")
            joint_position_pub.publish(np.deg2rad(train_td3_move_foreward_start_config))
            # Waiting until it reachs start configuration.
            time.sleep(time_sleep_move_to_start_config)  

            observation = env.reset()
            done = False
            score_go = 0
            step_counter = 0
            time_consume = 0
            # plot1.set_target_cartesian_pose(env.iiwa.target_ee_pose, env.iiwa.is_achieved)
            print('Start training move foreward for %dst iteration ...\n' % i)
            while not done and step_counter < episode_per_iter and not rospy.is_shutdown():
                # start training algorithms
                action = td3_agent_go.choose_action(observation)
                reward, observation_, done, info = env.step(action)
                if training_algorithms:
                    td3_agent_go.remember(observation, action, reward, observation_, done)
                    td3_agent_go.learn()
                score_go += reward
                # Publish control signal of next joint position to topic
                joint_position_pub.publish(np.deg2rad(observation_))
                # Publish cartesian pose
                cartesian_pose_pub.publish(env.iiwa.current_ee_pose)
                observation = observation_
                
                if info.split(':')[0] in info_arr:
                    print(env.iiwa.current_ee_position)
                    print(info)
                # if step_counter % 5 == 0:
                #     plot1.pose_figure(env.iiwa.current_ee_pose)

                step_counter += 1
                
                if done:
                    train_td3_move_foreward_success = env.iiwa.is_achieved
                    # plot1.pose_figure(env.iiwa.current_ee_pose)
                #     train_td3_move_backward_start_config = env.iiwa.current_configuration
                #     train_td3_move_backward_success = False

            # save agent models
            if train_td3_move_foreward_success and score_go >= max(score_history_go):
                td3_agent_go.save_models()
                train_td3_move_foreward_success = False

            time_end = time.time()
            time_recorder = time_end - traing_start_time - time_sleep_move_to_start_config
            # iteration, score, step_num, time_recorder, is_achieved
            iteration_arr.append(i)
            score_arr.append(score_go)
            step_num_arr.append(step_counter)
            time_consume_arr.append(time_consume)
            is_achieved_arr.append(env.iiwa.is_achieved)
            # plot2.learning_curve_figure(iteration_arr, score_arr, step_num_arr, env.iiwa.is_achieved)
            # plot2.time_consume_figure(iteration_arr, time_consume_arr,  env.iiwa.is_achieved)

            print('Iteration:', i, '| move_foreward:', train_td3_move_foreward_success, 
                  '| score: %.3f' % score_go, '| training_step:', step_counter, '\n')
    plot2.learning_curve_figure(iteration_arr, score_arr, step_num_arr, is_achieved_arr)
    plot2.time_consume_figure(iteration_arr, time_consume_arr, is_achieved_arr)
    plt.show()
    plot2.save_learning_curve_figure()
    plot2.save_time_consume_figure()
    print("*** Programm ended!!! ***\n")
        
