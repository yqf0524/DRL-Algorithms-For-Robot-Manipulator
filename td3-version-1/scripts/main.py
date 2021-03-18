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


def plot_pose_figure(conn1, conn2):
    print("Process 2 started ...")
    plot = Plotter1()
    # conn1 receives the target catesian pose training_success signal.
    target_cartesian_pose = conn1.recv()
    training_success = conn1.recv()
    print(target_cartesian_pose, training_success)
    plot.set_target_cartesian_pose(target_cartesian_pose, training_success)

    while not rospy.is_shutdown():
        # conn2 receives training infos and current cartesian pose.
        training_msg = conn2.recv()
        if training_msg == 'continue_training':
            current_cartesian_pose = conn2.recv()
            plot.pose_figure(current_cartesian_pose)
        elif training_msg == 'next_iteration':
            training_success = conn1.recv()
            plot.set_target_cartesian_pose(target_cartesian_pose, training_success)
        elif training_msg == 'finish_training':
            # Exit while loop.
            print("Training process finished.")
            break
        else:
            print('Something wrong!!!')
    plt.ioff()
    print("Process 2 terminated ...")


def plot_learning_curves(conn):
    print("Process 3 started ...")
    plot = Plotter2()
    iteration = []
    score = []
    step_num = []
    time_consume = []
    while conn.recv() == 'continue_training':
        # plot_infos = [iteration, score, step_num, time_consume, is_achieved]
        print("In plot learning curve loop.")
        plot_infos = conn.recv()
        iteration.append(plot_infos[0])
        score.append(plot_infos[1])
        step_num.append(plot_infos[2])
        time_consume.append(plot_infos[3])
        is_achieved = plot_infos[4]
        plot.learning_curve_figure(iteration, score, step_num, is_achieved)
        plot.time_consume_figure(iteration, time_consume, is_achieved)

    if conn.recv() == 'save_figure':
        plot.save_learning_curve_figure()
        plot.save_time_consume_figure()
        conn.send(True)
    plt.ioff()
    print("Process 3 terminated ...")


# class MainRunner(object):
#     def __init__(self, distance_in_axises):
#         # Create publishers for training process.
#         self.joint_position_pub = PublishJointPosition()
#         self.cartesian_pose_pub = PublishCartesianPose()
#         # Create subscribers for training process.
#         self.joint_state_sub = ReturnJointState()
#         self.cartesian_pose_sub = ReturnCartesianPose()

#         # initialize training environment
#         self.env = PositionControl()
#         # Set target_position and target_pose for DRL training.
#         self.target_ee_position = self.env.iiwa.start_ee_position + distance_in_axises
#         self.target_ee_rpy = self.env.iiwa.start_ee_rpy
#         self.env.iiwa.set_target_ee_pose(self.target_ee_position, self.target_ee_rpy)

if __name__ == '__main__':
    # Create init_node for DRL.
    rospy.init_node('ROS_node_in_DRL', anonymous=True)
    # Create publishers for training process.
    joint_position_pub = PublishJointPosition()
    cartesian_pose_pub = PublishCartesianPose()
    # Create subscribers for training process.
    joint_state_sub = ReturnJointState()
    cartesian_pose_sub = ReturnCartesianPose()

    # Use multiprocesses
    conn1, conn2 = Pipe()
    conn3, conn4 = Pipe()
    p1 = Process(target=plot_pose_figure, args=(conn1, conn2,))
    p2 = Process(target=callback_spin)
    p3 = Process(target=plot_learning_curves, args=(conn4,))
    p1.start()
    p2.start()
    p3.start()

    # initialize training environment
    env = PositionControl()
    # env.iiwa.set_init_configuration(init_configuration)
    # Set target_position and target_pose for DRL training.
    move_distance_xyz = np.array([0.15, 0.1, -0.1])
    target_ee_position = env.iiwa.start_ee_position + move_distance_xyz
    target_ee_rpy = env.iiwa.start_ee_rpy
    env.iiwa.set_target_ee_pose(target_ee_position, target_ee_rpy)
    print('Start pose:', env.iiwa.start_ee_pose)
    print('Target pose:', env.iiwa.target_ee_pose)

    # conn2 send the target_ee_pose to Process 2 for plotting.
    conn2.send(env.iiwa.target_ee_pose)
    # conn2 send the training_success signal to Process 2.
    conn2.send(False)
    # conn1 send the training signal.
    conn1.send('continue_training')
    # conn1 send the current pose to Process 2 for plotting.
    conn1.send(env.iiwa.current_ee_pose)

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

    # sac_agent = SACAgent(alpha=0.0003, beta=0.0003, tau=0.005, env=env,
    #                      input_dims=env.observation_space, layer1_size=512, 
    #                      layer2_size=512, layer3_size=512)
    
    # max_iteration = 20000
    # episode_per_iter = 1e4
    # time_consumes_td3_go = []
    # time_consumes_td3_back = []
    # train_td3_move_foreward_success = False
    # train_td3_move_foreward_start_config = env.reset()
    # train_td3_move_backward_success = True
    # train_td3_move_backward_start_config = []
    # score_history_go = []
    # score_history_back = []
    # load_checkpoint = True
    # training_algorithms = True
    
    # # if load_checkpoint:
    # #     td3_agent_go.load_models()
    # #     td3_agent_back.load_models()

    # # Print the bad training infos.
    # info_arr = ["Warning", "Error", "Fatal error"]
    # traing_start_time = time.time()
    # time_sleep_move_to_start_config = 2  # wait for 2 secs
    # iter = 0
    # # for i in range(max_iteration)
    # while iter < max_iteration and not rospy.is_shutdown():
    #     # Training robotic arm go to target pose.
    #     if train_td3_move_backward_success:
    #         env.iiwa.init_configuration = train_td3_move_foreward_start_config
    #         # If training move_foreward is not success, then move back to start.
    #         print("Robot manipulator move back to move_foreward_start_config ...\n")
    #         joint_position_pub.publish(np.deg2rad(train_td3_move_foreward_start_config))
    #         # Waiting until it reachs start configuration.
    #         time.sleep(time_sleep_move_to_start_config)  

    #         observation = env.reset()
    #         done = False
    #         score_go = 0
    #         step_counter = 0
    #         time_consume = 0
    #         td3_agent_go.time_step_counter = 0

    #         print('Start training move foreward for %dst iteration ...\n' % iter)
    #         while not done and step_counter < episode_per_iter and not rospy.is_shutdown():
    #             # start training algorithms
    #             action = td3_agent_go.choose_action(observation)
    #             reward, observation_, done, info = env.step(action)
    #             if training_algorithms:
    #                 td3_agent_go.remember(observation, action, reward, observation_, done)
    #                 td3_agent_go.learn()
    #             score_go += reward
    #             # Publish control signal of next joint position to topic
    #             joint_position_pub.publish(np.deg2rad(observation_))
    #             # Publish cartesian pose
    #             cartesian_pose_pub.publish(env.iiwa.current_ee_pose)
    #             observation = observation_
    #             # Plot current_ee_pose for every 3 steps.
    #             if step_counter % 3 == 0:
    #                 # First: conn1 send the training signal.
    #                 conn1.send('continue_training')
    #                 # Second: conn1 send the current pose to Process 2 for plotting.
    #                 conn1.send(env.iiwa.current_ee_pose)
                
    #             if info.split(':')[0] in info_arr:
    #                 print(env.iiwa.current_ee_position)
    #                 print(info)
                
    #             step_counter += 1

    #             if done:
    #                 train_td3_move_foreward_success = env.iiwa.is_achieved
    #             #     train_td3_move_backward_start_config = env.iiwa.current_configuration
    #             #     train_td3_move_backward_success = False
    #                 # conn1 send the training signal.
    #                 conn1.send('next_iteration')
    #                 # conn2 send the signal of is_achieved goal position.
    #                 conn2.send(env.iiwa.is_achieved)

    #         # save agent models
    #         if train_td3_move_foreward_success and score_go >= max(score_history_go):
    #             td3_agent_go.save_models()
    #             train_td3_move_foreward_success = False

    #         time_end = time.time()
    #         time_recorder = time_end - traing_start_time - time_sleep_move_to_start_config
    #         # iteration, score, step_num, time_recorder, is_achieved
    #         plot_infos = [iter, score_go, step_counter, time_recorder, env.iiwa.is_achieved]
    #         # queue1 send a finish signal of iteration, then start next iteration
    #         conn3.send('continue_training')
    #         conn3.send(plot_infos)
            
    #         # score_history_go.append(score_go)
    #         # time_consumes_td3_go.append(time_consume)
    #         print('Iteration:', iter, '| move_foreward:', train_td3_move_foreward_success, 
    #               '| score: %.3f' % score_go, '| training_step:', td3_agent_go.time_step_counter, '\n')
    #     # Start next iteration
    #     iter += 1

    # # Send signal to process 2 and process 3.
    # conn2.send('finish_training')
    # p1.terminate()
    # p2.terminate()
    # # break the for loop and end program.
    # conn3.send('save_figure')
    # if conn3.recv():
    #     p3.terminate()

    # print("*** Programm ended!!! ***\n")
        

















        # # Training robotic arm move back to target pose.
        # if train_td3_move_foreward_success:
        #     env.iiwa.init_configuration = train_td3_move_backward_start_config
        #     if not train_td3_move_backward_success:
        #         # If training move_back is not success, then manipulator move back 
        #         # to move_backward_start_config.
        #         joint_position_pub.publish(env.iiwa.init_configuration)
        #         # print("Robot manipulator move to initial move_backward_start_config ...")
        #         time.sleep(3)  # Waiting until it reachs goal.

        #     observation = env.reset()
        #     time.sleep(3)
        #     done = False
        #     score_back = 0
        #     step_counter = 0
        #     time_consume = 0
        #     time_start = time.time()

        #     while not done and step_counter < learn_steps and rospy.is_shutdown():
        #         action = td3_agent_back.choose_action(observation)
        #         observation_, reward, done, info = env.step(action)
        #         score_back += reward
        #         if training_algorithms:
        #             td3_agent_back.remember(observation, action, reward, observation_, done)
        #             td3_agent_back.learn()
        #         observation = observation_
        #         step_counter += 1

        #         if env.iiwa_is_achieved_goal:
        #             train_td3_move_backward_success = True
        #             train_td3_move_foreward_start_config = env.iiwa.current_configuration
        #             train_td3_move_foreward_success = False
        
        #     time_end = time.time()
        #     time_consume = time_end - time_start
        #     score_history_back.append(score_back)
        #     time_consumes_td3_back.append(time_consume)

        #     if train_td3_move_backward_success and score_back >= max(score_history_back):
        #         td3_agent_back.save_models()

        #     print('Iteration:', i, '| move_backward:', train_td3_move_backward_success, 
        #           '| score: %.5f' % score_back, '| training_step:', step_counter, '\n')