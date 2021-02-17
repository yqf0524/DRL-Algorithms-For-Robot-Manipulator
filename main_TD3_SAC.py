from MyEnv import PositionControl
from TD3PG.TD3Agent import TD3Agent
from softAC.SACAgent import SACAgent
import numpy as np
import time
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # env = KUKAiiwa("PositionControl")
    env = PositionControl()
    print(env.iiwa.current_ee_position)
    print(env.target_rpy)

    env.target_position = np.array([0.13, 0.1, -0.55])
    td3agent = TD3Agent(alpha=0.001, beta=0.001, input_dims=env.observation_space,
                        tau=0.005, env=env, layer1_size=512, layer2_size=512,
                        layer3_size=512)
    sacagent = SACAgent(env=env)
    n_games = 20000
    try_num = 1e5

    filename1 = "time consuming.png"
    filename2 = "average reward.png"
    figure_file1 = "TD3PG/plots/" + filename1
    figure_file2 = "TD3PG/plots/" + filename2

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        td3agent.load_models()
        sacagent.load_models()
        env.render()

    time_consumes = []
    # env.render()
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        try_counter = 0
        plt.ion()
        t = []
        y = []
        time_consume = 0
        t0 = time.time()
        while not done and try_counter < try_num:
            # env.render()
            action = td3agent.choose_action(observation)
            # print(action)
            observation_, reward, done, info = env.step(action)
            score += reward
            td3agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                td3agent.learn()
            observation = observation_

            plt.clf()
            t_now = 0.01 * try_counter
            t.append(t_now)
            y.append(reward)
            plt.plot(t, y)
            plt.pause(0.001)
            plt.ioff()
            try_counter += 1

        t1 = time.time()
        time_consume = t1 - t0
        time_consumes.append(time_consume)
        print(time_consume)
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        save_model = np.max(time_consumes) == time_consume
        if not load_checkpoint and save_model:
            td3agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)
