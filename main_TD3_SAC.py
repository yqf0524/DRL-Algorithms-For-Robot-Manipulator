from MyEnv import KUKAiiwa
from TD3PG.TD3Agent import TD3Agent
from softAC.SACAgent import SACAgent
import numpy as np
import time

if __name__ == '__main__':
    env = KUKAiiwa("PositionControl")
    # print(env.iiwa.current_ee_position)
    # print(env.target_rpy)
    env.target_position = [0.13, 0.1, 0.55]
    td3agent = TD3Agent(alpha=0.001, beta=0.001, input_dims=env.observation_space,
                        tau=0.005, env=env, layer1_size=512, layer2_size=512,
                        layer3_size=512)
    sacagent = SACAgent(env=env)
    n_games = 20000

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

    env.render()
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            env.render()
            action = td3agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            td3agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                td3agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                td3agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)


