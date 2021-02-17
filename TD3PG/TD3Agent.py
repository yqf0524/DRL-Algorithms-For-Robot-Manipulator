from TD3PG.buffer import ReplayBuffer
from TD3PG.networks import ActorNetwork
from TD3PG.networks import CriticNetwork
from TD3PG.OU_process import OrnsteinUhlenbeckNoise
import numpy as np
import torch as T
import torch.nn.functional as F


class TD3Agent:
    def __init__(self, alpha, beta, input_dims, tau, env, layer1_size,
                 layer2_size, layer3_size, gamma=0.99,
                 update_actor_interval=3, warmup=1000, n_actions=7,
                 max_size=1000000, batch_size=100, noise=0.01):
        self.gamma = gamma
        self.tau = tau
        self.max_action = T.tensor(env.action_space_high, dtype=T.float32)
        self.min_action = T.tensor(env.action_space_low, dtype=T.float32)
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, layer3_size,
                                  max_action=env.action_space_high, n_actions=n_actions, name='actor')

        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size, layer2_size,
                                      layer3_size, n_actions=n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size, layer2_size,
                                      layer3_size, n_actions=n_actions, name='critic_2')

        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, layer3_size,
                                         max_action=env.action_space_high, n_actions=n_actions, name='target_actor')

        self.target_critic_1 = CriticNetwork(beta, input_dims, layer1_size, layer2_size,
                                             layer3_size, n_actions=n_actions, name='target_critic_1')
        self.target_critic_2 = CriticNetwork(beta, input_dims, layer1_size, layer2_size,
                                             layer3_size, n_actions=n_actions, name='target_critic_2')

        # self.noise = np.deg2rad(noise)  # noise = 0.01 degree
        self.noise = OrnsteinUhlenbeckNoise(self.n_actions)  # in range around [-1, 1]
        self.update_network_parameters(tau=0.99)

    def choose_action(self, observation):
        # if self.time_step < self.warmup:
        #     mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,))
        #                   ).to(self.actor.device)
        # else:
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)  # [-1, 1]
        noise = self.noise() * 0.0005  # 0.029 degree
        mu_prime = mu + T.tensor(noise, dtype=T.float).to(self.actor.device)

        # mu_prime = T.clamp(mu_prime, self.min_action, self.max_action)
        self.time_step += 1

        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)

        target_actions = self.target_actor.forward(state_)
        noise = self.noise() * 0.0005
        target_actions = target_actions + T.tensor(noise, dtype=T.float).to(self.actor.device)
        # target_actions = T.clamp(target_actions, -1.0005, 1.0005)
        # target_actions = T.tensor(np.clip(target_actions.cpu().detach().numpy(),
        #                                   self.min_action, self.max_action)).to(self.actor.device)
        q1_ = self.target_critic_1.forward(state_, target_actions)
        q2_ = self.target_critic_2.forward(state_, target_actions)

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        q1_[done] = 0.0
        q2_[done] = 0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = T.min(q1_, q2_)

        target = reward + self.gamma * critic_value_
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1 = dict(critic_1_params)
        critic_2 = dict(critic_2_params)
        actor = dict(actor_params)
        target_actor = dict(target_actor_params)
        target_critic_1 = dict(target_critic_1_params)
        target_critic_2 = dict(target_critic_2_params)

        for name in critic_1:
            critic_1[name] = tau * critic_1[name].clone() + \
                             (1 - tau) * target_critic_1[name].clone()

        for name in critic_2:
            critic_2[name] = tau * critic_2[name].clone() + \
                             (1 - tau) * target_critic_2[name].clone()

        for name in actor:
            actor[name] = tau * actor[name].clone() + \
                          (1 - tau) * target_actor[name].clone()

        self.target_critic_1.load_state_dict(critic_1)
        self.target_critic_2.load_state_dict(critic_2)
        self.target_actor.load_state_dict(actor)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()
