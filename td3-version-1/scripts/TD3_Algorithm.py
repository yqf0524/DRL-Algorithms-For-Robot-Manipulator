import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os


class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, _state, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = _state
        self.terminal_memory[index] = done
        self.reward_memory[index] = reward
        self.action_memory[index] = action

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        _states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, _states, dones

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, fc3_dims, 
                 fc4_dims, fc5_dims, fc6_dims, max_action, 
                 n_actions, name, checkpoint_dir='TD3_model'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.fc5_dims = fc5_dims
        self.fc6_dims = fc6_dims
        self.n_actions = n_actions
        self.max_action = max_action
        self.checkpoint_file = os.path.join(checkpoint_dir, name + '_td3.pth')

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        self.fc5 = nn.Linear(self.fc4_dims, self.fc5_dims)
        self.fc6 = nn.Linear(self.fc5_dims, self.fc6_dims)
        self.mu = nn.Linear(self.fc6_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.dropout(prob)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.dropout(prob)
        prob = F.relu(prob)
        prob = self.fc3(prob)
        prob = F.dropout(prob)
        prob = F.relu(prob)
        prob = self.fc4(prob)
        prob = F.dropout(prob)
        prob = F.relu(prob)
        prob = self.fc5(prob)
        prob = F.dropout(prob)
        prob = F.relu(prob)
        prob = self.fc6(prob)
        prob = F.dropout(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        mu = T.tanh(mu) * T.tensor(self.max_action).to(self.device)

        return mu

    def save_checkpoint(self):
        print('saving checkpoint ' + self.checkpoint_file + ' ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('loading checkpoint ' + self.checkpoint_file + ' ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, fc3_dims, fc4_dims,
                  fc5_dims, fc6_dims, n_actions, name, checkpoint_dir='TD3_model'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.fc5_dims = fc5_dims
        self.fc6_dims = fc6_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(checkpoint_dir, name + '_td3.pth')

        self.fc1 = nn.Linear(self.input_dims + self.n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        self.fc5 = nn.Linear(self.fc4_dims, self.fc5_dims)
        self.fc6 = nn.Linear(self.fc5_dims, self.fc6_dims)
        self.q = nn.Linear(self.fc6_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1).float())
        action_value = F.dropout(action_value)
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.dropout(action_value)
        action_value = F.relu(action_value)
        action_value = self.fc3(action_value)
        action_value = F.dropout(action_value)
        action_value = F.relu(action_value)
        action_value = self.fc4(action_value)
        action_value = F.dropout(action_value)
        action_value = F.relu(action_value)
        action_value = self.fc5(action_value)
        action_value = F.dropout(action_value)
        action_value = F.relu(action_value)
        action_value = self.fc6(action_value)
        action_value = F.dropout(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        print('saving checkpoint ' + self.checkpoint_file + ' ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('loading checkpoint ' + self.checkpoint_file + ' ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class OrnsteinUhlenbeckNoise:
    """
    A Ornstein Uhlenbeck action noise, this is designed to aproximate brownian motion with friction.

    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

    :param dim: (tuple) the dimension of the noise
    :param mu: (float) the mean of the noise
    :param theta: (float) the rate of mean reversion, affect converge
    :param sigma: (float) the scale of the noise, affect random
    :param dt: (float) the timestep for the noise
    """

    def __init__(self, dim, mu=0, theta=0.15, sigma=0.2, dt=1.0):
        self.dim = dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()

    def reset(self):
        self.X_prev = np.ones(self.dim) * self.mu

    def __call__(self):
        drift = self.theta * (self.mu - self.X_prev) * self.dt
        random = self.sigma * np.sqrt(self.dt) * np.random.randn(self.dim)

        self.X = self.X_prev + drift + random

        return self.X


class Agent:
    def __init__(self, alpha, beta, tau, env, input_dims, n_actions, layer1_size,
                 layer2_size, layer3_size, layer4_size, layer5_size, layer6_size,
                 gamma=0.99, update_actor_interval=2, 
                 warmup=0, max_size=1000000, batch_size=128, 
                 noise=0.02, checkpoint_dir='TD3_model'):
        self.gamma = gamma
        self.tau = tau
        self.env = env
        self.max_action = env.action_space_high
        self.min_action = env.action_space_low
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.time_step_counter = 0
        self.learn_step_cntr = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, layer3_size, 
                                  layer4_size, layer5_size, layer6_size,
                                  max_action=self.max_action, n_actions=n_actions, 
                                  name='actor', checkpoint_dir=checkpoint_dir)

        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, layer3_size,
                        layer4_size, layer5_size, layer6_size, n_actions=n_actions,
                        name='critic_1', checkpoint_dir=checkpoint_dir)
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, layer3_size, 
                        layer4_size, layer5_size, layer6_size, n_actions=n_actions,
                        name='critic_2', checkpoint_dir=checkpoint_dir)

        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, layer3_size,
                            layer4_size, layer5_size, layer6_size,
                            max_action=self.max_action, n_actions=n_actions, 
                            name='target_actor', checkpoint_dir=checkpoint_dir)

        self.target_critic_1 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, layer3_size, 
                               layer4_size, layer5_size, layer6_size, n_actions=n_actions,
                               name='target_critic_1', checkpoint_dir=checkpoint_dir)
        self.target_critic_2 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, layer3_size, 
                               layer4_size, layer5_size, layer6_size, n_actions=n_actions,
                               name='target_critic_2', checkpoint_dir=checkpoint_dir)

        self.noise = noise  # noise = 0.02 degree per sec
        # OrnsteinUhlenbeckNoise() in range around [-1, 1]
        # self.noise_action = np.dot(OrnsteinUhlenbeckNoise(self.n_actions)(), self.noise) 
        # self.noise_action_target = np.dot(OrnsteinUhlenbeckNoise(self.n_actions)(), self.noise)
        
        self.update_network_parameters(tau=1)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        # observation in interval [-1, 1]
        observation = np.divide(observation, self.env.observation_space_high)

        if self.time_step_counter < self.warmup:
            # Firstly, select action randomly and collect data to the Buffer 
            # for the future updating of actor networks
            mu = np.random.normal(scale=0.5, size=7) 
            # mu = np.clip(mu, self.min_action, self.max_action, dtype=np.float32)
            # mu = np.dot(mu, self.max_action)
        else:
            # After get enough data, then select action with actor network. 
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
            # Convert torch tensor mu to numpy.narray.
            mu = mu.cpu().detach().numpy()
        # print("choose_action:", mu[0:5])
        # Create exploration noise, in interval [-0.02, 0.02] degree per sec
        self.action_noise = np.clip(np.random.normal(scale=0.02, size=self.n_actions), 
                                    -self.noise, self.noise, dtype=np.float32)
        # Add exploration noise to selected action.
        mu_prime = mu + self.action_noise
        mu_prime = np.clip(mu_prime, self.min_action, self.max_action, dtype=np.float32)
        # # Convert numpy.narray mu_prime to torch tensor.
        # mu_prime = T.tensor(mu_prime, dtype=T.float).to(self.actor.device)
        # mu_prime.cpu().detach().numpy()
        self.time_step_counter += 1
        return mu_prime

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        # Sample mini-bach of N transitions from Buffer
        states, actions, rewards, _states, dones = self.memory.sample_buffer(self.batch_size)
        # states in interval [-1, 1]
        states = np.divide(states, self.env.observation_space_high)
        _states = np.divide(_states, self.env.observation_space_high)

        states = T.tensor(states, dtype=T.float).to(self.critic_1.device)
        actions = T.tensor(actions, dtype=T.float).to(self.critic_1.device)
        _states = T.tensor(_states, dtype=T.float).to(self.critic_1.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.critic_1.device)
        dones = T.tensor(dones).to(self.critic_1.device)

        # Select next action according to target policy
        target_actions = self.target_actor.forward(_states).to(self.target_actor.device)
        target_actions_noise = np.clip(np.random.normal(scale=0.02, size=(self.batch_size, self.n_actions)),
                                       -self.noise, self.noise, dtype=np.float32)
        # Add exploration noise to target_actions
        target_actions = np.add(target_actions.cpu().detach().numpy(), target_actions_noise)
        target_actions = np.clip(target_actions, self.min_action, self.max_action, dtype=np.float32)
        # print("choose_target_action:", target_actions[0:5, 0:5])

        # Convert numpy() target_actions to torch tensor.
        target_actions = T.tensor(target_actions, dtype=T.float).to(self.target_actor.device)

        # Computer current Q_value: critic_networks
        q1 = self.critic_1.forward(states, actions).view(-1)
        q2 = self.critic_2.forward(states, actions).view(-1)

        # Computer target Q_value: target_critic_networks
        q1_ = self.target_critic_1.forward(_states, target_actions).view(-1)
        q2_ = self.target_critic_2.forward(_states, target_actions).view(-1)
        q1_[dones] = 0.0
        q2_[dones] = 0.0
        critic_value_ = T.min(q1_, q2_)
        target = rewards + self.gamma * critic_value_

        # Computer critic loss
        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)
        critic_loss = q1_loss + q2_loss

        # Optimize critic networks
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learn_step_cntr += 1

        # Delayed policy updates:
        if self.learn_step_cntr % self.update_actor_iter == 0:
            # Computer actor value and loss function
            # print("state:", states[0:5])
            policy_actions = self.actor.forward(states).to(self.actor.device)
            # print("delayed_action:", policy_actions[0:5])
            actor_q1_values = self.critic_1.forward(states, policy_actions)
            actor_loss = -T.mean(actor_q1_values)
            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            # Update network parameters
            self.new_update_network_parameters(self.tau)

    def update_network_parameters(self, tau):
        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        actor = dict(actor_params)
        critic_1 = dict(critic_1_params)
        critic_2 = dict(critic_2_params)
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
    
    def new_update_network_parameters(self, tau):
        with T.no_grad():
            for actor_params, target_actor_params in zip(self.actor.parameters(), \
                                            self.target_actor.parameters()):
                target_actor_params.data.copy_(tau * actor_params.data + \
                                        (1 - tau) * target_actor_params.data)
            
            for critic_1_params, target_critic_1_params in zip(self.critic_1.parameters(), \
                                            self.target_critic_1.parameters()):
                target_critic_1_params.data.copy_(tau * critic_1_params.data + \
                                        (1 - tau) * target_critic_1_params.data)

            for critic_2_params, target_critic_2_params in zip(self.critic_2.parameters(), \
                                            self.target_critic_2.parameters()):
                target_critic_2_params.data.copy_(tau * critic_2_params.data + \
                                        (1 - tau) * target_critic_2_params.data)

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
    