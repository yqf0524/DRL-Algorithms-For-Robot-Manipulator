import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
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
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, 
                 fc3_dims, fc4_dims, fc5_dims, fc6_dims, max_action, 
                 n_actions, name, checkpoint_dir='SAC_model'):
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
        self.checkpoint_file = os.path.join(checkpoint_dir, name + '_sac.pth')
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        self.fc5 = nn.Linear(self.fc4_dims, self.fc5_dims)
        self.fc6 = nn.Linear(self.fc5_dims, self.fc6_dims)
        self.mu = nn.Linear(self.fc5_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc5_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)
        prob = self.fc3(prob)
        prob = F.relu(prob)
        prob = self.fc4(prob)
        prob = F.relu(prob)
        prob = self.fc5(prob)
        prob = F.relu(prob)
        # prob = self.fc6(prob)
        # prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = T.clamp(sigma, min=self.reparam_noise, max=0.1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = T.tanh(actions) * T.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(-1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        print('saving checkpoint ' + self.checkpoint_file + ' ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('loading checkpoint ' + self.checkpoint_file + ' ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, fc3_dims, fc4_dims,
                  fc5_dims, fc6_dims, n_actions, name, checkpoint_dir='SAC_model'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.fc5_dims = fc5_dims
        self.fc6_dims = fc6_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(checkpoint_dir, name + '_sac.pth')

        self.fc1 = nn.Linear(self.input_dims + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        self.fc5 = nn.Linear(self.fc4_dims, self.fc5_dims)
        self.fc6 = nn.Linear(self.fc5_dims, self.fc6_dims)
        self.q = nn.Linear(self.fc5_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1).float())
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = self.fc3(action_value)
        action_value = F.relu(action_value)
        action_value = self.fc4(action_value)
        action_value = F.relu(action_value)
        action_value = self.fc5(action_value)
        action_value = F.relu(action_value)
        # action_value = self.fc6(action_value)
        # action_value = F.relu(action_value)

        q = self.q(action_value)

        return q
    
    def save_checkpoint(self):
        print('saving checkpoint ' + self.checkpoint_file + ' ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('loading checkpoint ' + self.checkpoint_file + ' ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, fc3_dims, fc4_dims,
                  fc5_dims, fc6_dims, name, checkpoint_dir='SAC_model'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.fc4_dims = fc4_dims
        self.fc5_dims = fc5_dims
        self.fc6_dims = fc6_dims
        self.checkpoint_file = os.path.join(checkpoint_dir, name + '_sac.pth')

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        self.fc5 = nn.Linear(self.fc4_dims, self.fc5_dims)
        self.fc6 = nn.Linear(self.fc5_dims, self.fc6_dims)
        self.v = nn.Linear(self.fc5_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc3(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc4(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc5(state_value)
        state_value = F.relu(state_value)
        # state_value = self.fc6(state_value)
        # state_value = F.relu(state_value)

        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        print('saving checkpoint ' + self.checkpoint_file + ' ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('loading checkpoint ' + self.checkpoint_file + ' ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:
    def __init__(self, alpha, beta, tau, env, input_dims, n_actions, layer1_size,
                 layer2_size, layer3_size, layer4_size, layer5_size, layer6_size, 
                 gamma=0.99, max_size=1000000, batch_size=100, reward_scale = 2,
                 checkpoint_dir='SAC_model'):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.scale = reward_scale

        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, layer3_size,
                                  layer4_size, layer5_size, layer6_size,
                                  max_action=env.action_space_high, n_actions=n_actions, 
                                  name='actor', checkpoint_dir=checkpoint_dir)

        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, layer3_size,
                                      layer4_size, layer5_size, layer6_size, n_actions=n_actions, 
                                      name='critic_1', checkpoint_dir=checkpoint_dir)
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, layer3_size,
                                      layer4_size, layer5_size, layer6_size, n_actions=n_actions, 
                                      name='critic_2', checkpoint_dir=checkpoint_dir)
        
        self.value = ValueNetwork(beta, input_dims, layer1_size, layer2_size, layer3_size,
                                  layer4_size, layer5_size, layer6_size, name='value', 
                                  checkpoint_dir=checkpoint_dir)
        self.target_value = ValueNetwork(beta, input_dims, layer1_size, layer2_size, layer3_size,
                                  layer4_size, layer5_size, layer6_size, name='target_value', 
                                  checkpoint_dir=checkpoint_dir)

        self.update_network_parameters(tau=0.99)

    def choose_action(self, observation):
        state = T.Tensor(observation).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau * value_state_dict[name].clone() + \
                                     (1 - tau) * target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        value = self.value.forward(state).view(-1)
        value_ = self.target_value.forward(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()