import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from collections import deque
import random


# 定义网络结构
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))  # 使用 tanh 确保动作在 [-1, 1] 之间
        return action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(torch.cat([state, action], dim=-1)))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class DDPG:
    def __init__(self, state_dim, action_dim, hidden_dim, lr=1e-3, gamma=0.99, tau=1e-2, buffer_size=int(1e6),
                 batch_size=64):
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, action_dim, hidden_dim)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

        # Initialize target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def select_action(self, state, noise=0.0):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        action += noise * np.random.randn(*action.shape)
        return np.clip(action, -1, 1)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        # Sample a batch of transitions
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Update Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q_values = self.critic_target(next_states, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * target_q_values

        q_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q_values, target_q_values)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Update Actor
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Soft update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def add_to_buffer(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))


# 训练过程
def train(env_name, num_episodes=1000):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    ddpg = DDPG(state_dim, action_dim, hidden_dim=256)

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        while True:
            action = ddpg.select_action(state, noise=0.1)
            next_state, reward, done, _ = env.step(action)
            ddpg.add_to_buffer(state, action, reward, next_state, done)
            ddpg.update()
            state = next_state
            episode_reward += reward

            if done:
                break

        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")


if __name__ == "__main__":
    train('Pendulum-v0', num_episodes=100)
