
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from DDPG_Framework.Network import Actor, Critic

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
        # 添加噪声，这里假设 noise 是噪声的强度
        if noise > 0:
            action += noise * np.random.randn(*action.shape)  # 或使用其他噪声机制

        # 将动作限制在 [0, 1] 范围内
        action = np.clip(action, 0, 1)

        return action

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
            # 使用 repeat 扩展张量的形状
            rewards_repeated = rewards.repeat(1, 3).unsqueeze(2)  # 扩展为 [64,3,1]
            # 使用 repeat 扩展张量的形状
            dones_repeated = dones.repeat(1, 3).unsqueeze(2)  # 扩展为 [64,3,1]

            target_q_values = rewards_repeated + (1 - dones_repeated) * self.gamma * target_q_values

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