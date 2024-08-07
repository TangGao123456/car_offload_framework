import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F


# 智能体相关，初始化、设置epsilon-greedy策略的参数、网络的训练
class DQNAgent:
    # 初始化
    def __init__(self, env, model, target_model, memory, batch_size=32, gamma=0.99, epsilon_start=1.0,
                 epsilon_end=0.01, epsilon_decay=0.995, target_update_freq=100):
        self.env = env                              # 出现env
        self.model = model
        self.target_model = target_model
        self.memory = memory
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        # self.optimizer = optim.Adam(model.parameters())
        self.optimizer = optim.Adam(model.parameters(), lr=0.1)

        self.loss_fn = nn.MSELoss()

        self.training_rewards = []  # 用于存储每个episode的总奖励

    # 设置epsilon-greedy策略的参数
    def epsilon_greedy_policy(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self.model(torch.tensor(state, dtype=torch.float32))
                return q_values.argmax().item()

    # 训练
    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            numbers = 0

            while not done:
                action = self.epsilon_greedy_policy(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                numbers = numbers + 1

                self.memory.add((state, action, reward, next_state, 1 if done else 0))
                state = next_state

                if len(self.memory) > self.batch_size:
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = self.memory.sample(
                        self.batch_size)
                    batch_states_np = np.array(batch_states, dtype=np.float32)
                    batch_states = torch.tensor(batch_states_np, dtype=torch.float32)
                    batch_actions = torch.tensor(batch_actions, dtype=torch.long).unsqueeze(1)
                    batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32)
                    batch_next_states_np = np.array(batch_next_states, dtype=np.float32)
                    batch_next_states = torch.tensor(batch_next_states_np, dtype=torch.float32)
                    batch_dones = torch.tensor(batch_dones, dtype=torch.float32)

                    # q_values是32组，每组n个智能体，每个智能体对应2个Q值，对应形状[32,n,2]
                    q_values = self.model(batch_states)     # [32,n,2]
                    # 那next_q_values就应该选取Q值的大的那一个，对应形状[32,3,1]，即对每个智能体都选取最大的那个Q值
                    next_q_values = self.target_model(batch_next_states).max(dim=2)[0]  # [32, n]

                    # 因为 batch_rewards、batch_dones 形状都是[32]，为了计算target，需要和 next_q_values 的形状匹配
                    batch_rewards = batch_rewards.unsqueeze(1)  # [32, 1]
                    batch_dones = batch_dones.unsqueeze(1)  # [32, 1]
                    targets = batch_rewards + (1 - batch_dones) * self.gamma * next_q_values    # [32,n]

                    # 下面2行代码的作用：把 q_values 的形状变为[32,n,2]
                    # 在第三个维度上随机选择一个 Q 值
                    random_q_index = torch.randint(2, size=(32, 3))
                    # 使用 gather 操作获取随机选择的 Q 值
                    selected_q_values = q_values.gather(dim=2, index=random_q_index.unsqueeze(-1)).squeeze(-1)

                    loss = F.mse_loss(selected_q_values, targets)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                if episode % self.target_update_freq == 0:
                    self.target_model.load_state_dict(self.model.state_dict())


            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            self.training_rewards.append(total_reward)  # 记录每个episode的总奖励

            print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

    # 保存结果
    def save_training_results(self, save_path):
        result_dir = os.path.join(os.getcwd(), 'Result')
        os.makedirs(result_dir, exist_ok=True)

        save_path = os.path.join(result_dir, save_path)

        with open(save_path, 'w') as f:
            for i, reward in enumerate(self.training_rewards):
                f.write(f"Episode {i + 1}: {reward}\n")

        print(f"Training results saved to {save_path}")

    # 把结果绘制成图像
    def plot_training_results(self, save_filename='training_results_plot.png'):
        result_dir = os.path.join(os.getcwd(), 'Result')
        save_path = os.path.join(result_dir, save_filename)

        episodes = []
        rewards = []

        with open(os.path.join(result_dir, 'training_results.txt'), 'r') as f:
            for line in f:
                episode, reward = line.strip().split(': ')
                episodes.append(int(episode.split()[1]))
                rewards.append(float(reward))

        plt.plot(episodes, rewards, marker='o')
        plt.title('Training Results')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        plt.savefig(save_path)
        plt.show()



