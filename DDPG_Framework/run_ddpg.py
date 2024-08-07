import gym
from DDPG_Framework.DDPG import DDPG
import os
import matplotlib.pyplot as plt

from DDPG_Framework.Environment import CarOffloadEnv


def train(env_name, num_episodes=1000):
    env = CarOffloadEnv(num_agents=3)                                           # 1个env
    state_dim = env.get_state_size()
    action_dim = env.get_action_size()
    ddpg = DDPG(state_dim, action_dim, hidden_dim=256)

    # 创建结果保存文件夹
    if not os.path.exists('Result'):
        os.makedirs('Result')

    # 用于记录每一集的奖励
    rewards = []

    # 打开文件以写入结果
    with open('Result/training_results.txt', 'w') as file:
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

            # 打印并保存每一集的奖励
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")
            file.write(f"Episode {episode + 1}, Reward: {episode_reward}\n")
            rewards.append(episode_reward)

    # 保存奖励图像
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid()
    plt.savefig('Result/training_rewards.png')
    plt.close()


if __name__ == "__main__":
    train('CarOffloadEnv-01', num_episodes=5000)
