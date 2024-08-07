from DQN_Framework.Environment import CarOffloadEnv
from DQN_Framework.Network import QNetwork
from DQN_Framework.Agent import DQNAgent
from DQN_Framework.Buffer import ReplayMemory


# 创建环境
# env = gym.make('CartPole-v1')
env = CarOffloadEnv(num_agents=3)
state_size = env.get_state_size()       # 3
action_size = env.get_action_size()     # 2


# 初始化模型和记忆
model = QNetwork(state_size, action_size)               # 创建网络
target_model = QNetwork(state_size, action_size)        # 创建网络
memory = ReplayMemory(capacity=10000)

# 初始化DQN智能体
agent = DQNAgent(env, model, target_model, memory)

# 训练智能体
agent.train(num_episodes=5000)

# 保存训练结果到txt文件
save_filename = 'training_results.txt'
agent.save_training_results(save_filename)

# 绘制训练结果并保存图像
plot_filename = 'training_results_plot.png'
agent.plot_training_results(plot_filename)







