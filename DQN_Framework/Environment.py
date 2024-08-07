import numpy as np
import random


# 智能体数量：10          相当于10辆车
# 1个RSU：假设一个横向的长度为1600m，[0,1600]。
#        RSU的位置：x = 800 ， 通信范围[300,1300]
# 智能体从x = 0 出发，速度为20m/s。
# 智能体随机产生[10-100]mb的任务
# 在RSU范围内，智能体有两种选择：     0：本地处理  ；  1：卸载到服务器
#                   RSU处理能力：50mb/s       车载处理能力：10mb/s
#                   假设任务传给RSU的时间和接收结果的时间为：1s
#
#                   任务卸载时间：
#                       如果本地处理： T_local） = 任务大小 / 车载处理能力
#                       如果卸载处理： T_offload = 任务大小 / RSU处理能力 + 1
#                   奖励： R = 20 / T
#
# 写代码思路：
#       步骤1：一开始可以不设置是否在RSU范围内，就默认在RSU范围内，直接选择卸载不卸载。然后没有移动速度。
#       步骤2：步骤1运行成功后，在设置RSU范围，有个速度。并且，如果选择卸载了，然后出了RSU范围，要设置一个负面的奖励

class ActionSpace:
    def __init__(self, actions):
        self.actions = actions  # 定义动作空间中的所有动作
        self.n = len(actions)  # 动作空间的大小

    def sample(self):
        return random.choice(self.actions)  # 从动作中随机选择一个

    def contains(self, action):
        """检查给定动作是否在动作空间中"""
        return action in self.actions

class CarOffloadEnv:
    # 定义一些参数
    process_rsu = 50    # rsu的处理能力
    process_car = 10    # 车载处理能力

    # num_agents=1 表示如果在创建 CarOffloadEnv 实例时没有明确指定 num_agents 参数，默认情况下环境将有 1 个智能体
    # 使用的例子：env_multiple_agents = CarOffloadEnv(num_agents=3)
    def __init__(self, num_agents=1, min_task_size=10, max_task_size=100,
                 process_rsu=50, process_car=10):
        # 定义状态空间和动作空间
        self.action_space = ActionSpace([0, 1])  # 动作空间包含两个动作: 0（本地处理）和 1（卸载）
        self.state_space = (num_agents, 3)  # 状态空间包括智能体的索引、任务大小和是否完成任务的标志
        self.num_agents = num_agents
        self.min_task_size = min_task_size
        self.max_task_size = max_task_size
        self.process_rsu = process_rsu  # 固定参数
        self.process_car = process_car  # 固定参数

        # 初始化状态
        self.state = None
        self.reset()

    # 获取状态空间
    def get_state_size(self):
        # 获取实际状态空间的特征数量
        return self.state_space[1]

    # 获取动作空间的大小
    def get_action_size(self):
        """获取动作空间的大小"""
        return self.action_space.n

    def reset(self):
        """重置环境到初始状态"""
        # 初始化状态，包含智能体的索引、随机任务大小和是否完成任务的标志
        self.state = np.zeros((self.num_agents, 3))  # 状态为全零矩阵
        for i in range(self.num_agents):
            self.state[i, 0] = i  # 设置第一个参数为智能体的索引
            self.state[i, 1] = np.random.uniform(self.min_task_size, self.max_task_size)  # 设置第二个参数为随机任务大小
            self.state[i, 2] = 0  # 第三个参数表示任务完成状态，初始化为0（未完成）        如果为1，就表示完成了
        return self.state

    def step(self, action):
        """执行一个动作并返回新的状态、奖励、是否结束和额外信息"""
        if not self.action_space.contains(action):
            # raise ValueError("无效的动作")
            action = self.action_space.sample()  # 如果动作不在动作空间中，重新选择一个动作

        # 简单的逻辑: 根据动作更新状态
        done = False
        reward = 0
        for i in range(self.num_agents):
            T_local = 0
            T_offload = 0
            if action == 0:     # 本地处理
                T_local = self.state[i, 1] / self.process_car
                self.state[i, 2] = 1
            elif action == 1:   # 卸载
                T_offload = self.state[i, 1] / self.process_rsu
                self.state[i, 2] = 1
            T = int(T_local + T_offload)

            if T > 0:
                reward += int(20 / T)
            else:
                reward += 0  # 如果 T 为 0，则奖励为 0

        done = True
        reward = int(reward) / self.num_agents  # 奖励取整
        return self.state, reward, done, {}


    def render(self):
        """渲染当前环境状态（这里只是打印状态）"""
        print(f"当前状态: {self.state}")

    def close(self):
        """关闭环境（没有实际作用，但在复杂环境中可能会有资源清理）"""
        pass

    def make(env_name):
        if env_name == 'CarOffloadEnv-01':
            return CarOffloadEnv()
        else:
            raise ValueError(f"未知的环境名称: {env_name}")


