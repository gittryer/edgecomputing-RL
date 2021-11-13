import numpy as np
import pandas as pd
from env import Environment
from env import ENV_TIME_LIMIT, SUBCARRIER_B, SUBCARRIER_O, SUBCARRIER_GO
import math


<<<<<<< HEAD
class QLearning:
    def __init__(self, e: Environment, learning_rate=0.7, gamma=0.65, epsilon=0.55):
=======
# 默认的损失函数
def loss_default(f):
    return 1 / f


class Bandit:
    """
    摇臂机模型
    """

    # 累计收益
    reward_all = 0

    def __init__(self, state_num=20, action_num=10, epsilon=0.7):
        """
        构造函数
        :param state_num: 状态个数
        :param action_num: 号码的个数
        :param epsilon: 贪婪度
        """
        self.state_num = state_num
        self.action_num = action_num
        self.K = np.zeros((state_num, action_num))  # N个摇臂摇中的次数
        self.Q = np.zeros((state_num, action_num))  # N摇臂的平均收益
        self.epsilon = epsilon

    def select(self, state):
        """
        选择action
        :param state: 状态
        :param epsilon: 贪心度
        :return: 返回选择的号码
        """
        # 随机选
        if np.random.uniform() > self.epsilon:
            return np.random.randint(0, self.action_num)
        # 选择平均收益最高的
        else:
            return self.getMaxIndex(state)

    def getMaxIndex(self, state):
        """
        多个相同最大值，随机选择他们的索引
        :param state: 状态
        :return: 返回索引
        """
        R = self.Q[state]
        ls_index = []
        max = R.max()
        ls_index.append(R.argmax())
        for i in range(ls_index[0], len(R)):
            if math.isclose(max, R[i], abs_tol=0.00001):
                ls_index.append(i)
        return ls_index[np.random.randint(0, len(ls_index))]

    def update(self, state, action, reward):
        """
        更新摇中次数和平均收益
        :param state: 状态
        :param action: 行为
        :param reward: 收益
        :return: 返回收益
        """
        # 更新摇中次数
        self.K[state, action] += 1
        # 更新累计平均收益
        self.Q[state, action] = self.Q[state, action] + \
                                (reward - self.Q[state, action]) / self.K[state, action]
        Bandit.reward_all += reward


class QLearning:
    def __init__(self, e: Environment, loss=loss_default, learning_rate=0.7, gamma=0.65, epsilon=0.55):
>>>>>>> 90a41b2 (修改了一些参数的设定值)
        """
        构造器
        :param learning_rate: 学习率
        :param gamma: 衰减率
        :param epsilon: 使用已有资源你的概率
        :param states: 状态数
        :param actions: 行为数(mec数)
        :param env: 环境
        """
<<<<<<< HEAD
        self.e = e
=======
        # 环境
        self.e = e
        # 损失函数
        self.loss = loss
>>>>>>> 90a41b2 (修改了一些参数的设定值)
        # 学习率p
        self.learning_rate = learning_rate
        # 衰减率
        self.gamma = gamma
        # 利用已有资源的概率
        self.epsilon = epsilon
<<<<<<< HEAD
        # 状态数
        self.states = self.e.service_num
        # 行为数
        self.actions = self.e.mec_num * self.e.subcarrier_num
=======
>>>>>>> 90a41b2 (修改了一些参数的设定值)
        # 行为表(mec编号,子信道编号)
        self.actionTable = []
        for i in range(self.e.mec_num):
            for j in range(self.e.subcarrier_num):
                self.actionTable.append((i, j))
        # 初始化Q表
<<<<<<< HEAD
        self.QTable = pd.DataFrame(data=[[0 for item in range(self.actions)] for item in range(self.states + 1)],
                                   index=range(self.states + 1), columns=range(self.actions))
=======
        self.QTable = pd.DataFrame(np.zeros((self.e.states, self.e.actions)))
        self.bandit = Bandit(self.QTable.shape[0], self.QTable.shape[1], self.epsilon)
>>>>>>> 90a41b2 (修改了一些参数的设定值)
        pd.set_option('display.max_columns', 1000)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', 1000)

<<<<<<< HEAD
    def select(self, state):
        """
        选择行为
        :param state: 当前状态
        :return: 返回行为
        """
        # 选择策略：避免局部最优
        # 随机结果为[0,epsilon]时选择Q值最高的action，否则随机选择
        val = self.QTable.loc[state, :].max()
        # 探索:
        if np.random.uniform() > self.epsilon or val == 0:
            return np.random.randint(0, self.actions)
        # 利用
        else:
            return self.QTable.loc[state, :].idxmax()

    def learn(self, state, action, next_state, reward):
        """
        更新Q表
        :param state: 当前状态
        :param action: 选择的行为
        :param next_state: 下一个状态
        :param reward: 反馈的收益
        """
        Q_predict = self.QTable.loc[state, action]
        Q_new = reward + self.gamma * (self.QTable.loc[next_state, :].idxmax())
        self.QTable.loc[state, action] = \
            self.QTable.loc[state, action] + self.learning_rate * (Q_new - Q_predict)
=======
        # 从Q-table中选取动作

    def select(self, state):
        """
        选择行为
        :param state: 状态
        :return: 返回选择的行为编号
        """
        return self.bandit.select(state)

    def learn(self, state, action, reward, next_state, done):
        """
        学习
        :param state: 状态
        :param action: 行为
        :param reward: 收益
        :param next_state: 下一个状态
        :param done: 是否结束
        """
        q_predict = self.QTable.loc[state, action]
        if done:
            q_new = reward
        else:
            q_new = reward + self.gamma * self.QTable.max(axis=1)[next_state]
        self.QTable.loc[state, action] += self.learning_rate * (q_new - q_predict)
        # 更新摇臂机
        self.bandit.update(state, action, reward)
>>>>>>> 90a41b2 (修改了一些参数的设定值)

    def run(self, cnt=10):
        """
        运行Q学习
        :param cnt: 学习次数
        """
        for item in range(cnt):
            state = self.e.reset()
            end = False
            while not end:
                action = self.select(state)
                (x, y) = self.actionTable[action]
<<<<<<< HEAD
                (k, p) = (self.__getK(state, x), self.__getP(state, x, y))
                reward, end = self.e.forward(state, x, y, p, k)
                if item % 100 == 0:
                    print(self.QTable)
                next_state = state + 1
                self.learn(state, action, next_state, reward)
=======
                reward, end = self.e.forward(state, x, y, self.e.getP(state, x, y),self.e.getK(state, x))
                reward = self.loss(reward)
                # if item % 100 == 0:
                #     print(self.QTable)
                next_state = state + 1
                self.learn(state, action, reward, next_state, end)
>>>>>>> 90a41b2 (修改了一些参数的设定值)
                state = next_state

    def play(self):
        """
        学有所成
        """
        ans = []
        self.epsilon = 1.
        state = self.e.reset()
        end = False
        while not end:
            action = self.select(state)
            (x, y) = self.actionTable[action]
<<<<<<< HEAD
            (k, p) = (self.__getK(state, x), self.__getP(state, x, y))
            print('{}:添加到{}号MEC,{}号子信道,传输功率{:.2f},主频{:.2f}'.format(state, x, y, p, k))
            reward, end = self.e.forward(state, x, y, p, k)
            ans.append(-reward)
            next_state = state + 1
            state = next_state
        return [np.arange(self.e.service_num).astype(dtype=np.str), ans]
=======
            reward, end = self.e.forward(state, x, y,
                                self.e.getP(state, x, y),self.e.getK(state, x))
            reward = self.loss(reward)
            # print('{}:添加到{}号MEC,{}号子信道,传输功率{:.2f},主频{:.2f}'.format(state, x, y, p, k))
            #if state % 5 == 0:
            ans.append(self.e.POWER)
            next_state = state + 1
            state = next_state
        return range(0, self.e.service_num), np.array(ans)
        # return range(0, self.e.service_num, 5), np.array(ans)
>>>>>>> 90a41b2 (修改了一些参数的设定值)

    def runForAVG(self, cnt=10):
        """
        运行Q学习
        :param cnt: 学习次数
        """
        for item in range(cnt):
            state = self.e.reset()
            end = False
            while not end:
                action = self.select(state)
                (x, y) = self.actionTable[action]
<<<<<<< HEAD
                (k, p) = (self.__getK(state, x), self.__getP(state, x, y))
                reward, end = self.e.forward(state, x, y, p, k)
                reward = reward / sum([self.e.services[j].data_size for j in range(0, state + 1)])
                if item % 100 == 0:
                    print(self.QTable)
                next_state = state + 1
                self.learn(state, action, next_state, reward)
=======
                reward, end = self.e.forward(state, x, y, self.e.getP(state, x, y),self.e.getK(state, x))
                reward = self.loss(sum(self.e.getPower(state,x,y)) / self.e.services[state].data_size)
                # if item % 100 == 0:
                #     print(self.QTable)
                next_state = state + 1
                self.learn(state, action, reward, next_state, end)
>>>>>>> 90a41b2 (修改了一些参数的设定值)
                state = next_state

    def playForAVG(self):
        """
        学有所成
        """
        ans = []
        self.epsilon = 1.
        state = self.e.reset()
        end = False
        while not end:
            action = self.select(state)
            (x, y) = self.actionTable[action]
<<<<<<< HEAD
            (k, p) = (self.__getK(state, x), self.__getP(state, x, y))
            # print('{}:添加到{}号MEC,{}号子信道,传输功率{:.2f},主频{:.2f}'.format(state, x, y, p, k))
            reward, end = self.e.forward(state, x, y, p, k)
            reward = reward / sum([self.e.services[j].data_size for j in range(0, state + 1)])
            ans.append(-reward)
            print("平均每bit能耗为", ans[-1])
            next_state = state + 1
            state = next_state
        return [np.arange(self.e.service_num).astype(dtype=np.str), ans]

    def runForTime(self, cnt=10):
        """
        运行Q学习
        :param cnt: 学习次数
        """
        for item in range(cnt):
            state = self.e.reset()
            end = False
            while not end:
                action = self.select(state)
                (x, y) = self.actionTable[action]
                (k, p) = (self.__getK(state, x), self.__getP(state, x, y))
                reward, end = self.e.forward(state, x, y, p, k)
                if item % 100 == 0:
                    print(self.QTable)
                next_state = state + 1
                self.learn(state, action, next_state, reward)
                state = next_state

    def playForTime(self):
        """
        学有所成
        """
        ans = []
        self.epsilon = 1.
        state = self.e.reset()
        end = False
        while not end:
            action = self.select(state)
            (x, y) = self.actionTable[action]
            (k, p) = (self.__getK(state, x), self.__getP(state, x, y))
            # print('{}:添加到{}号MEC,{}号子信道,传输功率{:.2f},主频{:.2f}'.format(state, x, y, p, k))
            reward, end = self.e.forward(state, x, y, p, k)
            ans.append(sum(self.e.getTime(state,x,y)))
            next_state = state + 1
            state = next_state
        return [np.arange(self.e.service_num).astype(dtype=np.str), ans]

    def __getK(self, service_index, mec_index):
        D = self.e.services[service_index].data_size
        C = self.e.mecs[mec_index].cpi
        L = ENV_TIME_LIMIT[0]
        return (D * C) / L

    def __getP(self, service_index, mec_index, subcarrier_index):
        D = self.e.services[service_index].data_size
        L = ENV_TIME_LIMIT[1]
        distance = (self.e.mecs[mec_index].location_x - self.e.services[service_index].location_x) ** 2 + (
                self.e.mecs[mec_index].location_y - self.e.services[service_index].location_y) ** 2
        if distance==0:
            distance=0.000001
        UP = (2 ** (D / (L * SUBCARRIER_B)) - 1) * (SUBCARRIER_O ** 2)
        DOWN = (SUBCARRIER_GO * self.e.subcarriers[subcarrier_index].go) / distance
        return UP / DOWN
=======
            reward, end = self.e.forward(state, x, y,
                                self.e.getP(state, x, y),self.e.getK(state, x))
            reward = self.loss(sum(self.e.getPower(state,x,y)) / self.e.services[state].data_size)
            # print('{}:添加到{}号MEC,{}号子信道,传输功率{:.2f},主频{:.2f}'.format(state, x, y, p, k))
            if state % 5 == 0:
                ans.append(sum(self.e.getPower(state,x,y)) / self.e.services[state].data_size)
            next_state = state + 1
            state = next_state
        return range(0, self.e.service_num,5), np.array(ans)

    # def runForAVG(self, cnt=10):
    #     """
    #     运行Q学习
    #     :param cnt: 学习次数
    #     """
    #     for item in range(cnt):
    #         state = self.e.reset()
    #         end = False
    #         while not end:
    #             action = self.select(state)
    #             (x, y) = self.actionTable[action]
    #             (k, p) = (self.__getK(state, x), self.__getP(state, x, y))
    #             reward, end = self.e.forward(state, x, y, p, k)
    #             reward = self.loss(reward / sum(item.data_size for item in self.e.services))
    #             # if item % 100 == 0:
    #             #     print(self.QTable)
    #             next_state = state + 1
    #             self.learn(state, action, reward, next_state, end)
    #             state = next_state

    # def playForAVG(self):
    #     """
    #     学有所成
    #     """
    #     ans = []
    #     self.epsilon = 1.
    #     state = self.e.reset()
    #     end = False
    #     while not end:
    #         action = self.select(state)
    #         (x, y) = self.actionTable[action]
    #         (k, p) = (self.__getK(state, x), self.__getP(state, x, y))
    #         reward, end = self.e.forward(state, x, y, p, k)
    #         reward = self.loss(reward / sum(item.data_size for item in self.e.services))
    #         print('{}:添加到{}号MEC,{}号子信道,传输功率{:.2f},主频{:.2f}'.format(state, x, y, p, k))
    #         ans.append(self.e.POWER / sum(item.data_size for item in self.e.services))
    #         next_state = state + 1
    #         state = next_state
    #     return [np.arange(self.e.service_num).astype(dtype=np.str), np.array(ans)]

>>>>>>> 90a41b2 (修改了一些参数的设定值)
