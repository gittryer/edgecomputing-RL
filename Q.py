import numpy as np
import pandas as pd
from env import Environment
from env import ENV_TIME_LIMIT, SUBCARRIER_B, SUBCARRIER_O, SUBCARRIER_GO
import math


class QLearning:
    def __init__(self, e: Environment, learning_rate=0.7, gamma=0.65, epsilon=0.55):
        """
        构造器
        :param learning_rate: 学习率
        :param gamma: 衰减率
        :param epsilon: 使用已有资源你的概率
        :param states: 状态数
        :param actions: 行为数(mec数)
        :param env: 环境
        """
        self.e = e
        # 学习率p
        self.learning_rate = learning_rate
        # 衰减率
        self.gamma = gamma
        # 利用已有资源的概率
        self.epsilon = epsilon
        # 状态数
        self.states = self.e.service_num
        # 行为数
        self.actions = self.e.mec_num * self.e.subcarrier_num
        # 行为表(mec编号,子信道编号)
        self.actionTable = []
        for i in range(self.e.mec_num):
            for j in range(self.e.subcarrier_num):
                self.actionTable.append((i, j))
        # 初始化Q表
        self.QTable = pd.DataFrame(data=[[0 for item in range(self.actions)] for item in range(self.states + 1)],
                                   index=range(self.states + 1), columns=range(self.actions))
        pd.set_option('display.max_columns', 1000)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', 1000)

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
                (k, p) = (self.__getK(state, x), self.__getP(state, x, y))
                reward, end = self.e.forward(state, x, y, p, k)
                if item % 100 == 0:
                    print(self.QTable)
                next_state = state + 1
                self.learn(state, action, next_state, reward)
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
            (k, p) = (self.__getK(state, x), self.__getP(state, x, y))
            print('{}:添加到{}号MEC,{}号子信道,传输功率{:.2f},主频{:.2f}'.format(state, x, y, p, k))
            reward, end = self.e.forward(state, x, y, p, k)
            ans.append(-reward)
            next_state = state + 1
            state = next_state
        return [np.arange(self.e.service_num).astype(dtype=np.str), ans]

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
                (k, p) = (self.__getK(state, x), self.__getP(state, x, y))
                reward, end = self.e.forward(state, x, y, p, k)
                reward = reward / sum([self.e.services[j].data_size for j in range(0, state + 1)])
                if item % 100 == 0:
                    print(self.QTable)
                next_state = state + 1
                self.learn(state, action, next_state, reward)
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
