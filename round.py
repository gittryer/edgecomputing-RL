from env import Environment
from env import ENV_TIME_LIMIT, SUBCARRIER_B, SUBCARRIER_O, SUBCARRIER_GO
import numpy as np


# 轮询
class RoundRobin:
    def __init__(self, e: Environment):
        """
        构造函数
        :param service_num: service的数量
        :param env: 环境
        """
        self.e = e
        self.id_mec = 0
        self.id_subcarrier = 0

    def __select(self, i):
        """
        选择mec编号
        :return: 返回mec编号
        """
        self.id_mec += 1
        self.id_subcarrier += 1
        if self.id_mec >= self.e.mec_num:
            self.id_mec = 0
        if self.id_subcarrier >= self.e.subcarrier_num:
            self.id_subcarrier = 0
        p = np.random.uniform(0, 10)
        k = np.random.uniform(1.0 * (10 ** 9), 3.0 * (10 ** 9))
        return self.id_mec, self.id_subcarrier, p, k

    def __getK(self, service_index, mec_index):
        D = self.e.services[service_index].data_size
        C = self.e.mecs[mec_index].cpi
        L = ENV_TIME_LIMIT[0]
        return (D * C) / L

    def __getP(self, service_index, mec_index, subcarrier_index):
        D = self.e.services[service_index].data_size
        L = ENV_TIME_LIMIT[1]
        distance = (self.e.mecs[mec_index].location_x - self.e.services[mec_index].location_x) ** 2 + (
                self.e.mecs[mec_index].location_y - self.e.services[mec_index].location_y) ** 2
        UP = (2 ** (D / (L * SUBCARRIER_B)) - 1) * (SUBCARRIER_O ** 2)
        DOWN = (SUBCARRIER_GO * self.e.subcarriers[subcarrier_index].go) / distance
        return UP / DOWN

    def run(self):
        """
        执行
        :return: [服务号,总能量消耗]
        """
        ls = []
        for i in range(self.e.service_num):
            mec_index, subcarrier_index, p, k = self.__select(i)
            while (not self.e.add(i, mec_index, subcarrier_index, p, k)):
                mec_index, subcarrier_index, p, k = self.__select(i)
            print("选择{}号mec,{}号子信道,传输功率{:.2f},主频{:.2f}ghz".format(
                mec_index, subcarrier_index, p, k / (10 ** 9)))
            ls.append(Environment.POWER)
        return [np.arange(self.e.service_num).astype(dtype=np.str), ls]

    def runForAVG(self):
        """
        执行
        :return: [服务号,总能量消耗]
        """
        ls = []
        for i in range(self.e.service_num):
            mec_index, subcarrier_index, p, k = self.__select(i)
            while (not self.e.add(i, mec_index, subcarrier_index, p, k)):
                mec_index, subcarrier_index, p, k = self.__select(i)
            ls.append(self.e.POWER/sum([self.e.services[j].data_size for j in range(0,i+1)]))
            print("平均每bit能耗为",ls[-1])
        return [np.arange(self.e.service_num).astype(dtype=np.str), ls]

    def runForTime(self):
        """
        执行
        :return: [服务号,总能量消耗]
        """
        ls = []
        for i in range(self.e.service_num):
            mec_index, subcarrier_index, p, k = self.__select(i)
            while (not self.e.add(i, mec_index, subcarrier_index, p, k)):
                mec_index, subcarrier_index, p, k = self.__select(i)
            print("选择{}号mec,{}号子信道,传输功率{:.2f},主频{:.2f}ghz".format(
                mec_index, subcarrier_index, p, k / (10 ** 9)))
            ls.append(sum(self.e.getTime(i,mec_index,subcarrier_index)))
        return [np.arange(self.e.service_num).astype(dtype=np.str), ls]
