from env import Environment
import numpy as np


class RandomSelect:
    """
    随机化选择算法
    """

    def __init__(self, e: Environment):
        self.e = e

    def __select(self):
        mec_index = np.random.randint(0, self.e.mec_num)
        subcarrier_index = np.random.randint(0, self.e.subcarrier_num)
        p = np.random.uniform(0, 10)
        k = np.random.uniform(1.0 * (10 ** 9), 3.0 * (10 ** 9))
        return mec_index, subcarrier_index, p, k

    def run(self):
        """
        执行
        :return: [服务号,总能量消耗]
        """
        ls = []
        for i in range(self.e.service_num):
            mec_index, subcarrier_index, p, k = self.__select()
            while (not self.e.add(i, mec_index, subcarrier_index, p, k)):
                mec_index, subcarrier_index, p, k = self.__select()
            print("选择{}号mec,{}号子信道,传输功率{:.2f},主频{:.2f}ghz".format(
                mec_index, subcarrier_index, p, k / (10 ** 9)))
            ls.append(self.e.POWER)
        return [np.arange(self.e.service_num).astype(dtype=np.str), ls]

    def runForAVG(self):
        """
        执行
        :return: [服务号,总能量消耗]
        """
        ls = []
        for i in range(self.e.service_num):
            mec_index, subcarrier_index, p, k = self.__select()
            while (False == self.e.add(i, mec_index, subcarrier_index, p, k)):
                mec_index, subcarrier_index, p, k = self.__select()
            ls.append(self.e.POWER / sum([self.e.services[j].data_size for j in range(0, i + 1)]))
            print("平均每bit能耗为", ls[-1])
        return [np.arange(self.e.service_num).astype(dtype=np.str), ls]

    def runForTime(self):
        """
        执行
        :return: [服务号,总能量消耗]
        """
        ls = []
        for i in range(self.e.service_num):
            mec_index, subcarrier_index, p, k = self.__select()
            while (not self.e.add(i, mec_index, subcarrier_index, p, k)):
                mec_index, subcarrier_index, p, k = self.__select()
            print("选择{}号mec,{}号子信道,传输功率{:.2f},主频{:.2f}ghz".format(
                mec_index, subcarrier_index, p, k / (10 ** 9)))
            ls.append(sum(self.e.getTime(i, mec_index, subcarrier_index)))
        return [np.arange(self.e.service_num).astype(dtype=np.str), ls]
