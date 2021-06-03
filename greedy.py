from env import Environment
import numpy as np


class MyGreedy:

    def __init__(self, e: Environment):
        self.e = e
        self.mec_per = [0 for i in range(self.e.mec_num)]
        self.mec_max = self.e.service_num / self.e.mec_num

    def __getMin(self, i):
        dict={}
        for j in range(self.e.mec_num):
            dis = (self.e.mecs[j].location_x - self.e.services[i].location_x) ** 2 + (
                    self.e.mecs[j].location_y - self.e.services[i].location_y) ** 2
            dict[j]=dis
        dict=sorted(dict.items(),key=lambda x:x[1],reverse=False)
        for item in dict:
            if self.mec_per[item[0]]<self.mec_max:
                return item[0]
        return -1
    def __select(self, i):
        mec_index = self.__getMin(i)
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
            mec_index, subcarrier_index, p, k = self.__select(i)
            while (not self.e.add(i, mec_index, subcarrier_index, p, k)):
                mec_index, subcarrier_index, p, k = self.__select(i)
            print("选择{}号mec,{}号子信道,传输功率{:.2f},主频{:.2f}ghz".format(
                mec_index, subcarrier_index, p, k / (10 ** 9)))
            self.mec_per[mec_index]+=1
            ls.append(self.e.POWER)
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
            self.mec_per[mec_index]+=1
            ls.append(self.e.POWER/sum([self.e.services[j].data_size for j in range(0,i+1)]))
            print("平均每bit能耗为", ls[-1])
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
            self.mec_per[mec_index]+=1
            ls.append(sum(self.e.getTime(i,mec_index,subcarrier_index)))
        return [np.arange(self.e.service_num).astype(dtype=np.str), ls]