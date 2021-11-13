from env import Environment
import numpy as np


class MyGreedy:

    def __init__(self, e: Environment):
        self.e = e
<<<<<<< HEAD
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
=======
        # 服务上传地距离每一个MEC的距离表
        self.dis: {int: float} = {}
        # self.mec_per = [0 for i in range(self.e.mec_num)]
        # self.mec_max = self.e.service_num / self.e.mec_num

    def __updateDis(self, i):
        """
        更新距离表
        """
        self.dis={}
        # 更新
        for j in range(self.e.mec_num):
            dis = (self.e.mecs[j].location_x - self.e.services[i].location_x) ** 2 + (
                    self.e.mecs[j].location_y - self.e.services[i].location_y) ** 2
            self.dis[j] = dis
        # 排序
        self.dis = sorted(self.dis.items(), key=lambda x: x[1], reverse=False)

    def __getMin(self, i):
        """
        选择最近的MEC服务器
        :param i:
        :return:
        """
        return self.dis[0][0]

    def __select(self, i):
        """
        选择MEC服务器
        :param i:
        """
        # 选择mec和子信道
        mec_index = self.__getMin(i)
        subcarrier_index = np.random.randint(0, self.e.subcarrier_num)
        # 随机分配传输功率和MEC频率
>>>>>>> 90a41b2 (修改了一些参数的设定值)
        p = np.random.uniform(0, 10)
        k = np.random.uniform(1.0 * (10 ** 9), 3.0 * (10 ** 9))
        return mec_index, subcarrier_index, p, k

    def run(self):
        """
        执行
        :return: [服务号,总能量消耗]
        """
<<<<<<< HEAD
        ls = []
        for i in range(self.e.service_num):
            mec_index, subcarrier_index, p, k = self.__select(i)
=======
        #记录总的能量消耗变化情况
        ls = []
        for i in range(self.e.service_num):
            self.__updateDis(i)
            mec_index, subcarrier_index, p, k = self.__select(i)
            # # 不满足时延要求
>>>>>>> 90a41b2 (修改了一些参数的设定值)
            while (not self.e.add(i, mec_index, subcarrier_index, p, k)):
                mec_index, subcarrier_index, p, k = self.__select(i)
            print("选择{}号mec,{}号子信道,传输功率{:.2f},主频{:.2f}ghz".format(
                mec_index, subcarrier_index, p, k / (10 ** 9)))
<<<<<<< HEAD
            self.mec_per[mec_index]+=1
            ls.append(self.e.POWER)
        return [np.arange(self.e.service_num).astype(dtype=np.str), ls]
=======
            # self.mec_per[mec_index]+=1
            if i %5 ==0:
                ls.append(self.e.POWER)
        return range(0, self.e.service_num), ls
        #return range(0,self.e.service_num,5), np.array(ls)
>>>>>>> 90a41b2 (修改了一些参数的设定值)

    def runForAVG(self):
        """
        执行
        :return: [服务号,总能量消耗]
        """
<<<<<<< HEAD
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
=======
        # 记录总的能量消耗变化情况
        ls = []
        for i in range(self.e.service_num):
            self.__updateDis(i)
            mec_index, subcarrier_index, p, k = self.__select(i)
            # # 不满足时延要求
>>>>>>> 90a41b2 (修改了一些参数的设定值)
            while (not self.e.add(i, mec_index, subcarrier_index, p, k)):
                mec_index, subcarrier_index, p, k = self.__select(i)
            print("选择{}号mec,{}号子信道,传输功率{:.2f},主频{:.2f}ghz".format(
                mec_index, subcarrier_index, p, k / (10 ** 9)))
<<<<<<< HEAD
            self.mec_per[mec_index]+=1
            ls.append(sum(self.e.getTime(i,mec_index,subcarrier_index)))
        return [np.arange(self.e.service_num).astype(dtype=np.str), ls]
=======
            # self.mec_per[mec_index]+=1
            if i % 5 == 0:
                ls.append(sum(self.e.getPower(i,mec_index,subcarrier_index)) / self.e.services[i].data_size)
        return range(0, self.e.service_num,5), np.array(ls)

    # def runForAVG(self):
    #     """
    #     执行
    #     :return: [服务号,总能量消耗]
    #     """
    #     ls = []
    #     for i in range(self.e.service_num):
    #         mec_index, subcarrier_index, p, k = self.__select(i)
    #         while (not self.e.add(i, mec_index, subcarrier_index, p, k)):
    #             mec_index, subcarrier_index, p, k = self.__select(i)
    #         self.mec_per[mec_index]+=1
    #         ls.append(self.e.POWER/sum([self.e.services[j].data_size for j in range(0,i+1)]))
    #         print("平均每bit能耗为", ls[-1])
    #     return [np.arange(self.e.service_num).astype(dtype=np.str), ls]
    #
    #
    # def runForTime(self):
    #     """
    #     执行
    #     :return: [服务号,总能量消耗]
    #     """
    #     ls = []
    #     for i in range(self.e.service_num):
    #         mec_index, subcarrier_index, p, k = self.__select(i)
    #         while (not self.e.add(i, mec_index, subcarrier_index, p, k)):
    #             mec_index, subcarrier_index, p, k = self.__select(i)
    #         print("选择{}号mec,{}号子信道,传输功率{:.2f},主频{:.2f}ghz".format(
    #             mec_index, subcarrier_index, p, k / (10 ** 9)))
    #         self.mec_per[mec_index]+=1
    #         ls.append(sum(self.e.getTime(i,mec_index,subcarrier_index)))
    #     return [np.arange(self.e.service_num).astype(dtype=np.str), ls]
>>>>>>> 90a41b2 (修改了一些参数的设定值)
