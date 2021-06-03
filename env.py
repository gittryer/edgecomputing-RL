import csv
import numpy as np
import math
from typing import List, Dict

'Service的数据配置'
# 数据大小(单位:bit)
SERVICE_DATA_SIZE_RANGE = [1000, 2000]
# 位置坐标范围(单位:m)
SERVICE_LOCATION_RANGE = [0, 100]
'Subcarrier的数据配置'
# 1m的信道增益(单位:db)
SUBCARRIER_go_RANGE = [1.42 * (10 ** -4), 1.42 * (10 ** -3)]
# 信道增益系数
SUBCARRIER_GO = 2.28
# 加性高斯白噪声(单位:w)
SUBCARRIER_O = 1.0 * (10 ** (-6))
# 信道带宽(单位:hz)
SUBCARRIER_B = 1.0 * (10 ** 6)
'MEC有关的数据配置'
# CPU系数
MEC_K = 1
# 位置坐标范围(单位:m)
MEC_LOCATION_RANGE = [0, 100]
# CPI(单位:个)
MEC_CPI_RANGE = [1.0 * (10 ** 3), 5.0 * (10 ** 3)]
# 时延约束(传播时延和运行时延,单位:s)
ENV_TIME_LIMIT = [0.005, 0.005]

'''
*********************************************
@Copyright 小茜的魔法巧克力工厂 2021
csharper@yeah.net
NEU 2021.5
                   _ooOoo_
                  o8888888o
                  88" . "88
                  (| -_- |)
                  O\  =  /O
               ____/`---'\____
             .'  \\|     |//  `.
            /  \\|||  :  |||//  \
           /  _||||| -:- |||||-  \
           |   | \\\  -  /// |   |
           | \_|  ''\---/''  |   |
           \  .-\__  `-`  ___/-. /
         ___`. .'  /--.--\  `. . __
      ."" '<  `.___\_<|>_/___.'  >'"".
     | | :  `- \`.;`\ _ /`;.`/ - ` : | |
     \  \ `-.   \_ __\ /__ _/   .-` /  /
======`-.____`-.___\_____/___.-`____.-'======
                   `=---='
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           佛祖保佑       永无BUG
'''


class Service:
    """
    服务
    """

    def __init__(self, index, data_size, location_x, location_y):
        """
        初始化服务
        :param index: 索引
        :param data_size: 数据大小
        :param location_x: x位置
        :param location_y: y位置
        """
        self.index = index
        self.data_size = data_size
        self.location_x = location_x
        self.location_y = location_y
        # 传输功率需要算法来设定
        self.P = None

    def __str__(self):
        """
        返回字符串
        :return: 返回字符串
        """
        return "服务编号{},数据大小{}(bit),位置({},{}),传输功率{}(w)".format(self.index, self.data_size, self.location_x,
                                                               self.location_y, self.P)

    @staticmethod
    def build(name='service.csv', cnt=100):
        """
        生成csv数据集文件
        :param name: 文件名称
        :param cnt: 循环次数
        """
        with open(name, 'w', newline="") as f:
            writer = csv.writer(f)
            for i in range(cnt):
                ls = []
                data_size = np.random.randint(SERVICE_DATA_SIZE_RANGE[0], SERVICE_DATA_SIZE_RANGE[1])
                location_x, location_y = np.random.randint(SERVICE_LOCATION_RANGE[0], SERVICE_LOCATION_RANGE[1]), \
                                         np.random.randint(SERVICE_LOCATION_RANGE[0], SERVICE_LOCATION_RANGE[1])
                ls.append(i)
                ls.append(data_size)
                ls.append(location_x)
                ls.append(location_y)
                writer.writerow(ls)
            f.close()

    @staticmethod
    def read(count,name='service.csv'):
        """
        从文件中读取服务
        :param name: 文件名
        :return: 返回读取完成的服务
        """
        ans = []
        c=0
        with open(name) as f:
            reader = csv.reader(f)
            for i, rows in enumerate(reader):
                if c<count:
                    ls = rows
                    ans.append(Service(ls[0], int(ls[1]), int(ls[2]), int(ls[3])))
                    c+=1
                else:
                    return ans
            return ans

    def setP(self, P):
        """
        设置传输功率
        :param P: 传输功率
        """
        self.P = P


class SubCarrier:
    """
    子信道
    """

    def __init__(self, index, go):
        """
        构造器
        :param index: 索引
        :param go: 1m的信道增益
        """
        self.index = index
        self.go = go

    @staticmethod
    def build(name='subcarrier.csv', cnt=100):
        """
        生成csv数据集文件
        :param name: 文件名称
        :param cnt: 循环次数
        """
        with open(name, 'w', newline="") as f:
            writer = csv.writer(f)
            for i in range(cnt):
                ls = []
                go = np.random.uniform(SUBCARRIER_go_RANGE[0], SUBCARRIER_go_RANGE[1])
                ls.append(i)
                ls.append(go)
                writer.writerow(ls)
            f.close()

    @staticmethod
    def read(name='subcarrier.csv', count=5):
        """
        从文件中读取子信道
        :param name: 文件名
        :return: 返回读取完成的服务
        """
        ans = []
        with open(name) as f:
            reader = csv.reader(f)
            for i, rows in enumerate(reader):
                if i < count:
                    ls = rows
                    ans.append(SubCarrier(ls[0], float(ls[1])))
        return ans

    def __str__(self):
        """
        返回字符串
        :return: 字符串
        """
        return "信道编号{},1m的信道增益{}(db)".format(self.index, self.go)


class MEC:
    """
    边缘服务器
    """

    def __init__(self, index, cpi, location_x, location_y):
        # 索引
        self.index = index
        # 每条指令所需要的时钟周期数
        self.cpi = cpi
        # 坐标位置
        self.location_x = location_x
        self.location_y = location_y
        # 主频
        self.K = None

    def __str__(self):
        """
        转换为字符串
        :return: 返回字符串
        """
        return "MEC编号{},cpi{}(个),主频{}(hz),位置({},{})".format(self.index, self.cpi, self.K, self.location_x,
                                                            self.location_y)

    @staticmethod
    def build(name='mec.csv', cnt=100):
        """
        生成csv数据集文件
        :param name: 文件名称
        :param cnt: 循环次数
        """
        with open(name, 'w', newline="") as f:
            writer = csv.writer(f)
            for i in range(cnt):
                ls = []
                index = i
                location_x, location_y = np.random.randint(MEC_LOCATION_RANGE[0], MEC_LOCATION_RANGE[1]), \
                                         np.random.randint(MEC_LOCATION_RANGE[0], MEC_LOCATION_RANGE[1])
                cpi = np.random.randint(MEC_CPI_RANGE[0], MEC_CPI_RANGE[1])
                ls.append(index)
                ls.append(cpi)
                ls.append(location_x)
                ls.append(location_y)
                writer.writerow(ls)
            f.close()

    @staticmethod
    def read(name='mec.csv', count=100):
        """
        从文件中读取子信道
        :param name: 文件名
        :return: 返回读取完成的服务
        """
        ans = []
        with open(name) as f:
            reader = csv.reader(f)
            for i, rows in enumerate(reader):
                if i < count:
                    ls = rows
                    index = ls[0]
                    cpi = int(ls[1])
                    location_x = int(ls[2])
                    location_y = int(ls[3])
                    ans.append(MEC(index=index, location_x=location_x, location_y=location_y, cpi=cpi))
        return ans

    def setK(self, K):
        """
        设置主频
        :param K: 主频
        """
        self.K = K


class Environment:
    """
    环境
    """
    # 能量
    POWER = 0
    # 单个MEC最大服务数量
    SINGLE_MEC_MAXSERVICE = 0

    def __init__(self, service_num, mec_num, subcarrier_num):
        """
        创建环境
        :param service_num: 服务的数量
        :param subcarrier_num: 子信道的数量
        :param mec_num: mec的数量
        """
        self.service_num = service_num
        self.subcarrier_num = subcarrier_num
        self.mec_num = mec_num
        # 服务集
        self.services = Service.read(count=self.service_num)
        # 子信道集
        self.subcarriers = SubCarrier.read(count=self.subcarrier_num)
        # mec集
        self.mecs = MEC.read(count=self.mec_num)
        # MEC分配集(长度:1*service_num,范围:[0,mec_num])
        self.u = np.full((1, self.service_num), -1, dtype=np.int32)[0]
        # 信道分配集(长度:1*service_num,范围:[0,subcarrier_num])
        self.w = np.full((1, self.service_num), -1, dtype=np.int32)[0]

    def reset(self):
        """
        重置环境
        :return:
        """
        self = self.__init__(self.service_num, self.mec_num, self.subcarrier_num)
        Environment.POWER = 0
        return 0

    def add(self, service_index, mec_index, subcarrier_index, P, K):
        """
        分配mec，子信道，传输功率，主频
        :param service_index: 服务编号
        :param mec_index: mec编号
        :param subcarrier_index: 子信道编号
        :param P: 传输功率
        :param W: 主频
        :return:
        """
        self.u[service_index] = mec_index
        self.w[service_index] = subcarrier_index
        self.services[service_index].setP(P)
        self.mecs[mec_index].setK(K)
        if sum(self.getTime(service_index, mec_index, subcarrier_index)) < sum(ENV_TIME_LIMIT):
            Environment.POWER += self.getPower(service_index, mec_index, subcarrier_index)
            return True
        else:
            self.u[service_index] = -1
            self.w[service_index] = -1
            self.services[service_index].setP(None)
            self.mecs[mec_index].setK(None)
            return False

    def forward(self,service_index, mec_index, subcarrier_index, P, K):
        self.u[service_index] = mec_index
        self.w[service_index] = subcarrier_index
        self.services[service_index].setP(P)
        self.mecs[mec_index].setK(K)
        Environment.POWER += self.getPower(service_index, mec_index, subcarrier_index)
        return -Environment.POWER,service_index+1==self.service_num

    def getTime(self, service_index, mec_index, subcarrier_index):
        """
        获取某个服务的运行时间
        :param service_index:服务编号
        :param mec_index: mec编号
        :param subcarrier_index: 子信道编号
        :param P: 传输功率(w)
        :param K: 主频(hz)
        :return: 返回时间(s)
        """
        # 数据量
        D = self.services[service_index].data_size
        # CPI
        CPI = self.mecs[mec_index].cpi
        # 主频
        K = self.mecs[mec_index].K
        # MEC运行时延
        T_mec = (D * CPI) / K
        distance = (self.mecs[mec_index].location_x - self.services[service_index].location_x) ** 2 + (
                self.mecs[mec_index].location_y - self.services[service_index].location_y) ** 2
        if distance==0:
            distance=0.000001
        G = (SUBCARRIER_GO * self.subcarriers[subcarrier_index].go) / distance
        # 传输功率
        P = self.services[service_index].P
        # MEC传输时延
        T_trans = D / (SUBCARRIER_B *(math.log(1 + (G * P) / (SUBCARRIER_O ** 2),2)))
        return T_mec, T_trans

    def getPower(self, service_index, mec_index, subcarrier_index):
        """
        获取某一个服务的能量消耗
        :param service_index: 服务编号
        :param mec_index: mec编号
        :param subcarrier_index: 子信道编号
        :return:返回某个服务的功率
        """
        E_mec = MEC_K * (self.mecs[mec_index].K ** 2) * self.services[service_index].data_size * self.mecs[
            mec_index].cpi
        E_trans = self.getTime(service_index, mec_index, subcarrier_index)[1] * self.services[service_index].P
        return E_mec + E_trans