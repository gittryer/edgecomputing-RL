from env import Environment
<<<<<<< HEAD
from rand import RandomSelect
from round import RoundRobin
from greedy import MyGreedy
from Q import QLearning
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def AVG(service_num=100,mec_num=10,subcarrier_num=5):
=======
from Q import QLearning
from round import RoundRobin
from greedy import MyGreedy
from rand import RandomSelect
import matplotlib.pyplot as plt

# global setting
MARKER_SIZE=8


def draw_iteration(service_num=50, mec_num=10, subcarrier_num=5):
    ls = [10, 1000, 1500, 2000, 3000,5000]
    plt.xticks(range(0, service_num, 5))
    e = Environment(service_num, mec_num, subcarrier_num)
    for item in ls:
        e.reset()
        print("=======执行QLearning算法(iteration=%d)========" % item)
        q = QLearning(e)
        q.run(item)
        x, y = q.play()
        plt.plot(x, y, label='iteration=%d' % item, marker='p', markersize=MARKER_SIZE)
    plt.legend()
    plt.xlabel('Services Num')
    plt.ylabel('Energy Consumption')
    plt.savefig('draw_iteration.pdf')
    plt.show()


def draw_LR(service_num=50, mec_num=10, subcarrier_num=5):
    ls = [0.5, 0.6, 0.7, 0.8, 0.9]
    e = Environment(service_num, mec_num, subcarrier_num)
    plt.xticks(range(0,service_num,5))
    for item in ls:
        e.reset()
        print("=======执行QLearning算法(LR=%.1f)========" % item)
        q = QLearning(e, learning_rate=item)
        q.run(1000)
        x, y = q.play()
        plt.plot(x, y, label='learning_rate=%.1f' % item, marker='p', markersize=MARKER_SIZE)
    plt.legend()
    plt.xlabel('Services Num')
    plt.ylabel('Energy Consumption')
    plt.savefig('draw_LR.pdf')
    plt.show()

def draw_loss(service_num=50, mec_num=10, subcarrier_num=5):
    ls = [lambda f: 1 / f, lambda f: -f]
    e = Environment(service_num, mec_num, subcarrier_num)
    plt.xticks(range(0, service_num, 5))
    for i in range(len(ls)):
        e.reset()
        print("=======执行QLearning算法========")
        q = QLearning(e, loss=ls[i])
        q.run(1000)
        x, y = q.play()
        if i == 0:
            plt.plot(x, y, label='loss=1/E', marker='p', markersize=MARKER_SIZE)
        else:
            plt.plot(x, y, label='loss=-E', marker='p', markersize=MARKER_SIZE)
    plt.legend()
    plt.xlabel('Services Num')
    plt.ylabel('Energy Consumption')
    plt.savefig('draw_loss.pdf')
    plt.show()

def draw_iteration_avg(service_num=50, mec_num=10, subcarrier_num=5):
    ls = [10, 100, 1000, 1500, 2000, 3000,5000]
    plt.xticks(range(0, service_num, 5))
    e = Environment(service_num, mec_num, subcarrier_num)
    for item in ls:
        e.reset()
        print("=======执行QLearning算法(iteration=%d)========" % item)
        q = QLearning(e)
        q.runForAVG(item)
        x, y = q.playForAVG()
        plt.plot(x, y, label='iteration=%d' % item, marker='p', markersize=MARKER_SIZE)
    plt.legend()
    plt.xlabel('Services Num')
    plt.ylabel('Energy Consumption/bit')
    plt.savefig('draw_iteration_avg.pdf')
    plt.show()


def draw_LR_avg(service_num=50, mec_num=10, subcarrier_num=5):
    ls = [0.5, 0.6, 0.7, 0.8, 0.9]
    e = Environment(service_num, mec_num, subcarrier_num)
    plt.xticks(range(0,service_num,5))
    for item in ls:
        e.reset()
        print("=======执行QLearning算法(LR=%.1f)========" % item)
        q = QLearning(e, learning_rate=item)
        q.runForAVG(1000)
        x, y = q.playForAVG()
        plt.plot(x, y, label='learning_rate=%.1f' % item, marker='p', markersize=MARKER_SIZE)
    plt.legend()
    plt.xlabel('Services Num')
    plt.ylabel('Energy Consumption/bit')
    plt.savefig('draw_LR_avg.pdf')
    plt.show()

def draw_loss_avg(service_num=100, mec_num=10, subcarrier_num=5):
    ls = [lambda f: 1 / f, lambda f: -f]
    e = Environment(service_num, mec_num, subcarrier_num)
    plt.xticks(range(0, service_num, 5))
    for i in range(len(ls)):
        e.reset()
        print("=======执行QLearning算法========")
        q = QLearning(e, loss=ls[i])
        q.runForAVG(1000)
        x, y = q.playForAVG()
        if i == 0:
            plt.plot(x, y, label='loss=1/f', marker='p', markersize=MARKER_SIZE)
        else:
            plt.plot(x, y, label='loss=-f', marker='p', markersize=MARKER_SIZE)
    plt.legend()
    plt.xlabel('Services Num')
    plt.ylabel('Energy Consumption/bit')
    plt.savefig('draw_loss_avg.pdf')
    plt.show()


MARKER_SIZE=3
def draw_compare(service_num=50, mec_num=20, subcarrier_num=5):
>>>>>>> 90a41b2 (修改了一些参数的设定值)
    """
    生成服务数量-总能量消耗对比图
    :param mec_num: mec数量
    :param service_num: 服务数量
    :param subcarrier_num: 子信道个数
    """
<<<<<<< HEAD
    e = Environment(service_num,mec_num,subcarrier_num)
    plt.gca().xaxis.set_major_locator(MultipleLocator(10))
    print("=======执行随机选择算法========")
    x, y = RandomSelect(e).runForAVG()
    print(x,y)
    plt.plot(x[10:], y[10:], label='Random', marker='*')
    e.reset()
    print("=======执行轮询算法========")
    x, y = RoundRobin(e).runForAVG()
    print(x,y)
    plt.plot(x[10:], y[10:], label='Round Robin', marker='x')
    e.reset()
    print("=======执行贪心算法========")
    x, y = MyGreedy(e).runForAVG()
    print(x, y)
    plt.plot(x[10:], y[10:], label='Greedy', marker='+')
    e.reset()
    print("=======执行QLearning算法========")
    q=QLearning(e)
    q.runForAVG(1200)
    x, y =q.playForAVG()
    print(x, y)
    plt.plot(x[10:], y[10:], label='QLearning',marker='3')
    e.reset()
    plt.xlabel("services num")
    plt.ylabel("energy/bit")
    # plt.title('Services-energy consumption/bit')
    plt.legend()
    plt.savefig("avg.jpg")
    plt.show()
def AVG_MEC(service_num=100,subcarrier_num=5):
    px = []
    py1 = []
    py2 = []
    py3 = []
    py4 = []
    for mec_num in range(10, 100, 10):
        print('$MEC_NUM=', mec_num)
        px.append(str(mec_num))
        e = Environment(service_num, mec_num, subcarrier_num)
        print("=======执行随机选择算法========")
        x, y = RandomSelect(e).runForAVG()
        print(x, y)
        py1.append(y[-1])
        e.reset()
        print("=======执行轮询算法========")
        x, y = RoundRobin(e).runForAVG()
        print(x, y)
        py2.append(y[-1])
        e.reset()
        print("=======执行贪心算法========")
        x, y = MyGreedy(e).runForAVG()
        print(x, y)
        py3.append(y[-1])
        e.reset()
        print("=======执行QLearning算法========")
        q = QLearning(e)
        q.runForAVG(1500)
        x, y = q.playForAVG()
        print(x, y)
        py4.append(y[-1])
        e.reset()
    plt.plot(px, py1, label='RandomSelect', marker='*')
    plt.plot(px, py2, label='RoundRobin', marker='x')
    plt.plot(px, py3, label='Greedy', marker='+')
    plt.plot(px, py4, label='QLearning', marker='3')
    # plt.title("Energy consumption/bit")
    plt.xlabel("MEC num")
    plt.ylabel("energy/bit")
    plt.legend()
    plt.savefig("avg_mec.jpg")
    plt.show()
def test(service_num=100,mec_num=20,subcarrier_num=10):
=======
    e = Environment(service_num, mec_num, subcarrier_num)
    plt.xticks(range(0, service_num, 5))
    print("=======执行随机选择算法========")
    x, y = RandomSelect(e).run()
    print(x, y)
    plt.plot(x, y, label='Random', marker='s', markersize=MARKER_SIZE)
    e.reset()
    print("=======执行贪心选择算法========")
    x, y = MyGreedy(e).run()
    print(x, y)
    plt.plot(x, y, label='Greedy', marker='^', markersize=MARKER_SIZE)
    e.reset()
    print("=======执行轮询选择算法========")
    x, y = RoundRobin(e).run()
    print(x, y)
    plt.plot(x, y, label='RoundRobin', marker='.', markersize=MARKER_SIZE)
    e.reset()
    print("=======执行QLearning算法(iteration=1000)========")
    q = QLearning(e)
    q.run(1000)
    x, y = q.play()
    print(x, y)
    plt.plot(x, y, label='Qlearning', marker='p', markersize=MARKER_SIZE)

    plt.legend()
    plt.xlabel('Services Num')
    plt.ylabel('Energy Consumption')
    plt.savefig('draw_compare.pdf')
    plt.show()

def draw_compare_avg(service_num=50, mec_num=20, subcarrier_num=5):
>>>>>>> 90a41b2 (修改了一些参数的设定值)
    """
    生成服务数量-总能量消耗对比图
    :param mec_num: mec数量
    :param service_num: 服务数量
    :param subcarrier_num: 子信道个数
    """
<<<<<<< HEAD
    e = Environment(service_num,mec_num,subcarrier_num)
    plt.gca().xaxis.set_major_locator(MultipleLocator(10))
    print("=======执行随机选择算法========")
    x, y = RandomSelect(e).run()
    print(x,y)
    plt.plot(x[10:], y[10:], label='Random', marker='*')
    e.reset()
    print("=======执行轮询算法========")
    x, y = RoundRobin(e).run()
    print(x,y)
    plt.plot(x[10:], y[10:], label='Round Robin', marker='x')
    e.reset()
    print("=======执行贪心算法========")
    x, y = MyGreedy(e).run()
    print(x, y)
    plt.plot(x[10:], y[10:], label='Greedy', marker='+')
    e.reset()
    print("=======执行QLearning算法========")
    q=QLearning(e)
    q.run(1500)
    x, y =q.play()
    print(x, y)
    plt.plot(x[10:], y[10:], label='QLearning',marker='3')
    e.reset()
    plt.xlabel("services num")
    plt.ylabel("all energy")
    # plt.title('Services-ALL Energy Consumption')
    plt.legend()
    plt.savefig("test.jpg")
    plt.show()
def test_MEC(service_num=100,subcarrier_num=5):
    px = []
    py1 = []
    py2 = []
    py3 = []
    py4 = []
    for mec_num in range(10, 100, 10):
        print('mec_num=', mec_num)
        px.append(str(mec_num))
        e = Environment(service_num, mec_num, subcarrier_num)
        print("=======执行随机选择算法========")
        x, y = RandomSelect(e).run()
        print(x, y)
        py1.append(y[-1])
        e.reset()
        print("=======执行轮询算法========")
        x, y = RoundRobin(e).run()
        print(x, y)
        py2.append(y[-1])
        e.reset()
        print("=======执行贪心算法========")
        x, y = MyGreedy(e).run()
        print(x, y)
        py3.append(y[-1])
        e.reset()
        print("=======执行QLearning算法========")
        q = QLearning(e)
        q.run(1500)
        x, y = q.play()
        print(x, y)
        py4.append(y[-1])
        e.reset()
    f = open("TEST_MEC.txt", "a+")
    f.writelines(px)
    f.writelines(str(py1))
    f.writelines(str(py2))
    f.writelines(str(py3))
    f.writelines(str(py4))
    f.close()
    plt.plot(px, py1, label='RandomSelect', marker='*')
    plt.plot(px, py2, label='RoundRobin', marker='x')
    plt.plot(px, py3, label='Greedy', marker='+')
    plt.plot(px, py4, label='QLearning', marker='3')
    # plt.title("Energy consumption")
    plt.xlabel("mec num")
    plt.ylabel("energy")
    plt.legend()
    plt.savefig("TEST_MEC.jpg")
    plt.show()
# def testForTime(service_num=100,mec_num=10,subcarrier_num=5):
#     e = Environment(service_num,mec_num,subcarrier_num)
#     plt.xlabel('algorithm')
#     plt.ylabel('average time')
#     px=['RandomSelect','RoundRobin','Greedy','QLearning']
#     py=[]
#     print("=======执行随机选择算法========")
#     x, y = RandomSelect(e).runForTime()
#     print(x,y)
#     py.append(y[-1]/float(x[-1]))
#     e.reset()
#     print("=======执行轮询算法========")
#     x, y = RoundRobin(e).runForTime()
#     print(x,y)
#     py.append(y[-1]/float(x[-1]))
#     e.reset()
#     print("=======执行贪心算法========")
#     x, y = MyGreedy(e).runForTime()
#     print(x, y)
#     py.append(y[-1]/float(x[-1]))
#     e.reset()
#     print("=======执行QLearning算法========")
#     q = QLearning(e)
#     q.runForTime(1500)
#     x, y = q.playForTime()
#     print(x, y)
#     py.append(y[-1]/float(x[-1]))
#     e.reset()
#     plt.bar(px,py)
#     plt.savefig("time.jpg")
#     plt.show()

AVG_MEC()
=======
    e = Environment(service_num, mec_num, subcarrier_num)
    plt.xticks(range(0, service_num, 5))
    print("=======执行随机选择算法========")
    x, y = RandomSelect(e).runForAVG()
    print(x, y)
    plt.plot(x, y, label='Random', marker='s', markersize=MARKER_SIZE)
    e.reset()
    print("=======执行贪心选择算法========")
    x, y = MyGreedy(e).runForAVG()
    print(x, y)
    plt.plot(x, y, label='Greedy', marker='^', markersize=MARKER_SIZE)
    e.reset()
    print("=======执行轮询选择算法========")
    x, y = RoundRobin(e).runForAVG()
    print(x, y)
    plt.plot(x, y, label='RoundRobin', marker='.', markersize=MARKER_SIZE)
    e.reset()
    print("=======执行QLearning算法(iteration=1000)========")
    q = QLearning(e)
    q.runForAVG(1000)
    x, y = q.playForAVG()
    print(x, y)
    plt.plot(x, y, label='Qlearning', marker='p', markersize=MARKER_SIZE)

    plt.legend()
    plt.xlabel('Services Num')
    plt.ylabel('Energy Consumption/bit')
    plt.savefig('draw_compare_avg.pdf')
    plt.show()


def draw_compare_mec(service_num=50, subcarrier_num=10):
    lx = range(10, 80, 10)
    ly1 = []
    ly2=[]
    ly3=[]
    ly4=[]
    for item in lx:
        e = Environment(service_num, item, subcarrier_num)
        print("=======执行随机选择算法========")
        x, y = RandomSelect(e).run()
        ly1.append(y[-1])
        e.reset()
        print("=======执行贪心选择算法========")
        x, y = MyGreedy(e).run()
        ly2.append(y[-1])
        e.reset()
        print("=======执行轮询选择算法========")
        x, y = RoundRobin(e).run()
        ly3.append(y[-1])
        e.reset()
        print("=======执行QLearning算法(iteration=1000)========")
        q = QLearning(e)
        q.run(1000)
        x, y = q.play()
        ly4.append(y[-1])
    plt.plot(lx, ly1, label='Random', marker='s', markersize=MARKER_SIZE)
    plt.plot(lx, ly2, label='Greedy', marker='^', markersize=MARKER_SIZE)
    plt.plot(lx, ly3, label='RoundRobin', marker='.', markersize=MARKER_SIZE)
    plt.plot(lx, ly4, label='Qlearning', marker='p', markersize=MARKER_SIZE)
    plt.legend()
    plt.xlabel('Num of MEC Servers')
    plt.ylabel('Energy Consumption')
    plt.savefig('draw_compare_mec.pdf')
    plt.show()
def draw_compare_mec_avg(service_num=50, subcarrier_num=10):
    lx = range(10, 80, 10)
    ly1 = []
    ly2=[]
    ly3=[]
    ly4=[]
    for item in lx:
        e = Environment(service_num, item, subcarrier_num)
        print("=======执行随机选择算法========")
        x, y = RandomSelect(e).runForAVG()
        ly1.append(y[-1])
        e.reset()
        print("=======执行贪心选择算法========")
        x, y = MyGreedy(e).runForAVG()
        ly2.append(y[-1])
        e.reset()
        print("=======执行轮询选择算法========")
        x, y = RoundRobin(e).runForAVG()
        ly3.append(y[-1])
        e.reset()
        print("=======执行QLearning算法(iteration=1000)========")
        q = QLearning(e)
        q.runForAVG(1000)
        x, y = q.playForAVG()
        ly4.append(y[-1])
    plt.plot(lx, ly1, label='Random', marker='s', markersize=MARKER_SIZE)
    plt.plot(lx, ly2, label='Greedy', marker='^', markersize=MARKER_SIZE)
    plt.plot(lx, ly3, label='RoundRobin', marker='.', markersize=MARKER_SIZE)
    plt.plot(lx, ly4, label='Qlearning', marker='p', markersize=MARKER_SIZE)
    plt.legend()
    plt.xlabel('Num of MEC Servers')
    plt.ylabel('Energy Consumption/bit')
    plt.savefig('draw_compare_mec_avg.pdf')
    plt.show()


if __name__ == '__main__':
    # draw_compare()
    draw_loss()
    # draw_compare_avg()
    # draw_loss()
    # draw_loss_avg()
    # draw_compare_mec_avg()
    print("end")
>>>>>>> 90a41b2 (修改了一些参数的设定值)
