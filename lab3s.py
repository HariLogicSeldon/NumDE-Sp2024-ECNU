from lab2 import num_ode_lab2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


class num_ode_nDemision():
    def __init__(self, start, end, N, u_start, f, u_true=None):
        self.start = start
        self.end = end
        self.N = N
        self.u_start = u_start
        self.f = f
        self.u_true = u_true

    def point(self):
        return np.linspace(start=self.start, stop=self.end, num=self.N + 1, dtype=np.float64)

    def __prp(self):

        self.t = self.point()
        self.h = self.t[1] - self.t[0]
        self.u_list = np.zeros((self.N + 1,len(self.u_start)))
        self.u_list[0] = self.u_start

    def euler_forward(self):
        self.__prp()
        for i in range(1, self.N + 1):
            up1 = self.u_list[i-1]
            self.u_list[i] = self.u_list[i - 1] + self.h * self.f(up1)
        return self.u_list

    def euler_backward(self,A):
        self.__prp()
        for i in range(1, self.N + 1):
            matrix =np.identity(A.shape[0])
            inv_matrix = np.linalg.inv(matrix - A*self.h)
            self.u_list[i] = np.matmul(inv_matrix, self.u_list[i-1])
        return self.u_list


def try_1():
    """
    测试样例 1
    :return:
    """
    def true_u(t):
        return np.exp(-20 * t)


    def f(x):
        t, u = x[0], x[1]
        return -20 * u


    def get_N(h, s, e):
        return float(e - s) / h


    start, end, u_start = 0, 2, 1

    # 按照步长测试
    h = [0.1*np.power(1/2.,i)for i in range(6)]
    N = [int(get_N(h[i], start, end)) for i in range(len(h))]
    # h = np.array([1 * np.power(0.5, i) for i in range(6)])
    # N = ((end - start) / h).astype(int)
    t_true = np.linspace(start, end, 1000)
    u_true = true_u(t_true)


    u_error = []
    for n in range(len(h)):
        # 构造算法类
        alg = num_ode_lab2(start, end, N[n], u_start, f, true_u)
        t_grid = alg.iter_point()  # 获得格点
        u_euler = alg.runge_kutta_4degree()  # 获得计算结果
        u_error.append(alg.error()[0])
        plt.scatter(t_grid, u_euler, label=N[n], s=50 - n * 9, alpha=0.5)

        if n > 0:
            e = np.log(u_error[n] / u_error[n - 1])
            order = e / np.log(2)
        else:
            order = 0
        print("step final error:%.5f sum error:%.5f order:%.2f" % (*alg.error(), np.abs(order)))
        u_error_all = alg.error_bypoint()
        print(u_error_all)
        plt.plot(t_true, u_true, label="U True")
        plt.legend()
        plt.show()

def try_2():
    A = np.array([[-0.1, -49.9, 0],
                  [0, -50, 0],
                  [0, 70, -30000]], dtype=np.float64)

    def u(t):
        return np.array([
            np.exp(-0.1 * t) + np.exp(-50 * t),
            np.exp(-50 * t),
            0.002337 * np.exp(-50 * t) + 1.997664 * np.exp(-3000 * t),
        ])

    def f(u):
        return np.matmul(A, u)

    start, end,  = 0, 50
    u_start = np.array([2.,1.,2.],dtype=np.float64)
    # step_sizes = np.array([1,1.5])
    step_sizes = np.array([0.1*np.power(1/2.,i)for i in range(3)])
    num_intervals = ((end - start) / step_sizes).astype(int)

    u_error = []
    t_fixed = np.linspace(start, end, 1000)
    u_true = u(t_fixed)

    name = ["Slow Decay","Slow Decay and Small Change","Fast Decay"]
    figure,axs = plt.subplots(nrows=3, ncols=1)

    for i in range(len(u_start)):
        axs[i].grid(alpha=0.2)
        axs[i].plot(t_fixed, u_true[i],label = "True Solution",alpha=0.5)
        axs[i].set_title(name[i])
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)

    color_map = mpl.colormaps['tab10']  # 颜色映射，可以选择不同的映射
    for n in range(len(num_intervals)):
        alg = num_ode_nDemision(start, end, num_intervals[n], u_start, f, u_true)
        t = alg.point()
        u_euler = alg.euler_backward(A).T
        for i in range(len(u_start)):
            axs[i].scatter(t, u_euler[i], s=4 + n * 3, alpha=0.5, color=color_map(n / len(num_intervals)),label = step_sizes[n])
            axs[i].set_title(name[i])
            axs[i].legend()
        u_error.append(u_true[0][-1] - u_euler[0][-1])
        if n > 0:
            e = np.log(u_error[n] / u_error[n - 1])
            order = e / np.log(2)
        else:
            order = 0
        print("order:%.2f" %np.abs(order))

    plt.show(dpi = 500)

if __name__ == '__main__':
    try_1()
    try_2()