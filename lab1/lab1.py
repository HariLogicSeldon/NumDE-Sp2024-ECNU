import matplotlib.pyplot as plt
from grad_F import grad_F
import numpy as np
class num_ode:
    def __init__(self,start,end,N,u_start,f,u_true=None):
        """

        :param start:起始点
        :param end:重点
        :param N:迭代次数
        :param u_start:终点
        :param f:ODE 右边项
        :param u_true:真实解
        """

        self.start = start
        self.end = end
        self.N = N
        self.u_start = u_start
        self.func = f
        self.utrue = u_true

    def iter_point(self):
        return np.linspace(start=self.start,stop=self.end,num=self.N + 1,dtype=np.float64)

    def output(self):
        print("N:%d"%self.N)

    def prp(self):
        """
        :return: 
        t:迭代点列
        h:迭代步长
        u_list:长度为迭代次数加一的最终输出结果存储列
        """""
        self.t = self.iter_point()
        self.h = t[1] - t[0]
        self.u_list = np.zeros(self.N + 1)
        self.u_list[0] = 1

    def error(self):
        true_value = self.utrue(self.t)
        optimal_value = self.u_list
        error_list = np.abs(true_value - optimal_value)
        return error_list[-1], error_list.sum()

    def euler(self):
        self.prp()
        for i in range(1,self.N+1):
            tu = np.array([self.t[i - 1], self.u_list[i - 1]])
            self.u_list[i] = self.u_list[i-1] + self.h * self.func(tu)
        self.output()
        return self.u_list

    def imp_euler(self):
        self.prp()
        for i in range(1,self.N+1):
            tu = np.array([self.t[i - 1], self.u_list[i - 1]])
            u_n = self.func(tu)
            u_n_step = self.func(np.array([self.t[i - 1] + self.h, u_n]))
            integral = (u_n + u_n_step) * 0.5
            self.u_list[i] = self.u_list[i - 1] + self.h * integral
        self.output()
        return self.u_list

    def taylor_3degree(self):
        self.prp()
        for i in range(1, self.N + 1):
            tu = np.array([self.t[i - 1], self.u_list[i - 1]])
            grad_f = grad_F(self.func)
            du = self.func(tu)
            ddu = grad_f.f_t(tu) + grad_f.f_u(tu) * du
            dddu = grad_f.f_tt(tu) + grad_f.f_tu(tu) * du + (grad_f.f_ut(tu)
                                                             + grad_f.f_uu(tu) * du) *du + grad_f.f_u(tu)*ddu

            self.u_list[i] = self.u_list[i - 1] + self.h * du + (self.h **2) * 0.5 *ddu + (self.h ** 3) * (1/6) *dddu
        self.output()
        return self.u_list

    def two_step(self,a):
        self.prp()
        self.u_list[1] = self.utrue(self.t[1])
        for i in range(2, self.N + 1):
            tu_1 = np.array([self.t[i - 1], self.u_list[i - 1]])
            tu_2 = np.array([self.t[i - 2], self.u_list[i - 2]])
            fpart = (3 - a) * self.func(tu_1) - (1 + a) * self.func(tu_2)
            self.u_list[i] = (1+a) * self.u_list[i-1] - a * self.u_list[i-2] +\
                             0.5 * self.h * fpart
        self.output()
        return self.u_list

    def admas_out(self):
        self.prp()

        for i in range(1, self.N + 1):
            tu = np.array([self.t[i - 1], self.u_list[i - 1]])
            self.u_list[i] = self.u_list[i - 1] + self.h * self.func(tu)
        self.output()
        return self.u_list



if __name__ == "__main__":
    def true_u(t):
        return (1. + t ** 2.) ** 2.

    def f(x):
        t, u = x[0], x[1]
        return 4. * t * (np.power(u,1/2))


    start = 0
    end = 2
    h = np.array([0.1, 0.05, 0.025, 0.0125, 0.00625])
    N = 2 / h
    N = N.astype(int)
    u_start = 1


    u_error = []
    t_list = []
    for n in range(3):
        alg = num_ode(start, end, N[n], u_start, f, true_u)
        t = alg.iter_point()
        u_euler = alg.imp_euler()

        t_list.append(t)

        u_error.append(alg.error()[0])
        plt.scatter(t, u_euler,label = N[n],s = 50 + n*3,alpha=0.5)

        if n >0:
            e = np.log(u_error[n]/u_error[n-1])
            order = e/np.log(2)
        else:
            order = 0
        # print(u_euler)
        print("step final error:%.5f sum error:%.5f order:%.2f" %(*alg.error(),np.abs(order)))

    u_true = true_u(t_list[-1])
    print(u_error)
    plt.plot(t_list[-1], u_true,label="U True")
    plt.legend()
    plt.show()






