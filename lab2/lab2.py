import matplotlib.pyplot as plt
import numpy as np

from lab1 import num_ode_lab1


class num_ode_lab2(num_ode_lab1):
    def adams_out(self):
        print("Adams-Bashforth")
        self.prp()

        alg_v1 = num_ode_lab1(self.start,self.start+2*self.h,2,self.u_start,self.func,u_true=self.utrue)
        outcome_v1 = alg_v1.imp_euler()
        for j in range(1, 3):
            self.u_list[j] = outcome_v1[j]

        for i in range(3, self.N + 1):
            tu1, tu2, tu3 = (np.array([self.t[i - 1], self.u_list[i - 1]]),
                             np.array([self.t[i - 2], self.u_list[i - 2]]),
                             np.array([self.t[i - 3], self.u_list[i - 3]]))

            integral = (self.h / 12) * (
                    23 * self.func(tu1)
                    - 16 * self.func(tu2)
                    + 5 * self.func(tu3) )
            self.u_list[i] = self.u_list[i - 1] + integral
        self.output()

        return self.u_list

    def adams_in(self,method = "Error_Correction"):
        print("Adams-Moulton",method)
        self.prp()
        alg_v1 = num_ode_lab1(self.start, self.start + 2 * self.h, 2, self.u_start, self.func,u_true=self.utrue)
        outcome_v1 = alg_v1.imp_euler()
        for j in range(1, 3):
            self.u_list[j] = outcome_v1[j]

        for i in range(3, self.N + 1):
            #预估矫正算法
            #计算预估值(利用 admas 外插)
            tui1, tui2, tui3 = (np.array([self.t[i - 1], self.u_list[i - 1]]),
                             np.array([self.t[i - 2], self.u_list[i - 2]]),
                             np.array([self.t[i - 3], self.u_list[i - 3]]))
            integral = (self.h / 12) * (
                    23 * self.func(tui1)
                    - 16 * self.func(tui2)
                    + 5 * self.func(tui3))
            u_nn = self.u_list[i - 1] + integral

            tu1, tu2, tu3 = (np.array([self.t[i], u_nn]),
                             np.array([self.t[i - 1], self.u_list[i - 1]]),
                             np.array([self.t[i - 2], self.u_list[i - 2]]))
            k1, k2, k3 = 5 * self.func(tu1), 8 * self.func(tu2), -self.func(tu3)

            if method == 'Error_Correction':
                integral = (self.h / 12) * (k1+ k2+ k3)
                self.u_list[i] = self.u_list[i - 1] + integral
            elif method == 'Iter':
                g = lambda x:(self.u_list[i - 1] +
                              (self.h / 12) * (5 * self.func(np.array([self.t[i], x])) +k2 +k3))
                u_iter = self.point_iteration(g,u_nn)
                integral = (self.h / 12) * (5 * self.func(np.array([self.t[i], u_iter])) +k2 +k3)
                self.u_list[i] = self.u_list[i - 1] + integral

        self.output()

        return self.u_list
    def runge_kutta_4degree(self):
        """经典的四阶RK方法"""
        print("Runge_Kutta_4degree")
        self.prp()

        alg_v1 = num_ode_lab1(self.start, self.start + 2 * self.h, 2, self.u_start, self.func, u_true=self.utrue)
        outcome_v1 = alg_v1.imp_euler()
        for j in range(1, 3):
            self.u_list[j] = outcome_v1[j]

        for i in range(3, self.N + 1):
            tu1 = np.array([self.t[i - 1], self.u_list[i - 1]])
            k1 = self.func(tu1)
            tu2 = np.array([self.t[i - 1] + self.h*0.5, self.u_list[i - 1] + k1 *0.5*self.h])
            k2 = self.func(tu2)
            tu3 = np.array([self.t[i - 1] + self.h*0.5, self.u_list[i - 1] + k2 *0.5*self.h])
            k3 = self.func(tu3)
            tu4 = np.array([self.t[i - 1] + self.h    , self.u_list[i - 1] + k3 * self.h])
            k4 = self.func(tu4)


            integral = self.h * (1./6) * (k1 + 2*k2 + 2*k3 + k4)
            self.u_list[i] = self.u_list[i - 1] + integral
        self.output()

        return self.u_list

        
if __name__ == "__main__":
    def true_u(t):
        return (1. + t ** 2.) ** 2.
    def f(x):
        t, u = x[0], x[1]
        return 4. * t * (np.power(np.abs(u), 1 / 2))

    def get_N(h,s,e):
        return float(e-s)/h

    start,end,u_start = 0, 5, 1
    # h = [1,0.5,0.1]
    # N = [int(get_N(h[i],start,end)) for i in range(len(h))]
    h = np.array([1 * np.power(0.5, i) for i in range(6)])
    N = ((end - start) / h).astype(int)


    u_error = []
    for n in range(len(h)):
        # 构造算法类
        alg = num_ode_lab2(start, end, N[n], u_start, f, true_u)
        t_grid = alg.iter_point()  # 获得格点
        u_euler = alg.runge_kutta_4degree()# 获得计算结果
        u_error.append(alg.error()[0])

        t_true = np.linspace(start, end, 1000)
        u_true = true_u(t_true)

        plt.plot(t_true, u_true, label="U True")
        plt.scatter(t_grid, u_euler, label=N[n], s=50 - n * 9, alpha=0.5)
        plt.legend()
        delay = 1

        plt.show()
        plt.pause(1)

        if n > 0:
            e = np.log(u_error[n] / u_error[n - 1])
            order = e / np.log(2)
        else:
            order = 0
        print("step final error:%.5f sum error:%.5f order:%.2f" % (*alg.error(), np.abs(order)))

    print(u_error)

    plt.show()

