from lab2 import *

if __name__ == "__main__":
    def true_u(t):
        return (1. + t ** 2.) ** 2.
    def f(x):
        t, u = x[0], x[1]
        return 4. * t * (np.power(np.abs(u), 1 / 2))

    start,end,u_start = 0, 5, 1
    h = np.array([0.1 * np.power(0.5,i) for i in range(6)])
    N =(2 / h).astype(int)


    u_error = []
    for n in range(len(h)):
        # 构造算法类
        alg = num_ode_lab2(start, end, N[n], u_start, f, true_u)
        t = alg.iter_point()  # 获得格点
        u_euler = alg.runge_kutta_4degree()# 获得计算结果
        u_true = true_u(t)
        u_error.append(alg.error()[0])

        plt.plot(t, u_true, label="U True")
        plt.legend()
        plt.scatter(t, u_euler, label=N[n], s=50 - n * 9, alpha=0.5)
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

