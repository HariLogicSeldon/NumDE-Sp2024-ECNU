import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

class num_pde():
    def __init__(self,timeStart,timeEnd,spaceStart,spaceEnd,spaceIndices,mu,uic,ubc):
        self.Xn = np.linspace(spaceStart,spaceEnd,spaceIndices+1)
        self.mu = mu
        deltaX = self.Xn[1] - self.Xn[0]
        n = 1
        Tn = [timeStart]
        tn = Tn[n-1] + pow(deltaX,2) * mu
        while(tn < timeEnd):
            tn = pow(deltaX,2) * mu + Tn[n-1]
            Tn.append(tn)
            n += 1
        self.Tn = Tn
        self.U = np.zeros((len(Tn),len(self.Xn)))
        self.phi = uic
        self.psi1,self.psi2 = ubc[0],ubc[1]

    def Uttimes(self,t):
        """t时刻 u 的所有值"""
        return self.U[t]
    # def Uxspace(self,x):
    #     """x位置处 u 的所有值"""
    #     return self.U[:,x]

    def __prp(self):
        self.U[0] = self.phi(self.Xn)
        self.U[:,0] = self.psi1(self.Tn)
        self.U[:,-1] = self.psi2(self.Tn)

    def deltaX(self,uTn,j):
        return uTn[j+1]+uTn[j-1]-2*uTn[j]

    def simple(self):
        self.__prp()
        for t in range(1,len(self.Tn)):
            uTn = self.Uttimes(t)
            uTn1 = self.Uttimes(t-1)
            for x in range(1,len(self.Xn)-1):
                uTn[x] = self.mu * self.deltaX(uTn1,x) + uTn1[x]
            self.U[t] = uTn
        return self.U

def standered(spaceIndices= 20, mu= 0.4,plot=False):
    def psi(x):
        return 0

    uic = lambda x: np.sin(x)

    def u(x, t):
        return np.exp(-t) * np.sin(x)

    timeStart, timeEnd = 0, 1
    spaceStart, spaceEnd = 0, np.pi

    ubc1, ubc2 = psi, psi

    alg = num_pde(timeStart, timeEnd, spaceStart, spaceEnd, spaceIndices, mu, uic, (ubc1, ubc2))
    u_solution = alg.simple()

    uTrueE = u(alg.Xn, alg.Tn[-1])
    error = np.abs(uTrueE - u_solution[-1]).max()
    print('error is {}'.format(error))

    if plot:
        x = np.linspace(spaceStart, spaceEnd, 1000)
        uTrueforplot = u(x, 1)
        plt.plot(x, uTrueforplot, zorder=-1, alpha=0.8, label='True')


        for time in range(u_solution.shape[0]):
            if time%(u_solution.shape[0]//4) == 0:
                plt.plot(alg.Xn, u_solution[time], alpha=0.5, marker='o', linestyle='-.',
                         label="t=%.2fs" % alg.Tn[time])
        plt.plot(alg.Xn, u_solution[-1], alpha=0.5, marker='o', linestyle='-.',
                 label="t=%.2fs" % alg.Tn[-1])
        plt.legend()
        plt.show()

    return error

def time_error():
    J = [20, 40, 80, 160, 320]
    errorList = np.zeros(len(J))
    for j in range(len(J)):
        errorList[j] = standered(J[j])
        if j > 0:
            e = np.log(errorList[j] / errorList[j - 1])
            order = e / np.log(2)
        else:
            order = 0
        print("Space devided into %d order:%.2f" % (J[j], np.abs(order)))



if __name__ == '__main__':
    time_error()
    error = standered(mu=0.4, plot=True)