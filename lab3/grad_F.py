import autograd.numpy as np
from autograd import elementwise_grad as egrad

class grad_F:
    """自动实现导数"""""
    def __init__(self, f):
        self.df = egrad(f)

    def f_t(self,x):
        return self.df(x)[0]

    def f_u(self,x):
        return self.df(x)[1]

    def f_tt(self,x):
        func = egrad(self.f_t)
        func_v = func(x)
        return func_v[0]

    def f_tu(self,x):
        func = egrad(self.f_t)
        func_v = func(x)
        return func_v[1]

    def f_uu(self,x):
        func = egrad(self.f_u)
        func_v = func(x)
        return func_v[1]

    def f_ut(self,x):
        func = egrad(self.f_u)
        func_v = func(x)
        return func_v[0]

def f(x):
    return x[0] ** 2 + (x[1] ** 3) * x[0]

