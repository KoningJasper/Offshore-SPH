import unittest
import abc
import numpy as np
from numba import njit, prange


class test_pass_func(unittest.TestCase):
    def test(self):
        x = np.arange(1e6); y = np.arange(1e6)
        z = test_pass_func.run(x, y, test_pass_func.func)

        self.assertEqual(len(z), 1e6)
        for j in range(len(x)):
            self.assertEqual(x[j] + y[j], z[j])

    def test_abstract(self):
        func: abstr = impl()

        x = np.arange(1e6); y = np.arange(1e6)
        z = test_pass_func.run(x, y, func.func)
        
        self.assertEqual(len(z), 1e6)
        for j in range(len(x)):
            self.assertEqual(x[j] + y[j], z[j])
            
    @staticmethod
    @njit(fastmath=True, parallel=True)
    def func(x, y):
        return x + y

    @staticmethod
    @njit(fastmath=True, parallel=True)
    def run(x: np.array, y: np.array, f):
        z = np.zeros_like(x)
        J = len(x)
        for j in prange(J):
            z[j] = f(x[j], y[j])

        return z

class abstr(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def func(x, y):
        pass

class impl(abstr):
    @staticmethod
    @njit(fastmath=True, parallel=True)
    def func(x, y):
        return x + y


if __name__ == "__main__":
    test_pass_func().test()