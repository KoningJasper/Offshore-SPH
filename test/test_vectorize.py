# Add parent folder to path
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import unittest
import numpy as np
from numba import vectorize, njit, jit
from time import perf_counter
from src.Equations.Continuity import Continuity
from src.Common import computed_dtype

class test_vectorize(unittest.TestCase):
    def test(self):
        x = np.arange(1e6)
        y = np.arange(1e6)
        
        test_vectorize.sum(test_vectorize.vec(x, y))

        start = perf_counter()
        v = test_vectorize.sum(test_vectorize.vec(x, y))
        vTime = perf_counter() - start
        print(f'Vectorize: {vTime:f} [s]')

        test_vectorize.non_vec(x, y)
        start = perf_counter()
        f = test_vectorize.non_vec(x, y)
        nTime = perf_counter() - start
        print(f'njit: {nTime:f} [s]')

        print(f'Vec provides {nTime / vTime}x speed-up.')

    def test_continuity(self):
        m = np.arange(1e6)

        vij = np.transpose(np.vstack((m, m)))
        dwij = np.transpose(np.vstack((m, m)))

        comp = np.zeros_like(m, dtype=computed_dtype)
        comp['m'] = m
        comp['vx'] = m
        comp['vy'] = m
        comp['dw_x'] = m
        comp['dw_y'] = m

        test_vectorize.sum(test_vectorize.continuity_vec(m, m, m)) + test_vectorize.sum(test_vectorize.continuity_vec(m, m, m))
        Continuity(np.array([]), comp)

        start = perf_counter()
        v = test_vectorize.sum(test_vectorize.continuity_vec(m, m, m)) + test_vectorize.sum(test_vectorize.continuity_vec(m, m, m))
        vTime = perf_counter() - start
        print(f'Vectorize: {vTime:f} [s]')

        start = perf_counter()
        f = Continuity(np.array([]), comp)
        nTime = perf_counter() - start
        print(f'njit: {nTime:f} [s]')

        print(f'Vec provides {nTime / vTime}x speed-up.')

    @staticmethod
    @vectorize('float64(float64, float64, float64)', fastmath=True)
    def continuity_vec(m, vij, dwij):
        dot = vij * dwij
        return m * dot

    @staticmethod
    @vectorize('float64(float64, float64)', fastmath=True)
    def vec(x, y):
        return x * y

    @staticmethod
    @njit('float64(float64[:])', fastmath=True)
    def sum(m):
        I = len(m); _ = 0.0
        for i in range(I):
            _ += m[i]
        return _

    @staticmethod
    @njit('float64(float64[:], float64[:])', fastmath=True)
    def non_vec(x, y):
        J = len(x); s = 0.0
        for j in range(J):
            s += x[j] * y[j]
        return s

if __name__ == "__main__":
    test_vectorize().test()
    test_vectorize().test_continuity()