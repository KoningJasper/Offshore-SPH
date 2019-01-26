# Add parent folder to path; for directly running the file
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import unittest
import numpy as np
from scipy.spatial.distance import cdist
from time import perf_counter

from src.Kernels.Gaussian import Gaussian

class test_numba_kernel(unittest.TestCase):
    def test(self):
        """ Compare the numba implementation against the old implementation. """
        x: np.array = np.linspace(0, 1000, num=1_000)
        y: np.array = np.linspace(0, 1000, num=1_000)
        r: np.array = np.transpose(np.vstack((x, y)))

        # Calculate the complete distance matrix.
        d: np.array = cdist(r, r)

        # alpha constant parameter
        alpha: float = 1 / np.pi

        # Create kernel
        kernel = Gaussian()

        # Evaluate the old method.
        o_t: float = 0.
        n_t: float = 0.

        wij_o_total: np.array = np.zeros([len(r), len(r) - 1])
        wij_n_total: np.array = np.zeros([len(r), len(r) - 1])
        dwij_o_total: np.array = np.zeros([len(r), len(r) - 1, 2])
        dwij_n_total: np.array = np.zeros([len(r), len(r) - 1, 2])
        for i in range(len(r)):
            dd: np.array = np.delete(d[i, :], i)
            h: np.array  = dd * 1.3
            xij: np.array = r[i, :] - np.delete(r, i, axis=0)

            # Old
            s_o = perf_counter()
            wij_o = self.old_func(dd, h, alpha)
            dwij_o = self.old_gradient(xij, dd, h, alpha)
            o_t += perf_counter() - s_o

            # Numba
            s_n = perf_counter()
            wij_n = kernel.evaluate(dd, h)
            dwij_x = kernel.gradient(xij[:, 0], dd, h)
            dwij_y = kernel.gradient(xij[:, 1], dd, h)
            n_t += perf_counter() - s_n

            # Add to giant matrices
            wij_o_total[i, :] = wij_o
            wij_n_total[i, :] = wij_n


        # Compare the results
        I, J = wij_o_total.shape
        for i in range(I):
            for j in range(J):
                self.assertAlmostEqual(wij_o_total[i, j], wij_n_total[i, j])

        print(f'Timing:')
        print(f'Old: {o_t:f} [s]')
        print(f'New: {n_t:f} [s]')

    def old_func(self, r: np.array, h: np.array, alpha: float):
        # Normalize distance
        q: np.array = np.divide(r, h)

        # Calculate alpha complete
        alpha_c: np.array = alpha / np.power(h, 2)

        # Calculate kernel values where smaller or equal than 3.
        k = np.zeros(len(r))
        mask = q <= 3
        k[mask] = np.multiply(alpha_c[mask], np.exp(
            np.multiply(-q[mask], q[mask])))

        return k

    def old_derivative(self, r, h, alpha):
        # Normalize distance
        q: np.array = np.divide(r, h)

        # Calculate alpha complete
        alpha_c: np.array = alpha / np.power(h, 2)

        # Calculate kernel gradient where smaller or equal than 3.
        k = np.zeros(len(r))
        mask = q <= 3
        k[mask] = -2 * np.multiply(q[mask], np.multiply(alpha_c[mask],
                                                        np.exp(np.multiply(-q[mask], q[mask]))))

        return k

    
    def old_gradient(self, x, r, h, alpha):
        # compute the gradient.
        w_grad = np.zeros(len(r))

        # Treshold value to prevent divide by zero, q should always be bigger than 0.
        mask = r > 1e-12
        w_grad[mask] = self.old_derivative(r[mask], h[mask], alpha) / (h[mask] * r[mask])

        # Duplicate for number of dimensions
        grad_dim = np.ones([len(r), 2])
        grad_dim[:, 0] = w_grad
        grad_dim[:, 1] = w_grad

        return grad_dim * x