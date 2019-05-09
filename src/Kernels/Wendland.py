import numpy as np
from math import pow, pi
from numba import njit
from src.Kernels.Kernel import Kernel

class Wendland(Kernel):
    """ Wendland C4 kernel """

    @staticmethod
    @njit(fastmath=True)
    def evaluate(r: np.array, h: np.array):
        """
        Evaluates the kernel function for the given points.

        Parameters
        ------

        r: Euclidian distance (scalar)

        h: Smoothing length
        """
        k = np.zeros_like(r)
        fac = 9.0 / (4.0 * pi)
        for j in range(len(r)):
            alpha = fac / (h[j] * h[j])
            q = r[j] / h[j]

            if q >= 2.0:
                # Early exit.
                continue
            else:
                inner = 1.0 - 0.5 * q
                k[j] = alpha * (pow(inner, 6) * (35.0/12.0 * q * q + 3.0 * q + 1.0))
        return k
    
    @staticmethod
    @njit(fastmath=True)
    def gradient(x: np.array, r: np.array, h: np.array):
        """
        Calculates the gradient of the cubic spline at the given points.

        Parameters
        ------

        x: Coordinate of the point (1D).

        r: Euclidian distance (scalar)

        h: Smoothing length
        """
        k = np.zeros_like(r)
        fac = 9.0 / (4.0 * pi)
        for j in range(len(r)):
            alpha = fac / (h[j] * h[j])
            q = r[j] / h[j]

            if (q >= 2.0) or (r[j] < 1e-10):
                # Early exit
                continue
            else:
                inner = 1.0 - 0.5 * q
                grad = pow(inner, 5) * (-14.0 / 3.0) * q * (1 + 2.5 * q)
                norm = alpha * grad / (h[j] * r[j]) # Normalize
                k[j] = norm * x[j]
        return k