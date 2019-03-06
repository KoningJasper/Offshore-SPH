import numpy as np
from math import pow, pi
from numba import njit, prange
from src.Kernels.Kernel import Kernel

class CubicSpline(Kernel):
    alpha: float

    def __init__(self):
        self.alpha = 10 / (7 * pi)

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
        fac = 10 / (7 * pi)
        for j in prange(len(r)):
            alpha = fac / (h[j] * h[j])
            q = r[j] / h[j]

            # Three cases
            if q > 2.0:
                continue
            elif q > 1.0:
                k[j] = alpha * 0.25 * pow(2 - q, 3)
            else:
                k[j] = alpha * (1 - 1.5 * pow(q, 2) * (1 - 0.5 * q))
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
        fac = 10 / (7 * pi)
        for j in prange(len(r)):
            alpha = fac / (h[j] * h[j])
            q = r[j] / h[j]

            # Calc derivative.
            grad = 0.0
            if (q > 2.0) or (r[j] < 1e-10):
                # Early exit, second condition for divide by zero.
                continue
            elif q > 1.0:
                grad = -0.75 * pow(2 - q, 2)
            else:
                grad = -3 * q * (1 - 0.75 * q)
            
            norm = alpha * grad / (h[j] * r[j]) # Normalize
            k[j] = norm * x[j]
        return k