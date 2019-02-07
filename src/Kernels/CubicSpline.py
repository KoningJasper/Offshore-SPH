import numpy as np
from math import pow, pi
from numba import vectorize
from src.Kernels.Kernel import Kernel

class CubicSpline(Kernel):
    alpha: float

    def __init__(self):
        self.alpha = 10 / (7 * pi)

    def evaluate(self, r: np.array, h: np.array):
        """
        Evaluates the kernel function for the given points.

        Parameters
        ------

        r: Euclidian distance (scalar)

        h: Smoothing length
        """
        return _kernel(self.alpha, r, h)
    
    def gradient(self, x: np.array, r: np.array, h: np.array):
        """
        Calculates the gradient of the cubic spline at the given points.

        Parameters
        ------

        x: Coordinate of the point (1D).

        r: Euclidian distance (scalar)

        h: Smoothing length
        """
        return _gradient(self.alpha, x, r, h)

@vectorize(['float64(float64, float64, float64)'], target='parallel')
def _kernel(fac, r, h):
    alpha = fac / (h * h)
    q = r / h

    # Three cases
    if q > 2.0:
        return 0.0
    elif q > 1.0:
        return alpha * 0.25 * pow(2 - q, 3)
    else:
        return alpha * (1 - 1.5 * pow(q, 2) * (1 - 0.5 * q))

@vectorize(['float64(float64, float64, float64, float64)'], target='parallel')
def _gradient(fac, x, r, h):
    alpha = fac / (h * h)
    q = r / h

    # Calc derivative.
    grad = 0.0
    if (q > 2.0) or (r < 1e-10):
        # Early exit, second condition for divide by zero.
        return 0.0
    elif q > 1.0:
        grad = -0.75 * pow(2 - q, 2)
    else:
        grad = -3 * q * (1 - 0.75 * q)
    
    norm = alpha * grad / (h * r) # Normalize
    return norm * x