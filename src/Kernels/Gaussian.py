import numpy as np
import math
from numba import vectorize, njit
from src.Kernels.Kernel import Kernel


class Gaussian(Kernel):
    """
    This represents the gaussian kernel.

    The equation of the gaussian kernel is as follows:

    W(R, h) = alpha * e ^ (-R^2)

    """

    alpha: float

    def __init__(self):
        # Calculate alpha coeff.
        # 2D, according to Liu, M.B. (2009)
        self.alpha = 1 / (np.pi)

    def evaluate(self, r: np.array, h: np.array) -> np.array:
        return _evaluate_vec(r, h, self.alpha)
        
    def derivative(self, r: np.array, h: np.array) -> np.array:
        """ Computes the derivative of the gradient for all the points. """
        return _derivative_vec(r, h, self.alpha)

    def gradient(self, x: np.array, r: np.array, h: np.array) -> np.array:
        """ 
        Evaluates the gradient with respect to point x1 and x2 at point x1.

        grad = -2 * q * alpha * exp(-q^2)
        """
        return _gradient_vec(x, r, h, self.alpha)

""" Outside of class for numba. """
@vectorize(['float64(float64, float64, float64)'], target='parallel')
def _evaluate_vec(r, h, alpha):
    """ Vectorized method for evaluation of kernel function. """
    q = r / h
    if q <= 3:
        alpha_c = alpha / (h * h)
        return alpha_c * math.exp(-q * q)
    else:
        return 0

@vectorize(['float64(float64, float64, float64)'], target='parallel')
def _derivative_vec(r, h, alpha):
    q = r / h
    if q <= 3:
        alpha_c = alpha / (h * h)
        return -2 * q * alpha_c * math.exp(-q * q)
    else:
        return 0

@vectorize(['float64(float64, float64, float64, float64)'], target='parallel')
def _gradient_vec(x, r, h, alpha):
    tmp = r * h
    if tmp > 1e-12:
        # Treshold value to prevent divide by zero, q should always be bigger than 0.
        q = r / h
        if q <= 3:
            alpha_c = alpha / (h * h)
            dwdq = -2 * q * alpha_c * math.exp(-q * q)

            return dwdq / tmp * x
        else:
            return 0
    else:
        return 0