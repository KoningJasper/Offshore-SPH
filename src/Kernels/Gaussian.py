import math
from numba import vectorize
from src.Kernels.Kernel import Kernel


class Gaussian(Kernel):
    """
    This represents the gaussian kernel.

    The equation of the gaussian kernel is as follows:

    W(R, h) = alpha * e ^ (-R^2)

    """

    @staticmethod
    @vectorize('float64(float64, float64)', fastmath=True)
    def evaluate(r, h):
        alpha = 1 / math.pi
        q = r / h
        if q <= 3:
            alpha_c = alpha / (h * h)
            return alpha_c * math.exp(-q * q)
        else:
            return 0.0

    @staticmethod
    @vectorize('float64(float64, float64)', fastmath=True)
    def derivative(r, h):
        alpha = 1 / math.pi
        q = r / h
        if q <= 3:
            alpha_c = alpha / (h * h)

            return -2 * q * alpha_c * math.exp(-q * q)
        else:
            return 0.0

    @staticmethod
    @vectorize('float64(float64, float64, float64)', fastmath=True)
    def gradient(x, r, h):
        """ 
        Evaluates the gradient with respect to point x1 and x2 at point x1.

        grad = -2 * q * alpha * exp(-q^2)
        """
        alpha = 1 / math.pi
        tmp = r * h
        if tmp > 1e-12:
            # Treshold value to prevent divide by zero, q should always be bigger than 0.
            q = r / h
            if q <= 3:
                alpha_c = alpha / (h * h)
                dwdq = -2 * q * alpha_c * math.exp(-q * q)

                return dwdq / tmp * x
            else:
                return 0.0
        else:
            return 0.0