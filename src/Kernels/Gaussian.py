import numpy as np
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

    def evaluate(self, x: np.array, r: np.array, h: np.array) -> np.array:
        # Normalize distance
        q: np.array = np.divide(r, np.power(h, 2))

        # Calculate alpha complete
        alpha_c: np.array = self.alpha / np.power(h, 2)

        # Calculate kernel values where smaller or equal than 3.
        k = np.zeros(len(r))
        mask = q <= 3
        k[mask] = np.multiply(alpha_c[mask], np.exp(
            np.multiply(-q[mask], q[mask])))

        return k

    def derivative(self, r: np.array, h: np.array) -> np.array:
        """ Computes the derivative of the gradient for all the points. """
        # Normalize distance
        q: np.array = np.divide(r, np.power(h, 2))

        # Calculate alpha complete
        alpha_c: np.array = self.alpha / np.power(h, 2)

        # Calculate kernel gradient where smaller or equal than 3.
        k = np.zeros(len(r))
        mask = q <= 3
        k[mask] = -2 * np.multiply(q[mask], np.multiply(alpha_c[mask],
                                                        np.exp(np.multiply(-q[mask], q[mask]))))

        return k

    def gradient(self, x: np.array, r: np.array, h: np.array) -> np.array:
        """ 
        Evaluates the gradient with respect to point x1 and x2 at point x1.

        grad = -2 * q * alpha * exp(-q^2)
        """

        # compute the gradient.
        w_grad = np.zeros(len(r))

        # Treshold value to prevent divide by zero, q should always be bigger than 0.
        mask = r > 1e-12
        w_grad[mask] = self.derivative(r[mask], h[mask]) / (h[mask] * r[mask])

        # Duplicate for number of dimensions
        grad_dim = np.ones([len(r), 2])
        grad_dim[:, 0] = w_grad
        grad_dim[:, 1] = w_grad

        return grad_dim * x
