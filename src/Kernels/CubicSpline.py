import numpy as np


class CubicSpline:
    def __init__(self):
        M_1_PI = 1.0 / np.pi

        self.dim = 2
        self.fac = 10 * M_1_PI / 7.0

    def evaluate(self, x: np.array, r: float, h: float):
        """
        Evaluates the kernel function for the given points.

        Parameters
        ------
        x: Distance vector

        r: Euclidian distance (scalar)

        h: Smoothing length
        """
        h1 = 1. / h
        q = r * h1

        # get the kernel normalizing factor
        if self.dim == 1:
            fac = self.fac * h1
        elif self.dim == 2:
            fac = self.fac * h1 * h1
        elif self.dim == 3:
            fac = self.fac * h1 * h1 * h1

        tmp2 = 2. - q

        # q > 2.0
        val = np.zeros(len(q))

        # q > 1.0
        mask = [(q > 1.0) & (q < 2.0)]
        val[mask] = 0.25 * tmp2[mask] * tmp2[mask] * tmp2[mask]

        # q < 1.0
        mask = [q < 1.0]
        val[mask] = 1 - 1.5 * q[mask] * q[mask] * (1 - 0.5 * q[mask])

        return val * fac