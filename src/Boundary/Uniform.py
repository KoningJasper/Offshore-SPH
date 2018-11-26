import numpy as np


class Uniform:
    @staticmethod
    def create(start: int, stop: int, cross_start: int, cross_stop: int, N: int, cross_N: int):
        """
        Creates an uniform boundary running from start to stop.

        Generally start and stop represent X-axis and cross represents position on y-axis

        Parameters
        start : integer
        """

        # Create a uniform grid of particles
        xv = np.linspace(start, stop, N)
        yv = np.linspace(cross_start, cross_stop, cross_N)
        x, y = np.meshgrid(xv, yv, indexing='ij')

        # Convert to a single matrix
        x = np.concatenate(x)
        y = np.concatenate(y)

        # Return the single coordinate matrix
        N_elem = N * cross_N
        return np.hstack((x.reshape(N_elem, 1), y.reshape(N_elem, 1)))

