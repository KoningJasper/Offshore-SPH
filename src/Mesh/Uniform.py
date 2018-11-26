import numpy as np
import src.Config.Props


class Uniform:
    @staticmethod
    def create(P: src.Config.Props):
        # Create a uniform grid of particles
        xv = np.linspace(0, 2, P.L)
        yv = np.linspace(0, 2, P.L)
        x, y = np.meshgrid(xv, yv, indexing='ij')

        # Convert to a single matrix
        x = np.concatenate(x)
        y = np.concatenate(y)

        # Return the single coordinate matrix
        return np.hstack((x.reshape(P.N, 1), y.reshape(P.N, 1)))
