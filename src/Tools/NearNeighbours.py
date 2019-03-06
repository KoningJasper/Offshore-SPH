import numpy as np
from numba import njit, prange, jit, jitclass

@jitclass([])
class NearNeighbours:
    def __init__(self):
        pass
        
    def near(self, i: int, h: np.array, dist: np.array):
        """
            Find the neighbours near to a certain particle at index i.
        """

        # Create empty complete matrices
        q_i = np.zeros_like(h)
        h_i = np.zeros_like(h)
        near = [] # indices of near particles

        # Check each particle.
        J = len(h)
        for j in prange(J):
            h_i[j] = 0.5 * (h[i] + h[j]) # averaged h.
            q_i[j] = dist[j] / h_i[j] # q (norm-dist)

            if q_i[j] <= 3.0:
                near.append(j)
        return (h_i, q_i, np.array(near))