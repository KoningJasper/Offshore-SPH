from src.Tools.NearNeighbours import NearNeighbours
import numpy as np
from math import sqrt
from numba import jitclass, prange, float64, int64

spec = [
    ('dist', float64[:, :])
]
@jitclass(spec)
class NNEnumerate(NearNeighbours):
    def __init__(self):
        pass

    def update(self, pA: np.array):
        J = len(pA)
        self.dist = np.zeros((J, J))
        for i in prange(J):
            for j in prange(J):
                self.dist[i, j] = sqrt((pA['x'][i] - pA['x'][j]) ** 2 + (pA['y'][i] - pA['y'][j]) ** 2)
    
    def near(self, i: int, pA: np.array):
        near = [] # indices of near particles

        # Check each particle.
        J = len(pA)
        for j in prange(J):
            h_i = 0.5 * (pA['h'][i] + pA['h'][j]) # averaged h.
            q_i = self.dist[i, j] / h_i # q (norm-dist)

            if q_i <= 3.0:
                near.append(j)
        return np.array(near)