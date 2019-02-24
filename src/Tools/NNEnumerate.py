from NearNeighbours import NearNeighbours
import numpy as np
from scipy.spatial.distance import cdist
from numba import jitclass, float64, int64

spec = [
    ('dist', float64[:, :])
]
@jitclass(spec)
class NNEnumerate(NearNeighbours):
    def __init__(self):
        pass

    def update(self, pA: np.array):
        self.dist = cdist(r, r)
        pass
    
    def near(self, i: int):
        pass