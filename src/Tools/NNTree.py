import numpy as np
from scipy.spatial import cKDTree
from src.Tools.NearNeighbours import NearNeighbours

class NNTree(NearNeighbours):
    def __init__(self):
        pass
    
    def update(self, pA: np.array):
        self.h    = pA['h']
        self.r    = np.stack((pA['x'], pA['y']), axis=-1)
        self.tree = cKDTree(self.r)

    def near(self, i: int) -> np.array:
        return np.array(self.tree.query_ball_point(self.r[i], 3 * self.h[i]))