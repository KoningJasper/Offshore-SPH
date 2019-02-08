import numpy as np
import math
from numba import njit, prange
from src.Common import get_label_code

class BoundaryForce():
    """
        Computes the boundary force
        based on the equations provided in Monaghan 1992
    """

    r0: float
    D: float
    p1: float
    p2: float

    def __init__(self, r0: float, D: float, p1: float = 4, p2: float = 2):
        """
            r0: Initial spacing between particles

            D: Force factor of dimension [v * v], for dams/bores/weirs equal to 5gH.
        """
        self.r0 = r0
        self.D = D

        # Check condition
        assert(p1 > p2)
        self.p1 = p1
        self.p2 = p2

    def calc(self, p: np.array, comp: np.array) -> np.array:
        [a_x, a_y] = _loop(get_label_code('boundary'), self.r0, self.D, self.p1, self.p2, p, comp)

        # Set the new properties
        p['ax'] += a_x
        p['ay'] += a_y
        return p

def _loop(bc: int, r0: float, D: float, p1: float, p2: float, p: np.array, comp: np.array):
    f = [0., 0.]
    J = len(comp)
    for j in range(J):
        if (comp[j]['label'] != bc) or (comp[j]['r'] > r0):
            continue
        elif comp[j]['r'] > 1e-12:
            frac = r0 / comp[j]['r']
            tmp = math.pow(frac, p1) - math.pow(frac, p2)
            fac = D * tmp

            f[0] = fac * comp[j]['x'] / math.pow(comp[j]['r'], 2),
            f[1] = fac * comp[j]['y'] / math.pow(comp[j]['r'], 2)
    return f