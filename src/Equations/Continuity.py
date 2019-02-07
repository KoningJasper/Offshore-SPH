import numpy as np
import math
from numba import njit, prange

class Continuity():
    def calc(self, mass: np.array, dwij: np.array, vij: np.array) -> float:
        """
            SPH continuity equation; Calculates the change is density of the particles.
        """
        return _loop(mass, vij, dwij)

@njit
def _loop(m, vij, dwij):
    _arho = 0.0
    I = len(m)
    for i in prange(I):
        dot = vij[i, 0] * dwij[i, 0] + vij[i, 1] * dwij[i, 1]
        _arho += m[i] * dot
    return _arho