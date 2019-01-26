import numpy as np
import math
from numba import vectorize, jit, prange

class Continuity():
    def calc(self, mass: np.array, dwij: np.array, vij: np.array) -> float:
        """
            SPH continuity equation; Calculates the change is density of the particles.
        """
        return _calc_outer(mass, dwij, vij)

@vectorize(['float64(float64, float64, float64)'], target='parallel')
def _calc_inner(mass, dwij, vij):
    return mass * vij * dwij

@jit
def _calc_outer(mass, dwij, vij):
    _arho = 0.0
    I = len(dwij)
    inner = _calc_inner(mass, dwij, vij)

    # Stupid sum; for numba
    for i in prange(I):
        _arho += inner[i]

    return _arho
