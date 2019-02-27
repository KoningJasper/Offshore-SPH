import numpy as np
from math import sqrt, pow
from numba import njit, prange

@njit(fastmath=True)
def KineticEnergy(J, pA) -> float:
    k = 0.0
    for j in prange(J):
        v2 = pow(pA[j]['vx'], 2) + pow(pA[j]['vy'], 2)
        k += 0.5 * pA[j]['m'] * v2
    return k
