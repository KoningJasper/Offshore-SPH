import numpy as np
from typing import List
from numba import njit, prange

class XSPH():
    """ XSPH correction. """
    epsilon: float

    def __init__(self, epsilon: float = 0.5):
        self.epsilon = epsilon

    def calc(self, rho_i: float, m_j: np.array, rho_j: np.array, vij: np.array, wij: np.array) -> List[float]:
        return _loop(self.epsilon, rho_i, m_j, rho_j, vij, wij)

@njit
def _loop(eps, rho_i, m_j, rho_j, vij, wij):
    # vij is 2D.
    xsph = [0.0, 0.0]
    J = len(m_j)
    for j in range(J):
        rho_ij = 0.5 * (rho_i + rho_j[j])
        fac = - eps * m_j[j] * wij[j] / rho_ij
        xsph[0] += fac * vij[j, 0]
        xsph[1] += fac * vij[j, 1]
    return xsph