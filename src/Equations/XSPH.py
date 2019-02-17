import numpy as np
from typing import List
from numba import njit, prange

@njit(fastmath=True, parallel=True)
def XSPH(epsilon, rho_i, m_j, rho_j, vij, wij):
    """
        XSPH Correction


        Parameters
        -------

        epsilon: scaling parameter for XSPH correction, normally 0.5

        rho_i: particle self density.

        m_j: masses of other particles.

        rho_j: densities of other particles.

        vij: Speed difference between self and other particles. (2D)

        wij: Kernel values of other particles.


        Returns
        -------
        A list containing [xsph_x, xsph_y]
    """
    xsph = [0.0, 0.0]
    J = len(m_j)
    for j in prange(J):
        rho_ij = 0.5 * (rho_i + rho_j[j])
        fac = - epsilon * m_j[j] * wij[j] / rho_ij
        xsph[0] += fac * vij[j, 0]
        xsph[1] += fac * vij[j, 1]
    return xsph