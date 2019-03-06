import numpy as np
from typing import List
from numba import njit

@njit(fastmath=True)
def XSPH(epsilon, p, comp):
    """
        XSPH Correction


        Parameters
        -------

        epsilon: scaling parameter for XSPH correction, normally 0.5

        p: self-array

        comp: Other particles

        Returns
        -------
        A list containing [xsph_x, xsph_y]
    """
    xsph = [0.0, 0.0]
    J = len(comp)
    for j in range(J):
        rho_ij = 0.5 * (p['rho'] + comp[j]['rho'])
        fac = - epsilon * comp[j]['m'] * comp[j]['w'] / rho_ij
        xsph[0] += fac * comp[j]['vx']
        xsph[1] += fac * comp[j]['vy']
    return xsph