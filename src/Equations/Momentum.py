import numpy as np
from numba import vectorize, jit, prange, njit
from typing import List
from src.Particle import Particle

class Momentum():
    """ Monaghan momentum equation. """
    alpha: float
    beta: float

    def __init__(self, alpha: float = 0.01, beta: float = 0.0):
        # Monaghan parameters
        self.alpha = alpha
        self.beta = beta

    def calc(self, m: np.array, p: Particle, xij: np.array, rij: np.array, vij: np.array, pressure: np.array, rho: np.array, hij: np.array, cij: np.array, wij: np.array, dwij: np.array) -> float:
        """
            Monaghan Momentum equation


            Parameters:
            ------

            p: The particle to compute the acceleration for.

            xij: Position difference

            rij: (Norm) distance between particles.

            vij: Velocity difference (vi - vj)

            pressure: Pressures of the near particles.

            rho: Densities of the near particles.

            hij:

            cij: Speed of sound [m/s].

            dwij: Kernel gradient


            Returns:
            ------

            acceleration
        """

        # Average density.
        rhoij: np.array = 0.5 * (p.rho + rho)

        # Compute first (easy) part.
        _a = _acc_press(m, p.p, p.rho, pressure, rho, dwij)

        # Change in density due to diffusion.
        _ad = _sum(_diff_vec(vij, xij, hij, rij, self.beta, self.alpha, cij, rhoij, m, dwij))

        return _a + _ad

@njit(fastmath=True)
def _xsph(m: np.array, rho_s: float, rho_o: np.array, vij: np.array, wij: np.array) -> float:
    _xsp = 0.0
    I = len(vij)
    for i in prange(I):
        rhoij = 0.5 * (rho_s + rho_o)
        _xsphtmp = m[i] / rhoij * wij[i]
        _xsp += _xsphtmp * -vij[i]
    return _xsp

@njit(fastmath=True)
def _acc_press(m: np.array, p_s: float, rho_s: float, p_o: np.array, rho_o: np.array, dwij: np.array) -> float:
    I = len(p_o)
    _a = 0.0
    slf = p_s / (rho_s * rho_s)
    for i in prange(I):
        tmp = slf + p_o[i] / (rho_o[i] * rho_o[i])
        _a += - m[i] * tmp * dwij[i]
    return _a

@vectorize(['float64(float64, float64, float64, float64, float64, float64, float64, float64, float64, float64)'], target='parallel')
def _diff_vec(vij, xij, hij, rij, beta, alpha, cij, rhoij, m, dwij):
    dot = vij * xij
    if dot < 0.:
        muij = hij * dot / (rij * rij + 0.01 * hij * hij)
        ppij = muij * (beta * muij - alpha * cij)
        piij = ppij / rhoij
        return - m * piij * dwij
    return 0.

@njit(fastmath=True)
def _sum(matrix: np.array):
    _s = 0
    I, J = matrix.shape
    for i in prange(I):
        for j in prange(J):
            _s += matrix[i, j]
    return _s
