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

    def calc(self, rho_i: float, p_i: float, cs_i: float, h_i: float, m_j: np.array, rho_j: np.array, p_j: np.array, cs_j: np.array, h_j: np.array, xij, rij, vij, dwij) -> float:
        """
            Monaghan Momentum equation


            Parameters:
            ------

            rho_i: Density of the particle

            p_i: Pressure of the particle
            
            cs_i: Speed of sound of the particle

            h_i: h of the particle

            m_j: masses of the other particles

            rho_j: densities of the other particles.

            p_j: pressures of the other particles.

            cs_j: speed of sound of the other particles.

            h_j: h of the other particles

            xij: Position difference

            rij: (Norm) distance between particles.

            vij: Velocity difference (vi - vj)

            dwij: Kernel gradient


            Returns:
            ------

            acceleration
        """

        return _loop(self.alpha, self.beta, rho_i, p_i, cs_i, h_i, m_j, rho_j, p_j, cs_j, h_j, xij, rij, vij, dwij);

@njit
def _loop(alpha: float, beta: float, rho_i: float, p_i: float, cs_i: float, h_i: float, m_j: np.array, rho_j: np.array, p_j: np.array, cs_j: np.array, h_j: np.array, xij, rij, vij, dwij):
    slf = p_i / (rho_i * rho_i) # Lifted from the loop since it's constant.

    a = 0.0
    J = len(p_j)
    for j in prange(J):
        # Compute acceleration due to pressure.
        othr = p_j[j] / (rho_j[j] * rho_j[j])

        # (Artificial) Viscosity
        dot = vij[j] * xij[j]
        PI_ij = 0.0
        if dot < 0:
            # Averaged properties
            hij = 0.5 * (h_i + h_j[j]) # Averaged h.
            cij = 0.5 * (cs_i + cs_j[j]) # Averaged speed of sound.
            rhoij = 0.5 * (rho_i + rho_j[j]) # Averaged density.

            muij = hij * dot / (rij[j] * rij[j] + 0.01 * hij * hij)
            ppij = muij * (beta * muij - alpha * cij)
            PI_ij = ppij / rhoij
        
        # Compute final factor
        factor = slf + othr + PI_ij

        # Compute acceleration
        a += - m_j[j] * factor * dwij[j]
    return a