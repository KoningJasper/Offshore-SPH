import numpy as np
from typing import List
from src.Particle import Particle

class Momentum():
    """ Monaghan momentum equation. """
    alpha: float
    beta: float
    eta: float

    def __init__(self, alpha: float = 0.01, beta: float = 0.0, eta: float = 0.5):
        # Monaghan parameters
        self.alpha = alpha
        self.beta = beta
        self.eta = eta

    def calc(self, mass: float, p: Particle, xij: np.array, rij: np.array, vij: np.array, pressure: np.array, rho: np.array, hij: np.array, cij: np.array, wij: np.array, dwij: np.array, xsph: bool) -> List:
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

            xsph: XSPH correction velocity.



            Returns:
            ------

            If xsph is true a list is returned with [acceleration, xsph]

            If xsph is false a list is returned with [acceleration]
        """

        # Average density.
        rhoij: np.array = 0.5 * (p.rho + rho)

        # Compute first (easy) part.
        tmp = p.p / (p.rho * p.rho) + pressure / (rho * rho)
        _au = np.sum(- mass * tmp * dwij[:, 0], axis=0)
        _av = np.sum(- mass * tmp * dwij[:, 1], axis=0)

        # Diffusion
        dot = np.sum(vij * xij, axis=1) # Row by row dot product
        piij = np.zeros(len(pressure))

        # Perform diffusion for masked entities
        mask       = dot < 0
        if any(mask):
            muij       = hij[mask] * dot[mask] / (rij[mask] * rij[mask] + 0.01 * hij[mask] * hij[mask])
            muij       = muij
            piij[mask] = muij * (self.beta * muij - self.alpha * cij[mask])
            piij[mask] = piij[mask] / rhoij[mask]

        # Calculate change in density.
        _au_d = np.sum(- mass * piij * dwij[:, 0])
        _av_d = np.sum(- mass * piij * dwij[:, 1])

        # XSPH
        if xsph == True:
            _xsphtmp = mass / rhoij * wij
            _xsphx = np.sum(_xsphtmp * -vij[:, 0], axis=0) # -vij = vji
            _xsphy = np.sum(_xsphtmp * -vij[:, 1], axis=0)

            return [np.array([_au + _au_d, _au + _av_d]), np.array([self.eta * _xsphx, self.eta * _xsphy])]
        else:
            return [np.array([_au + _au_d, _au + _av_d])]