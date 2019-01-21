import numpy as np
from src.Particle import Particle
from typing import List

class WCSPH:
    """ Weakly compressible SPH """

    """ Tait EOS factor """
    gamma: float

    """ Rest density of fluid [kg/m^3] """
    rho0: float

    """ Factor for Tait EOS [(m/s)^2]"""
    B: float

    """ Speed of sound [m/s] """
    co: float

    """ Water column height """
    H: float

    beta: float
    alpha: float
    eta: float

    def __init__(self, height: float = 1.0, gamma: float = 7.0, rho0: float = 1000.0, beta: float = 0.0, alpha: float = 0.01, eta: float = 0.5):
        self.gamma = gamma
        self.rho0 = rho0
        self.H = height

        # Monaghan (2002) p. 1746
        self.co = 10.0 * np.sqrt(2 * 9.81 * self.H)
        self.B = self.co * self.co * self.rho0 / self.gamma

        # Monaghan parameters
        self.beta = beta
        self.alpha = alpha
        self.eta = eta

    def inital_condition(self, p: Particle) -> None:
        self.height_density(p)
        self.TaitEOS(p)

    def height_density(self, p: Particle) -> None:
        """ Sets the density of particle based on height (y) of particle """
        y = p.r[1]
        frac  = self.rho0 * 9.81 * (self.H - y) / self.B
        p.rho = self.rho0 * (1 + frac) ** (1 / self.gamma)

    @classmethod
    def loop_initialize(self, p: Particle):
        """ Initialize loop, resets acceleration and density change. """
        p.a = np.array([0., 0.])
        p.drho = 0.

    def Momentum(self, mass: float, p: Particle, xij: np.array, rij: np.array, vij: np.array, pressure: np.array, rho: np.array, hij: np.array, cij: np.array, wij: np.array, dwij: np.array, xsph: bool) -> List:
        """
            Monaghan Momentum equation


            Parameters:

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

    @classmethod
    def Continuity(self, mass: float, dwij: np.array, vij: np.array) -> float:
        """
            SPH continuity equation; Calculates the change is density of the particles.
        """
        # Init
        _arho = 0.0

        # Calc change in density
        vijdotwij = np.sum(vij * dwij, axis=1) # row by row dot product
        _arho = np.sum(mass * vijdotwij, axis=0)

        return _arho


    @classmethod
    def Gravity(self, a: np.array, gx: float, gy: float) -> np.array:
        return a + np.array([gx, gy])

    def TaitEOS(self, pi: Particle) -> float:
        ratio = pi.rho / self.rho0
        temp  = ratio ** self.gamma
        pi.p  = (temp - 1.0) * self.B

        return pi.p
