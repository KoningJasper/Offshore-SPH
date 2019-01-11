import numpy as np
from src.Particle import Particle


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

    def __init__(self, height: float = 1.0, gamma: float = 7.0, rho0: float = 1000.0):
        self.gamma = gamma
        self.rho0 = rho0
        self.H = height
        g = 9.81
        self.co = np.sqrt(200 * g * self.H)
        self.B = 200 * g * self.H / (self.rho0 * self.gamma)

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

    def Momentum(self, mass: float, p: Particle, pressure: np.array, rho: np.array, dwij: np.array) -> None:
        """
            Monaghan Momentum equation
        """
        rj = 1.0 / (p.rho * p.rho)
        pj = p.p * rj
        for i in range(len(pressure)):
            ri = 1.0 / (rho[i] * rho[i])
            qt = pressure[i] * ri
            pt = qt + pj

            # DO NOT USE +=
            # THIS DOES NOT WORK
            # IT HAS TAKEN YEARS OF MY LIFE.
            p.a = p.a - mass * pt * dwij[i, :]
        return p.a
        # pii: np.array = np.divide(pressure, rho * rho) # Others
        # pjj: float = p.p / (p.rho * p.rho) # Self
        # tmp: np.array = pii + pjj # Sum

        # # Create for multiple dimensions
        # fac: np.array = mass * tmp
        # vec = np.zeros([len(pressure), 2])
        # vec[:, 0] = fac
        # vec[:, 1] = fac

        # # Assign the acceleration
        # p.a += -1 * np.sum(np.multiply(vec, dwij), axis=0)

    @classmethod
    def Continuity(self, mass: float, pi: Particle, dwij: np.array, vij: np.array, numParticles: int) -> None:
        """
            SPH continuity equation
        """
        for i in range(numParticles):
            vdotw: float = np.dot(vij[i, :], dwij[i, :])
            pi.drho = pi.drho + mass * vdotw
        return pi.drho
        # vdotw = np.diag(np.dot(vij, np.transpose(dwij)))
        # pi.drho = pi.drho + np.sum(mass * vdotw)

    @classmethod
    def Gravity(self, p: Particle, gx: float, gy: float) -> None:
        p.a = p.a + np.array([gx, gy])
        return p.a

    def TaitEOS(self, pi: Particle) -> None:
        pi.p = self.B * ((pi.rho / self.rho0) ** self.gamma - 1)
        return pi.p
