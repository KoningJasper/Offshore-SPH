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
        self.co = 10.0 * np.sqrt(2 * 9.81 * height)
        self.B = self.co * self.co * self.rho0 / self.gamma

    def inital_condition(self, p: Particle) -> None:
        self.height_density(p)
        self.TaitEOS(p)

    def height_density(self, p: Particle) -> None:
        """ Sets the density of particle based on height (y) of particle """
        y = p.r[1]
        p.rho = p.rho * \
            (1 + (p.rho * 9.81 * (self.H - y)) / self.B) ** (1/self.gamma)

    @classmethod
    def loop_initialize(self, p: Particle):
        """ Initialize loop, resets certain properties """
        p.a = np.array([0., 0.])
        p.drho = 0.

    @classmethod
    def Momentum(self, mass: float, p: Particle, pressure: np.array, rho: np.array, dwij: np.array) -> None:
        """
            Monaghan Momentum equation
        """
        pii: np.array = np.divide(pressure, np.power(rho, 2))
        pjj: float = p.p / p.rho ** 2
        tmp: np.array = pii + pjj

        # Create for multiple dimensions
        fac: np.array = mass * tmp * p.rho
        vec = np.zeros([len(pressure), 2])
        vec[:, 0] = fac
        vec[:, 1] = fac

        # Assign the acceleration
        p.a += np.sum(np.multiply(vec, dwij), axis=0)

    @classmethod
    def Continuity(self, mass: float, pi: Particle, xij: np.array, rij: np.array, dwij: np.array, vij: np.array) -> None:
        """
            SPH continuity equation
        """
        vdotw = np.diag(np.dot(vij, np.transpose(dwij)))
        pi.drho += np.sum(mass * vdotw)

    @classmethod
    def Gravity(self, p: Particle, gx: float, gy: float) -> None:
        p.a[0] += gx
        p.a[1] += gy

    def TaitEOS(self, pi: Particle) -> None:
        pi.p = self.B * ((pi.rho / self.rho0) ** self.gamma - 1)
