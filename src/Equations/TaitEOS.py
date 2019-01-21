import numpy as np
from src.Particle import Particle

class TaitEOS():
    """
        TaitEOS

        Equation of state for that relates the pressure to the density of a particle.
    """

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

        # Monaghan (2002) p. 1746
        self.co = 10.0 * np.sqrt(2 * 9.81 * self.H)
        self.B = self.co * self.co * self.rho0 / self.gamma

    def initialize() -> float
        """ returns the density of particle based on height (y) of particle """
        y = p.r[1]
        frac  = self.rho0 * 9.81 * (self.H - y) / self.B
        return self.rho0 * (1 + frac) ** (1 / self.gamma)      

    def calc(self, pi: Particle) -> float:
        ratio = pi.rho / self.rho0
        temp  = ratio ** self.gamma
        pi.p  = (temp - 1.0) * self.B

        return pi.p