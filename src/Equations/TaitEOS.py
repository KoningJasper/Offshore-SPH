import numpy as np
import math
from numba import vectorize, jit
from src.Particle import Particle

class TaitEOS():
    """
        TaitEOS
        -------
        
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
        """
            TaitEOS
            -------
            
            Equation of state for that relates the pressure to the density of a particle.
        """
        self.gamma = gamma
        self.rho0 = rho0
        self.H = height

        # Monaghan (2002) p. 1746
        self.co = 10.0 * np.sqrt(2 * 9.81 * self.H)
        self.B = self.co * self.co * self.rho0 / self.gamma

    def initialize(self, y: float) -> float:
        """ returns the density of particle based on height (y) of particle """
        frac  = self.rho0 * 9.81 * (self.H - y) / self.B
        return self.rho0 * (1 + frac) ** (1 / self.gamma)      

    def calc(self, rho: np.array) -> np.array:
        return _compute_pressure_vec(rho, self.rho0, self.gamma, self.B)


@vectorize(['float64(float64, float64, float64, float64)'], target='parallel')
def _compute_pressure_vec(rho, rho0, gamma, B):
    ratio = (rho / rho0) ** gamma
    return (ratio - 1.0) * B