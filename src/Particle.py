import numpy as np


class Particle:
    def __init__(self, label: str, x: float, y: float, mass: float, rho: float = 1000):
        self.label = label
        self.r = np.array([x, y])
        self.m = mass
        self.rho = rho

    """ Mass of the particle. """
    m: float

    """ Current position of the particle. """
    r: np.array

    """ Current velocity of the particle """
    v: np.array = np.zeros(2)

    """ XSPH correction velocity of particle. """
    vx: np.array = np.zeros(2)

    """ Acceleration of the particle. """
    a: np.array = np.zeros(2)

    """ Custom label for the particle, generally either fluid or boundary. """
    label: str

    """ Density of the particle. """
    rho: float

    """ Pressure of the particle. """
    p: float = 0.

    """ Density change of the particle. """
    drho: float = 0.

    def __eq__(self, other):
        return self.r[0] == other.r[0] and self.r[1] == other.r[1]
