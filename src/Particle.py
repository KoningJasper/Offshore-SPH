import numpy as np


class Particle:
    def __init__(self, label, x, y, rho: float = 1000):
        self.label = label
        self.r = np.array([x, y])
        self.v = np.zeros(2)
        self.a = np.zeros(2)
        self.rho = rho
        self.p = 0.
        self.drho = 0.

    r: np.array  # Position
    v: np.array  # Velocity
    a: np.array  # Acceleration
    label: str
    rho: float
    p: float
    drho: float

    def __eq__(self, other):
        return self.r[0] == other.r[0] and self.r[1] == other.r[1]
