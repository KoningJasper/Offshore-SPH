import numpy as np
from typing import List
from src.Methods.Method import Method
from src.Equations import Continuity, Gravity, Momentum, TaitEOS
from src.Particle import Particle

class WCSPH(Method):
    """ The WCSPH method as described by Monaghan in the 1992 free-flow paper. """
    def __init__(self, height: float, rho0: float):
        # Assign primary variables.
        self.height = height
        self.rho0   = rho0

        # Initalize the equations
        self.momentum   = Momentum.Momentum()
        self.continuity = Continuity.Continuity()
        self.taitEOS    = TaitEOS.TaitEOS(height=self.height,  rho0=self.rho0)
        self.gravity    = Gravity.Gravity(0, -9.81)

    def initialize(self, p: Particle):
        """ Initialize the particles, should only be run once at the start of the solver. """
        p.rho = self.taitEOS.initialize(p)

    def compute_acceleration(self, p: Particle, xij: np.array, rij: np.array, vij: np.array, pressure: np.array, rho: np.array, hij: np.array, cij: np.array, wij: np.array, dwij: np.array):
        # Run EOS
        a = self.momentum.calc(p.m)
        a = self.gravity.calc(a)

        return a
