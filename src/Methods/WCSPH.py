import numpy as np
from typing import List
from src.Methods.Method import Method
from src.Equations import Continuity, Gravity, Momentum, TaitEOS
from src.Particle import Particle

class WCSPH(Method):
    """ The WCSPH method as described by Monaghan in the 1992 free-flow paper. """

    xsphs: List[np.array] = []
    def __init__(self, height: float, rho0: float, num_particles: int):
        # Assign primary variables.
        self.height = height
        self.rho0   = rho0

        # Initalize the equations
        self.momentum   = Momentum.Momentum() # Default alpha, beta and eta parameters
        self.continuity = Continuity.Continuity()
        self.taitEOS    = TaitEOS.TaitEOS(height=self.height,  rho0=self.rho0)
        self.gravity    = Gravity.Gravity(0, -9.81)

        # Create empty list
        self.xsphs = [None] * num_particles

    # Override
    def initialize(self, p: Particle) -> Particle:
        """ Initialize the particles, should only be run once at the start of the solver. """
        p.rho = self.taitEOS.initialize(p)
        return p

    def compute_speed_of_sound(self, p: Particle) -> float:
        return self.taitEOS.co

    def compute_pressure(self, p: Particle) -> float:
        return self.taitEOS.calc(p)

    # Override
    def compute_acceleration(self, i: int, p: Particle, xij: np.array, rij: np.array, vij: np.array, pressure: np.array, rho: np.array, hij: np.array, cij: np.array, wij: np.array, dwij: np.array) -> np.array:
        # Momentum
        [a, xsph] = self.momentum.calc(p.m, p, xij, rij, vij, pressure, rho, hij, cij, wij, dwij, True)

        # Gravity
        a = self.gravity.calc(a)

        # Store xsph for later retrieval
        self.xsphs[i] = xsph

        return a

    # Override
    def compute_velocity(self, i: int, p: Particle) -> np.array:
        # Retrieve the stored xsph-velocity, already contains the eta factor.
        xsph = self.xsphs[i]

        # Calc new velocity
        return p.v + xsph

    # override
    def compute_density_change(self, p: Particle, vij: np.array, dwij: np.array) -> float:
        return self.continuity.calc(p.m, dwij, vij)
