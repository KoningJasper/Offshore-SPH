import numpy as np
from numba import njit, jit, float64, jitclass, prange
from typing import List
from src.Common import _stack
from src.Methods.Method import Method
from src.Equations import Continuity, BodyForce, Momentum, TaitEOS, XSPH
from src.Particle import Particle

spec = [
    ('useXSPH', float64),
    ('height', float64),
    ('rho0', float64),

    # XSPH
    ('epsilon', float64),

    # Tait
    ('gamma', float64),
    ('co', float64),
    ('B', float64),

    # Momentum
    ('alpha', float64),
    ('beta', float64)
]
@jitclass(spec)
class WCSPH(Method):
    """ The WCSPH method as described by Monaghan in the 1992 free-flow paper. """
    def __init__(self, height, rho0, num_particles, useXSPH = 1.0):
        """
            useXSPH has to be enabled in both the integrator and the method!
        """
        # Assign primary variables.
        self.height  = height

        self.rho0    = rho0
        self.useXSPH = useXSPH

        # Assign parameters
        self.epsilon = 0.5
        
        self.gamma = 7.0
        self.co    = TaitEOS.TaitEOS_co(self.height)
        self.B     = TaitEOS.TaitEOS_B(self.co, self.rho0, self.gamma)

        self.alpha = 0.01
        self.beta  = 0.0

    def getOptions(self):
        return [0.0]

    # Override
    def initialize(self, pA):
        """ Initialize the particles, should only be run once at the start of the solver. """
        rho = TaitEOS.TaitEOS_height(
            self.rho0,
            self.height,
            self.B,
            self.gamma,
            pA['y']
        )

        for j in prange(len(pA)):
            pA[j]['rho'] = rho[j]

        return pA

    def compute_speed_of_sound(self, pA):
        # Speed of sound is a constant with Tait, so just return that.
        J = len(pA)
        cs = np.zeros(J, dtype=np.float64)
        for j in prange(J):
            cs[j] = self.co
        return cs

    def compute_pressure(self, pA: np.array):
        return TaitEOS.TaitEOS(
            self.gamma,
            self.B,
            self.rho0,
            pA['rho']
        )

    # Override
    def compute_acceleration(self, p, comp):
        # Momentum
        [a_x, a_y] = Momentum.Momentum(
            self.alpha,
            self.beta,
            p,
            comp
        )

        # Gravity
        return [a_x, a_y - 9.81]

    # Override
    def compute_velocity(self, p, comp):
        if self.useXSPH > 0.0:
            # Compute XSPH Term
            [xsph_x, xsph_y] = XSPH.XSPH(self.epsilon, p, comp)

            # Velocity stays the same, xsph correction is changed.
            return [p['vx'], p['vy'], p['vx'] + xsph_x, p['vy'] + xsph_y]
        else:
            # Velocity stays the same
            return [p['vx'], p['vy'], p['vx'], p['vy'],]

    # override
    def compute_density_change(self, p, comp):
        return Continuity.Continuity(p, comp)

