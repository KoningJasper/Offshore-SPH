import numpy as np
from numba import njit, jit, float64, jitclass, prange
from typing import List
from src.Common import _stack
from src.Methods.Method import Method
from src.Equations import Continuity, BodyForce, Momentum, TaitEOS, XSPH

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
    def initialize(self, pA: np.array):
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

    def compute_speed_of_sound(self, pA: np.array):
        # Speed of sound is a constant with Tait, so just return that.
        J = len(pA)
        cs = np.zeros(J, dtype=np.float64)
        for j in prange(J):
            cs[j] = self.co
        return cs

    def compute_pressure(self, pA: np.array):
        """
            Computes the pressure based on the Tait EOS.

            Parameters
            ----------

            pA: np.array
                complete particle array of type particle_dtype

            Returns
            -------
            p: np.array
                Array of pressures
        """
        return TaitEOS.TaitEOS(
            self.gamma,
            self.B,
            self.rho0,
            pA['rho']
        )

    def compute_acceleration(self, p: np.array, comp: np.array):
        """
            Computes the acceleration of the particle using Momentum Equation. Includes gravity of 9.81 m/s in negative y-direction.

            Parameters
            ----------
            p: np.array
                Single particle for which to compute the momentum for.
            comp: np.array
                Computed properties of neighbouring particles that interact with particle p.

            Returns
            -------
            a_x: float
                acceleration in x-direction for particle p.
            a_y: float
                acceleration in y-direction for particle p, including gravity (-9.81).
        """
        # Momentum
        [a_x, a_y] = Momentum.Momentum(
            self.alpha,
            self.beta,
            pA,
            comp
        )

        # Gravity
        return [a_x, a_y - 9.81]

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

