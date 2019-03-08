import numpy as np
from numba import njit, jit, float64, jitclass, prange, boolean
from typing import List
from src.Common import _stack
from src.Methods.Method import Method
from src.Equations import Continuity, BodyForce, Momentum, TaitEOS, XSPH, BoundaryForce
from src.Particle import Particle

spec = [
    ('useXSPH', boolean),
    ('height', float64),
    ('rho0', float64),

    # Lennard-Jones BoundaryForce
    ('r0', float64),
    ('D', float64),
    ('p1', float64),
    ('p2', float64),

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
    def __init__(self, height: float, r0: float, rho0: float, useXSPH: bool):
        """
            The WCSPH method as described by Monaghan in the 1992 free-flow paper.

            Parameters
            ----------
            height: float
                Maximum height of the fluid.
            r0: float
                Particle separation
            rho0: float
                Initial particle/fluid density
            useXSPH: bool
                Should XSPH correction be used. XSPH modifies the velocity of the particles to be an averaged property.

            Notes
            -----
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

        # Compute LJ BoundaryForce parameters
        self.r0 = r0
        self.D  = 5 * 9.81 * self.height
        self.p1 = 4
        self.p2 = 2

    # Override
    def initialize(self, pA: np.array):
        """
            Initialize the particles, should only be run once at the start of the solver. 

            Parameters
            ----------
            pA: np.array
                Particle array, all particles

            Returns
            -------
            pA: np.array
                Initialized particle array, all particles.
        """
        rho = TaitEOS.TaitEOS_height(
            self.rho0,
            self.height,
            self.B,
            self.gamma,
            pA['y']
        )

        # Outside of compute loop so prange can be used.
        for j in prange(len(pA)):
            pA[j]['rho'] = rho[j]

        return pA

    def compute_speed_of_sound(self, pA: np.array) -> np.array:
        """
            Parameters
            ----------
            pA: np.array
                Particle array, all particles

            Returns
            -------
            cs: np.array
                Array of speeds of sound, length N.
        """
        # Speed of sound is a constant with Tait, so just return that.
        J = len(pA)
        cs = np.zeros(J, dtype=np.float64)
        
        # Outside of compute loop so prange can be used.
        for j in prange(J):
            cs[j] = self.co
        return cs

    def compute_pressure(self, pA: np.array) -> np.array:
        """
            Parameters
            ----------
            pA: np.array
                Particle array, all particles

            Returns
            -------
            p: np.array
                Array with pressure for each of the particles
        """
        return TaitEOS.TaitEOS(
            self.gamma,
            self.B,
            self.rho0,
            pA['rho']
        )

    def compute_acceleration(self, p: np.array, comp: np.array):
        """
            Parameters
            ----------
            p: np.array
                Single particle
            comp:
                Computed properties of near particles.
        """
        # Momentum
        [a_x, a_y] = Momentum.Momentum(
            self.alpha,
            self.beta,
            p,
            comp
        )

        # Compute boundary forces
        [b_x, b_y] = BoundaryForce.BoundaryForce(self.r0, self.D, self.p1, self.p2, p, comp)

        # Gravity
        return [a_x + b_x, a_y + b_y - 9.81]

    def compute_velocity(self, p: np.array, comp: np.array):
        """
            Parameters
            ----------
            p: np.array
                Single particle
            comp:
                Computed properties of near particles.
        """
        if self.useXSPH == True:
            # Compute XSPH Term
            [xsph_x, xsph_y] = XSPH.XSPH(self.epsilon, p, comp)

            # Velocity stays the same, xsph correction is changed.
            return [p['vx'], p['vy'], p['vx'] + xsph_x, p['vy'] + xsph_y]
        else:
            # Velocity stays the same
            return [p['vx'], p['vy'], p['vx'], p['vy'],]

    def compute_density_change(self, p: np.array, comp: np.array):
        """
            Parameters
            ----------
            p: np.array
                Single particle
            comp:
                Computed properties of near particles.
        """
        return Continuity.Continuity(p, comp)

