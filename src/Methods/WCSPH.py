import numpy as np
from numba import jitclass, njit
from numba import int64, float64, boolean
from typing import List
from src.Common import _stack
from src.Methods.Method import Method
from src.Equations import Continuity, Gravity, Momentum, TaitEOS, XSPH
from src.Particle import Particle

spec = [
    ('useXSPH', boolean),
    ('num_particles', int64),
    ('height', float64),
    ('rho0', float64),

    # XSPH
    ('epsilon', float64), # XSPH

    # Tait
    ('co', float64),
    ('', float64),
    ('', float64)

]
@jitclass(spec)
class WCSPH(Method):
    """ The WCSPH method as described by Monaghan in the 1992 free-flow paper. """
    def __init__(self, height, rho0, num_particles, useXSPH = True):
        """
            useXSPH has to be enabled in both the integrator and the method!
        """
        # Assign primary variables.
        self.height  = height
        self.rho0    = rho0
        self.useXSPH = useXSPH

        # Initalize the equations
        # self.momentum   = Momentum.Momentum() # Default alpha, beta and eta parameters
        # self.continuity = Continuity.Continuity()
        # self.xsph       = XSPH.XSPH()
        # self.taitEOS    = TaitEOS.TaitEOS(height=self.height,  rho0=self.rho0)
        # self.gravity    = Gravity.Gravity(0, -9.81)

    # Override
    def initialize(self, p):
        """ Initialize the particles, should only be run once at the start of the solver. """
        p['rho'] = TaitEOS.TaitEOS(self.height, self.rho0, 7.0).initialize(p['y'])
        return p

    def compute_speed_of_sound(self, p):
        return TaitEOS.TaitEOS(height=self.height, rho0=self.rho0).co

    def compute_pressure(self, p):
        return TaitEOS.TaitEOS(height=self.height, rho0=self.rho0).calc(p['rho'])

    # Override
    @staticmethod
    @njit(parallel=True, fastmath=True)
    def compute_acceleration(self, p, comp):
        # Concat arrays
        xij = _stack(comp['x'], comp['y'])
        vij = _stack(comp['vx'], comp['vy'])
        dwij = _stack(comp['dw_x'], comp['dw_y'])

        # Momentum
        [a_x, a_y] = Momentum.Momentum().calc(p['rho'], p['p'], p['c'], p['h'], comp['m'], comp['rho'], comp['p'], comp['c'], comp['h'], xij, comp['r'], vij, dwij)
        a = np.array([a_x, a_y])

        # Gravity
        a = Gravity.Gravity(0, -9.81).calc(a)

        return [a[0], a[1]]

    # Override
    @staticmethod
    @njit(parallel=True, fastmath=True)
    def compute_velocity(self, p, comp):
        if self.useXSPH == True:
            # Concat array
            vij = _stack(comp['vx'], comp['vy'])

            # Compute XSPH Term
            [xsph_x, xsph_y] = XSPH.XSPH().calc(p['rho'], comp['m'], comp['rho'], vij, comp['w'])

            # Velocity stays the same, xsph correction is changed.
            return [p['vx'], p['vy'], p['vx'] + xsph_x, p['vy'] + xsph_y]
        else:
            # Velocity stays the same
            return [p['vx'], p['vy'], 0.0, 0.0]

    # override
    def compute_density_change(self, p, comp):
        # Concat the arrays
        vij = _stack(comp['vx'], comp['vy'])
        dwij = _stack(comp['dw_x'], comp['dw_y'])
        return Continuity.Continuity().calc(comp['m'], vij, dwij)
