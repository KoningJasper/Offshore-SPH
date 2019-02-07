import numpy as np
from typing import List
from src.Methods.Method import Method
from src.Equations import Continuity, Gravity, Momentum, TaitEOS, XSPH
from src.Particle import Particle

class WCSPH(Method):
    """ The WCSPH method as described by Monaghan in the 1992 free-flow paper. """
    useXSPH: bool

    def __init__(self, height: float, rho0: float, num_particles: int, useXSPH: bool = True):
        # Assign primary variables.
        self.height  = height
        self.rho0    = rho0
        self.useXSPH = useXSPH

        # Initalize the equations
        self.momentum   = Momentum.Momentum() # Default alpha, beta and eta parameters
        self.continuity = Continuity.Continuity()
        self.xsph       = XSPH.XSPH()
        self.taitEOS    = TaitEOS.TaitEOS(height=self.height,  rho0=self.rho0)
        self.gravity    = Gravity.Gravity(0, -9.81)

    # Override
    def initialize(self, p: np.array) -> np.array:
        """ Initialize the particles, should only be run once at the start of the solver. """
        p['rho'] = self.taitEOS.initialize(p['y'])
        return p

    def compute_speed_of_sound(self, p: np.array) -> float:
        return self.taitEOS.co

    def compute_pressure(self, p: np.array) -> float:
        return self.taitEOS.calc(p['rho'])

    # Override
    def compute_acceleration(self, p: np.array, comp: np.array) -> List[float]:
        # Concat arrays
        xij = WCSPH._stack(comp['x'], comp['y'])
        vij = WCSPH._stack(comp['vx'], comp['vy'])
        dwij = WCSPH._stack(comp['dw_x'], comp['dw_y'])

        # Momentum
        [a_x, a_y] = self.momentum.calc(p['rho'], p['p'], p['c'], p['h'], comp['m'], comp['rho'], comp['p'], comp['c'], comp['h'], xij, comp['r'], vij, dwij)
        a = np.array([a_x, a_y])

        # Gravity
        a = self.gravity.calc(a)

        return [a[0], a[1]]

    # Override
    def compute_velocity(self, p: np.array, comp: np.array) -> List[float]:
        if self.useXSPH == True:
            # Concat array
            vij = WCSPH._stack(comp['vx'], comp['vy'])

            # Compute XSPH Term
            [xsph_x, xsph_y] = self.xsph.calc(p['rho'], comp['m'], comp['rho'], vij, comp['w'])

            # Calc new velocity
            return [p['vx'] + xsph_x, p['vy'] + xsph_y]
        else:
            return [p['vx'], p['vy']]

    # override
    def compute_density_change(self, p: np.array, comp: np.array) -> float:
        # Concat the arrays
        vij = WCSPH._stack(comp['vx'], comp['vy'])
        dwij = WCSPH._stack(comp['dw_x'], comp['dw_y'])
        return self.continuity.calc(comp['m'], vij, dwij)

    @staticmethod
    def _stack(m1, m2):
        return np.transpose(np.vstack((m1, m2)))