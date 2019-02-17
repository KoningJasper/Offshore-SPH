import numpy as np
from numba import njit, prange
from typing import List, Tuple
from src.Common import ParticleType
from src.Integrators.Integrator import Integrator

class PEC(Integrator):
    """
    The predictor-corrector as described by Monaghan in his 1992 paper.
    """

    seq: List[Tuple[str, str]]

    def __init__(self, useXSPH: bool = True):
        if useXSPH == True:
            # When using XSPH use xsph for moving instead of vx, vy
            self.seq = [('x', 'xsphx'), ('y', 'xsphy'), ('vx', 'ax'), ('vy', 'ay'), ('rho', 'drho')]
        else:
            self.seq = [('x', 'vx'), ('y', 'vy'), ('vx', 'ax'), ('vy', 'ay'), ('rho', 'drho')]
        
    def isMultiStage(self) -> bool:
        return False
    
    def predict(self, dt: float, p: np.array, damping: float) -> np.array:
        for end, acc in self.seq:
            # Skip if not fluid
            if (end != 'rho') and (p['label'] != ParticleType.Fluid):
                continue

            # Store nought for later retrieval.
            p[end + '0'] = p[end]

            # Compute mid-point
            if end == 'vx' or end == 'vy':
                p[end] = (p[end] + 0.5 * dt * p[acc]) / (1 + 0.5 * damping)
            else:
                p[end] = p[end] + 0.5 * dt * p[acc]
        # End for
        return p

    def correct(self, dt: float, p: np.array, damping: float) -> np.array:
        for end, acc in self.seq:
            # Skip if not fluid
            if (end != 'rho') and (p['label'] != ParticleType.Fluid):
                continue

            # Correct the midpoint
            if end == 'vx' or end == 'vy':
                mid = (p[end + '0'] + 0.5 * dt * p[acc]) / (1 + 0.5 * damping)
            else:
                mid = p[end + '0'] + 0.5 * dt * p[acc]

            # Compute end
            p[end] = 2 * mid - p[end + '0']
        return p