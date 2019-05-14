import numpy as np
from numba import njit, prange, jitclass, boolean
from typing import List, Tuple
from src.Common import ParticleType
from src.Integrators.Integrator import Integrator

spec = [
    ('useXSPH', boolean),
    ('strict', boolean)
]
@jitclass(spec)
class PEC():
    def __init__(self, useXSPH: bool = True, strict: bool = True):
        """
            The predictor-corrector as described by Monaghan in his 1992 paper.

            Parameters
            ----------

            useXSPH: bool
                Should XSPH correction be used.

            strict: bool
                Should the density be always positive.
        """
        self.useXSPH = useXSPH
        self.strict  = strict
        
    def isMultiStage(self) -> bool:
        return False
    
    def predict(self, dt: float, pA: np.array, damping: float) -> np.array:
        J = len(pA)
        # Outside of compute loop so prange can be used.
        for j in prange(J):
            # Store noughts
            pA[j]['x0'] = pA[j]['x']
            pA[j]['y0'] = pA[j]['y']
            pA[j]['vx0'] = pA[j]['vx']
            pA[j]['vy0'] = pA[j]['vy']

            # Compute mid-point
            if (self.useXSPH == True) and (pA[j]['label'] == ParticleType.Fluid):
                pA[j]['x'] = pA[j]['x'] + 0.5 * dt * pA[j]['xsphx']
                pA[j]['y'] = pA[j]['y'] + 0.5 * dt * pA[j]['xsphy']
            else:
                pA[j]['x'] = pA[j]['x'] + 0.5 * dt * pA[j]['vx']
                pA[j]['y'] = pA[j]['y'] + 0.5 * dt * pA[j]['vy']

            pA[j]['vx'] = (pA[j]['vx'] + 0.5 * dt * pA[j]['ax']) / (1 + 0.5 * damping)
            pA[j]['vy'] = (pA[j]['vy'] + 0.5 * dt * pA[j]['ay']) / (1 + 0.5 * damping)
            
            # Do these for all
            pA[j]['rho0'] = pA[j]['rho']
            pA[j]['rho']  = pA[j]['rho'] + 0.5 * dt * pA[j]['drho']

            if self.strict and pA[j]['rho'] < 0.0:
                pA[j]['rho'] = 0.0
        return pA

    def correct(self, dt: float, pA: np.array, damping: float) -> np.array:
        J = len(pA)
        # Outside of compute loop so prange can be used.
        for j in prange(J):
            # Correct mid-point
            if (self.useXSPH == True) and (pA[j]['label'] == ParticleType.Fluid):
                mid_x = pA[j]['x0'] + 0.5 * dt * pA[j]['xsphx']
                mid_y = pA[j]['y0'] + 0.5 * dt * pA[j]['xsphy']
            else:
                mid_x = pA[j]['x0'] + 0.5 * dt * pA[j]['vx']
                mid_y = pA[j]['y0'] + 0.5 * dt * pA[j]['vy']

            mid_vx = (pA[j]['vx0'] + 0.5 * dt * pA[j]['ax']) / (1 + 0.5 * damping)
            mid_vy = (pA[j]['vy0'] + 0.5 * dt * pA[j]['ay']) / (1 + 0.5 * damping)

            # Compute final
            pA[j]['x'] = 2 * mid_x - pA[j]['x0']
            pA[j]['y'] = 2 * mid_y - pA[j]['y0']
            pA[j]['vx'] = 2 * mid_vx - pA[j]['vx0']
            pA[j]['vy'] = 2 * mid_vy - pA[j]['vy0']
            
            # Do these for all
            mid_rho = pA[j]['rho0'] + 0.5 * dt * pA[j]['drho']
            pA[j]['rho'] = 2 * mid_rho - pA[j]['rho0']

            # Enforce density
            if self.strict and pA[j]['rho'] < 0.0:
                pA[j]['rho'] = 0.0
        return pA