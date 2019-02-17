import numpy as np
from numba import njit, prange, jitclass, boolean
from typing import List, Tuple
from src.Common import ParticleType
from src.Integrators.Integrator import Integrator

spec = [
    ('useXSPH', boolean)
]
@jitclass(spec)
class PEC():
    """
    The predictor-corrector as described by Monaghan in his 1992 paper.
    """

    def __init__(self, useXSPH: bool = True):
        self.useXSPH = useXSPH
        
    def isMultiStage(self) -> bool:
        return False
    
    def predict(self, dt: float, pA: np.array, damping: float) -> np.array:
        J = len(pA)
        for j in range(J):
            if pA[j]['label'] == ParticleType.Fluid:
                # Store noughts
                pA[j]['x0'] = pA[j]['x']
                pA[j]['y0'] = pA[j]['y']
                pA[j]['vx0'] = pA[j]['vx']
                pA[j]['vy0'] = pA[j]['vy']

                # Compute mid-point
                if self.useXSPH == True:
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
        return pA

    def correct(self, dt: float, p: np.array, damping: float) -> np.array:
        seq = [('x', 'xsphx'), ('y', 'xsphy'), ('vx', 'ax'), ('vy', 'ay'), ('rho', 'drho')]
        for end, acc in seq:
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

    def correct(self, dt: float, pA: np.array, damping: float) -> np.array:
        J = len(pA)
        for j in range(J):
            if pA[j]['label'] == ParticleType.Fluid:
                # Correct mid-point
                if self.useXSPH == True:
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
        return pA