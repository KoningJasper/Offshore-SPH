import numpy as np
from numba import njit, prange, jitclass, boolean
from typing import List, Tuple
from src.Common import ParticleType
from src.Integrators.Integrator import Integrator

@jitclass([])
class PEC():
    def __init__(self):
        """
            The predictor-corrector as described by Monaghan in his 1992 paper.
        """
        
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
            pA[j]['rho0'] = pA[j]['rho']

            pA[j]['x'] = pA[j]['x0'] + 0.5 * dt * pA[j]['vx']
            pA[j]['y'] = pA[j]['y0'] + 0.5 * dt * pA[j]['vy']

            pA[j]['vx']  = pA[j]['vx0'] + 0.5 * dt * pA[j]['ax']
            pA[j]['vy']  = pA[j]['vy0'] + 0.5 * dt * pA[j]['ay']
            pA[j]['rho'] = pA[j]['rho0'] + 0.5 * dt * pA[j]['drho']

        return pA

    def correct(self, dt: float, pA: np.array, damping: float) -> np.array:
        J = len(pA)
        # Outside of compute loop so prange can be used.
        for j in prange(J):
            pA[j]['x'] = pA[j]['x0'] + dt * pA[j]['vx']
            pA[j]['y'] = pA[j]['y0'] + dt * pA[j]['vy']


            pA[j]['vx']  = pA[j]['vx0'] + dt * pA[j]['ax']
            pA[j]['vy']  = pA[j]['vy0'] + dt * pA[j]['ay']
            pA[j]['rho'] = pA[j]['rho0'] + dt * pA[j]['drho']

        return pA