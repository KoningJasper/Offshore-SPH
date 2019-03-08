import numpy as np
from numba import jitclass, prange
from src.Common import ParticleType
from src.Integrators.Integrator import Integrator

@jitclass([])
class Euler(Integrator):
    """ Stupidly simple Euler Integrator """

    def isMultiStage(self) -> bool:
        return False

    def predict(self, dt: float, pA: np.array, damping: float) -> np.array:
        """ Predict does nothing in eurler-integrator. """
        return pA

    def correct(self, dt: float, pA: np.array, damping: float) -> np.array:
        # Outside of compute loop so prange can be used.
        for j in prange(len(pA)):
            pA[j]['x'] = pA[j]['x'] + dt * pA[j]['vx'] + 0.5 * dt * dt * pA[j]['ax']
            pA[j]['y'] = pA[j]['y'] + dt * pA[j]['vy'] + 0.5 * dt * dt * pA[j]['ay']

            pA[j]['vx'] = pA[j]['vx'] + dt * pA[j]['ax']
            pA[j]['vy'] = pA[j]['vy'] + dt * pA[j]['ay']

            pA[j]['rho'] = pA[j]['rho'] + dt * pA[j]['drho']
        return pA