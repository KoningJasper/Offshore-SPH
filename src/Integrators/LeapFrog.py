import numpy as np
import numba

spec([
    ('useXSPH', numba.boolean)
])
@numba.jitclass(spec)
class Verlet():
    def __init__(useXSPH: bool):
        """ 
            Monaghan symplectic Verlet integrator using drift-kick-drift.

            J. Monaghan, "Smoothed Particle Hydrodynamics", Reports on
            Progress in Physics, 2005
            
            Parameters
            ----------

            useXSPH: bool
                Should XSPH correction be used.
        """
        self.useXSPH = useXSPH

    def isMultiStage(self) -> bool:
        """ Only a single evaluate is used. """
        return False

    def predict(self, dt: float, pA: np.array, damping: float) -> np.array:
        J = len(pA)
        # Outside of compute loop so prange can be used.
        for j in prange(J):
            # Update position based on current velocity.
            pA[j]['x'] += 0.5 * dt * pA[j]['vx']
            pA[j]['y'] += 0.5 * dt * pA[j]['vy']

        return pA

    def correct(self, dt: float, pA: np.array, damping: float) -> np.array:
        J = len(pA)
        # Outside of compute loop so prange can be used.
        for j in prange(J):
            # Update velocity based on newly evaluated force/acceleration.
            pA[j]['vx'] += dt * pA['ax']
            pA[j]['vy'] += dt * pA['ay']

            # Update position
            if (self.useXSPH == True):
                # Use XSPH
                pA[j]['x'] += 0.5 * dt * pA[j]['xsphx']
                pA[j]['y'] += 0.5 * dt * pA[j]['xsphy']
            else:
                # Just use raw velocity
                pA[j]['x'] += 0.5 * dt * pA[j]['vx']
                pA[j]['y'] += 0.5 * dt * pA[j]['vy']

        return pA