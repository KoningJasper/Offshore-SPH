import numpy as np
from math import sqrt
from numba import njit, prange
from src.Common import ParticleType
from typing import Tuple

class TimeStep:
    @staticmethod
    def compute(J: int, pA: np.array, gamma_c: float = 0.25, gamma_f: float = 0.25) -> Tuple[float, float, float]:
        """
            Computes the minimum time-step

            Parameters
            ----------

            J: int
                length of the particle array

            pA: np.array
                Particle array, should have particle_dtype as dtype

            gamma_c: float
                cfl factor for the courant condition, default 0.4

            gamma_f: float
                cfl factor for the force condition, default 0.25

            Returns
            -------

            Minimum time-step

            Time-step based on courant condition

            Time-step based on force condition
        """

        min_h, max_c, max_a2 = TimeStep.computeVars(J, pA)

        c = TimeStep.courant(gamma_c, min_h, max_c)
        f = TimeStep.force(gamma_f, min_h, max_a2)

        return min(c, f), c, f

    @staticmethod
    @njit('float64(float64, float64, float64)', fastmath=True)
    def courant(cfl, h_min, c_max) -> float:
        """ Timestep due to courant condition. """
        return cfl * h_min / c_max

    @staticmethod
    @njit('float64(float64, float64, float64)', fastmath=True)
    def force(cfl, min_h, max_a) -> float:
        """ Time-step due to force. """
        if max_a < 1e-12:
            return 1e10
        else:
            return cfl * sqrt(min_h / max_a)

    @staticmethod
    @njit(fastmath=True)
    def computeVars(J: int, pA: np.array):
        """
            Computes the minimum h, maximum speed of sound, and maximum acceleration for a given particle array.

            Parameters
            ----------

            J: int
                Length of particle array

            pA: np.array
                Particle array


            Returns
            -------

            a tuple with
                minimum h, maximum speed of sound, and maximum acceleration
        """
        h = []; c = []; a2 = []
        # Outside of compute loop so prange can be used.
        for j in prange(J):
            if pA[j]['label'] == ParticleType.Fluid:
                h.append(pA[j]['h'])
                c.append(pA[j]['c'])
                a2.append(pA[j]['ax'] * pA[j]['ax'] + pA[j]['ay'] * pA[j]['ay'])

        # Find the maximum this can not be done parallel.
        min_h  = np.min(np.array(h))
        max_c  = np.max(np.array(c))
        max_a2 = np.max(np.array(a2))

        return min_h, max_c, max_a2