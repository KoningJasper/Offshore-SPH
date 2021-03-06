# Add parent folder to path
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import unittest
import numpy as np
from time import perf_counter
from src.Equations.TaitEOS import TaitEOS, TaitEOS_B, TaitEOS_co

class test_numba_taiteos(unittest.TestCase):
    def test_vec(self):
        rhos = np.linspace(0, 2000, 1_000_000)

        # thingies
        rho0 = 1000.0
        gamma = 7
        co = TaitEOS_co(H=2.0)
        B = TaitEOS_B(co=co, rho0=rho0, gamma=gamma)

        # Calc vectorized
        start_vec = perf_counter()
        p = TaitEOS(gamma, B, rho0, rhos)
        t_vec = perf_counter() - start_vec

        t_old = 0
        for i, rho in enumerate(rhos):
            # Old
            start_old = perf_counter()
            p_s = self.old_calc(rho, rho0, gamma, B)
            t_old += perf_counter() - start_old

            # Verify
            self.assertAlmostEqual(p_s, p[i], 3)

        print('Completed TaitEOS')
        print(f'Timing:')
        print(f'Old: {t_old:f} [s]')
        print(f'New: {t_vec:f} [s]')
        print(f'Speed-up: {t_old / t_vec:f}x')

    def old_calc(self, rho: float, rho0: float, gamma: float, B: float) -> float:
        ratio = rho / rho0
        temp  = ratio ** gamma
        p  = (temp - 1.0) * B

        return p

if __name__ == "__main__":
    test_numba_taiteos().test_vec()