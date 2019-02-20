# Add parent folder to path
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import unittest
import numpy as np
from time import perf_counter
from src.Equations.Continuity import Continuity
from src.Common import computed_dtype

class test_numba_continuity(unittest.TestCase):
    def test_vec(self):
        dwij_x = np.linspace(0, 2000, 10_000_000)
        dwij_y = np.linspace(0, 2000, 10_000_000)
        dwij = np.transpose(np.vstack((dwij_x, dwij_y)))

        vij_x = np.linspace(0, 2000, 10_000_000)
        vij_y = np.linspace(0, 2000, 10_000_000)
        vij = np.transpose(np.vstack((vij_x, vij_y)))

        m = np.ones(len(dwij_x))

        # Create particle arrays
        p = np.array([])
        comp = np.zeros_like(dwij_x, dtype=computed_dtype)
        comp['m'] = m
        comp['vx'] = vij_x
        comp['vy'] = vij_y
        comp['dw_x'] = dwij_x
        comp['dw_y'] = dwij_y

        # Pre-compile
        Continuity(p, comp)

        # Calc vectorized
        start_vec = perf_counter()
        arho = Continuity(p, comp)
        t_vec = perf_counter() - start_vec

        # Calc old
        start_old = perf_counter()
        arho_o = self.old_calc(m[0], dwij, vij)
        t_old = perf_counter() - start_old

        # Assert
        self.assertAlmostEqual(arho_o / arho, 1)

        print('Completed Continuity')
        print(f'Timing:')
        print(f'Old: {t_old:f} [s]')
        print(f'New: {t_vec:f} [s]')
        print(f'Speed-up: {t_old / t_vec:f}x')

    def old_calc(self, mass: float, dwij: np.array, vij: np.array) -> float:
        """
            SPH continuity equation; Calculates the change is density of the particles.
        """
        # Init
        _arho = 0.0

        # Calc change in density
        vijdotwij = np.sum(vij * dwij, axis=1) # row by row dot product
        _arho = np.sum(mass * vijdotwij, axis=0)

        return _arho

if __name__ == "__main__":
    test_numba_continuity().test_vec()