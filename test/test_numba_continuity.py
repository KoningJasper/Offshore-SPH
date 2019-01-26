# Add parent folder to path
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import unittest
import numpy as np
from time import perf_counter
from src.Equations.Continuity import Continuity

class test_numba_continuity(unittest.TestCase):
    def test_vec(self):
        cc = Continuity()

        dwij_x = np.linspace(0, 2000, 10_000_000)
        dwij_y = np.linspace(0, 2000, 10_000_000)
        dwij = np.transpose(np.vstack((dwij_x, dwij_y)))

        vij_x = np.linspace(0, 2000, 10_000_000)
        vij_y = np.linspace(0, 2000, 10_000_000)
        vij = np.transpose(np.vstack((vij_x, vij_y)))

        m = np.ones(len(dwij_x))

        # Calc vectorized
        start_vec = perf_counter()
        arho = cc.calc(m, dwij_x, vij_x) + cc.calc(m, dwij_y, vij_y)
        t_vec = perf_counter() - start_vec

        # Calc old
        start_old = perf_counter()
        arho_o = self.old_calc(m[0], dwij, vij)
        t_old = perf_counter() - start_old

        # Assert
        self.assertAlmostEqual(arho_o / arho, 1)

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