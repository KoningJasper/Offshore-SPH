# Add parent folder to path; for directly running the file
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import unittest
import numpy as np
from scipy.spatial.distance import cdist
from typing import List
from time import perf_counter

from src.Particle import Particle
from src.Kernels.Gaussian import Gaussian
from src.Equations.Momentum import Momentum

class test_numba_momentum(unittest.TestCase):
    alpha: float = 0.01
    beta: float = 0.0
    eta: float = 0.5

    def test(self):
        mom = Momentum()

        num = 10_000

        xij_x = np.linspace(0, 2000, num)
        xij_y = np.linspace(0, 2000, num)
        xij = np.transpose(np.vstack((xij_x, xij_y)))

        rij = cdist(xij, xij)

        dwij_x = np.linspace(0, 2000, num)
        dwij_y = np.linspace(0, 2000, num)
        dwij = np.transpose(np.vstack((dwij_x, dwij_y)))

        vij_x = np.linspace(0, 2000, num)
        vij_y = np.linspace(0, 2000, num)
        vij = np.transpose(np.vstack((vij_x, vij_y)))

        p_o = np.linspace(0, 10_000, num)
        rho_o = np.ones_like(p_o) * 1025
        h_j = np.ones_like(p_o) * 1.3
        wij = np.linspace(0, 10_000, num)
        m = np.ones(len(dwij_x))

        p = Particle('fluid', 0, 0, 1.0, 1000)

        cs = 10.0 * np.sqrt(2 * 9.81 * 1.0)
        c_j = np.ones(len(dwij_x)) * cs

        # Pre-compile
        mom.calc(1000.0, 0.0, cs, 1.3, m, rho_o, p_o, c_j, h_j, xij_x, rij[0, :], vij_x, dwij_x)

        # Calc vectorized
        start_vec = perf_counter()
        a_x = mom.calc(1000.0, 0.0, cs, 1.3, m, rho_o, p_o, c_j, h_j, xij_x, rij[0, :], vij_x, dwij_x)
        a_y = mom.calc(1000.0, 0.0, cs, 1.3, m, rho_o, p_o, c_j, h_j, xij_y, rij[0, :], vij_y, dwij_y)
        t_vec = perf_counter() - start_vec

        # Calc old
        start_old = perf_counter()
        [a_old] = self._calc_old(m, p, xij, rij[:, 0], vij, p_o, rho_o, h_j, c_j, wij, dwij, False)
        t_old = perf_counter() - start_old

        # Assert
        self.assertAlmostEqual(a_old[0], a_x)
        self.assertAlmostEqual(a_old[1], a_y)

        print(f'Timing:')
        print(f'Old: {t_old:f} [s]')
        print(f'New: {t_vec:f} [s]')
        print(f'Speed-up: {t_old / t_vec:f}x')

    def _calc_old(self, mass: float, p: Particle, xij: np.array, rij: np.array, vij: np.array, pressure: np.array, rho: np.array, hij: np.array, cij: np.array, wij: np.array, dwij: np.array, xsph: bool) -> List:
        # Average density.
        rhoij: np.array = 0.5 * (p.rho + rho)

        # Compute first (easy) part.
        tmp = p.p / (p.rho * p.rho) + pressure / (rho * rho)
        _au = np.sum(- mass * tmp * dwij[:, 0], axis=0)
        _av = np.sum(- mass * tmp * dwij[:, 1], axis=0)

        # Diffusion
        dot = np.sum(vij * xij, axis=1) # Row by row dot product
        piij = np.zeros(len(pressure))

        # Perform diffusion for masked entities
        mask       = dot < 0
        if any(mask):
            muij       = hij[mask] * dot[mask] / (rij[mask] * rij[mask] + 0.01 * hij[mask] * hij[mask])
            muij       = muij
            piij[mask] = muij * (self.beta * muij - self.alpha * cij[mask])
            piij[mask] = piij[mask] / rhoij[mask]

        # Calculate change in density.
        _au_d = np.sum(- mass * piij * dwij[:, 0])
        _av_d = np.sum(- mass * piij * dwij[:, 1])

        # XSPH
        if xsph == True:
            _xsphtmp = mass / rhoij * wij
            _xsphx = np.sum(_xsphtmp * -vij[:, 0], axis=0) # -vij = vji
            _xsphy = np.sum(_xsphtmp * -vij[:, 1], axis=0)

            return [np.array([_au + _au_d, _au + _av_d]), np.array([self.eta * _xsphx, self.eta * _xsphy])]
        else:
            return [np.array([_au + _au_d, _au + _av_d])]

if __name__ == "__main__":
    test_numba_momentum().test()