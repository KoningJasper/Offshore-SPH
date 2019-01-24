# Add parent folder to path
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import unittest
import numpy as np
from src.Equations.TaitEOS import TaitEOS, compute_pressure_vec

class test_taiteos(unittest.TestCase):
    def test_vec(self):
        tait = TaitEOS()

        rhos = np.linspace(0, 2000, 10_000)

        p = compute_pressure_vec(rhos, tait.rho0, tait.gamma, tait.B)

        # Compute manually
        for i, rho in enumerate(rhos):
            p_s = tait.calc(rho)
            self.assertAlmostEqual(p_s, p[i])