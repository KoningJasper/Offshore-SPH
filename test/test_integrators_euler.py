# Add parent folder to path
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import unittest
import numpy as np

from src.Common import particle_dtype, ParticleType
from src.Integrators.Euler import Euler

class test_integrators_euler(unittest.TestCase):
    def test_fluid(self):
        i = Euler()

        dt = 2.0
        p = np.zeros(1, dtype=particle_dtype)
        p = p[0]
        p['label'] = ParticleType.Boundary
        p['vx'] = 1.0
        p['vy'] = 3.0
        p['ax'] = 5.0
        p['ay'] = 0
        p['drho'] = 10
        p2 = i.correct(dt, p)

        # Verify
        self.assertAlmostEqual(p2['x'], 0.0)
        self.assertAlmostEqual(p2['y'], 0.0)
        self.assertAlmostEqual(p2['rho'], 10 * dt)

        # Now as a fluid
        p['rho'] = 0.0
        p['label'] = ParticleType.Fluid
        p3 = i.correct(dt, p)

        # Verify
        self.assertAlmostEqual(p3['x'], 1.0 * dt + 0.5 * 5.0 * dt * dt)
        self.assertAlmostEqual(p3['y'], 3.0 * dt)
        self.assertAlmostEqual(p3['rho'], 10 * dt)
        
if __name__ == "__main__":
    unittest.test()