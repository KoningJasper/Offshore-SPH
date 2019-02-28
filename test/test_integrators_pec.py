# Add parent folder to path
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import unittest
import numpy as np
from copy import copy

from src.Common import particle_dtype, ParticleType
from src.Integrators.PEC import PEC
from src.Integrators.Euler import Euler

class test_integrators_euler(unittest.TestCase):
    def test_boundary(self):
        # Create some particle.
        i = PEC(useXSPH=False, strict=False)
        dt, p = self.some_particle(ParticleType.Boundary)

        # Verify predictor step
        p2 = i.predict(dt, p, 0)
        self.assertAlmostEqual(p2['x'], 0.0)
        self.assertAlmostEqual(p2['y'], 0.0)
        self.assertAlmostEqual(p2['rho'], 1000 + 0.5 * 10 * dt)

        # Verify corrector step.
        p2_c = i.correct(dt, p2, 0)
        self.assertAlmostEqual(p2_c['x'], 0.0)
        self.assertAlmostEqual(p2_c['y'], 0.0)
        self.assertAlmostEqual(p2_c['rho'], 1000 + 10 * dt)

    def test_fluid(self):
        # Init
        i = PEC(useXSPH=False, strict=False)
        dt, p = self.some_particle(ParticleType.Fluid)

        # Verify predictor step
        p2 = i.predict(dt, p, 0)
        self.assertAlmostEqual(p2['x'], dt * 0.5)
        self.assertAlmostEqual(p2['y'], dt * 0.5 * 3.0)
        self.assertAlmostEqual(p2['rho'], 1000 + dt * 0.5 * 10.0)

        # Verify corrector step.
        p2_c = i.correct(dt, p2, 0)
        self.assertAlmostEqual(p2_c['x'], dt * 1.0 + 0.5 * 5 * 2 * 2)
        self.assertAlmostEqual(p2_c['y'], dt * 3.0)
        self.assertAlmostEqual(p2_c['rho'], 1000 + 10 * dt)

    def test_compare_euler(self):
        """ Euler and PEC should be equal if forces are not re-computed. """
        # init
        i = PEC(useXSPH=False, strict=False); e = Euler()
        dt, p_i = self.some_particle(ParticleType.Fluid)
        p_e = copy(p_i)

        # Act
        p_i = i.predict(dt, p_i, 0)
        p_i = i.correct(dt, p_i, 0)
        p_e = e.predict(dt, p_e)
        p_e = e.correct(dt, p_e)

        # Verify
        self.assertAlmostEqual(p_i['x'], p_e['x'])
        self.assertAlmostEqual(p_i['y'], p_e['y'])
        self.assertAlmostEqual(p_i['vx'], p_e['vx'])
        self.assertAlmostEqual(p_i['vy'], p_e['vy'])
        self.assertAlmostEqual(p_i['rho'], p_e['rho'])

    def test_strict(self):
        """ Verifies if the strict enforces the criteria. """
        # init
        i = PEC(useXSPH=False, strict=True)
        dt, p_i = self.some_particle(ParticleType.Fluid)

        # Check drho
        p_i = i.predict(dt, p_i, 0)
        p_i = i.correct(dt, p_i, 0)

        self.assertAlmostEqual(p_i['rho'][0], 1000 + dt * p_i['drho'])

        # Set giant -drho
        p_i['drho'][0] = - 10_000
        p_i = i.predict(dt, p_i, 0)
        p_i = i.correct(dt, p_i, 0)

        self.assertAlmostEqual(p_i['rho'][0], 1000)

    def some_particle(self, label: int):
        dt = 2.0
        p = np.zeros(1, dtype=particle_dtype)
        p['label'] = label
        p['vx'] = 1.0
        p['vy'] = 3.0
        p['ax'] = 5.0
        p['ay'] = 0
        p['drho'] = 10
        p['rho'] = 1000
        return (dt, p)
        
if __name__ == "__main__":
    unittest.test()