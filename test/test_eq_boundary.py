import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import unittest
import numpy as np
from src.Equations.BoundaryForce import BoundaryForce
from src.Common import ParticleType, particle_dtype, computed_dtype

class test_eq_boundary(unittest.TestCase):
    def test(self):
        # Create self; (0, 0)
        p = np.zeros(1, dtype=particle_dtype)[0]
        p['label'] = ParticleType.Fluid

        # Create other
        o = np.zeros(2, dtype=computed_dtype)
        o[0]['label'] = ParticleType.Boundary
        o[0]['x'] = 1 # (-1, 0)
        o[0]['r'] = 1
        o[1]['label'] = ParticleType.Fluid
        o[1]['x'] = -1 # (1, 0)
        o[1]['r'] = 1

        # Calc some forces
        [ax, ay] = BoundaryForce(r0=2.0, D=5 * 9.81 * 1.0, p1=4, p2=2, p=p, comp=o)
        self.assertGreater(ax, 0)
        self.assertEqual(ay, 0)

if __name__ == "__main__":
    unittest.test()