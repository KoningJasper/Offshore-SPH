import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import unittest
import numpy as np
from src.Equations.BoundaryForce import BoundaryForce
from src.Common import get_label_code, particle_dtype, computed_dtype

class test_eq_boundary(unittest.TestCase):
    def test(self):
        f = BoundaryForce(r0=2.0, D=5 * 9.81 * 1.0)
        
        # Create self; (0, 0)
        p = np.zeros(1, dtype=particle_dtype)[0]
        p['label'] = get_label_code('fluid')

        # Create other
        o = np.zeros(2, dtype=computed_dtype)
        o[0]['label'] = get_label_code('boundary')
        o[0]['x'] = 1 # (-1, 0)
        o[0]['r'] = 1
        o[1]['label'] = get_label_code('fluid')
        o[1]['x'] = -1 # (1, 0)
        o[1]['r'] = 1

        # Calc some forces
        p = f.calc(p, o)
        self.assertGreater(p['ax'], 0)
        self.assertEqual(p['ay'], 0)

if __name__ == "__main__":
    unittest.test()