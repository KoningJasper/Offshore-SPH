import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import unittest
import numpy as np
from src.Equations.Courant import Courant

class test_eq_courant(unittest.TestCase):
    def test(self):
        self.assertAlmostEqual(Courant(0.4, np.array([0]), np.array([1])), 0)
        self.assertAlmostEqual(Courant(0.4, np.array([1]), np.array([1])), 0.4)
        self.assertAlmostEqual(Courant(0.4, np.array([2]), np.array([4])), 0.5 * 0.4)

if __name__ == "__main__":
    unittest.test()