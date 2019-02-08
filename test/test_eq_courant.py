import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import unittest
from src.Equations.Courant import Courant

class test_eq_courant(unittest.TestCase):
    def test(self):
        c = Courant()
        self.assertAlmostEqual(c.alpha, 0.4)
        self.assertAlmostEqual(c.calc(0, 1), 0)
        self.assertAlmostEqual(c.calc(1, 1), 0.4)
        self.assertAlmostEqual(c.calc(2, 4), 0.5 * 0.4)

if __name__ == "__main__":
    unittest.test()