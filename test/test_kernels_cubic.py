import unittest
import numpy as np

# Add parent folder to path
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from src.Kernels.CubicSpline import CubicSpline

class test_kernels_cubic(unittest.TestCase):
    def test_evaluate(self):
        k = CubicSpline()
        r = np.array([1.0])

        # Case 1
        q = 0.5
        h = 2.0
        kv = k.evaluate(r, np.array([h]))[0]
        self.assertAlmostEqual(kv, self.calc_fac(h) * (1 - 1.5 * q * q * (1.0 - 0.5 * q)))

        q = 1.0
        h = 1.0
        kv = k.evaluate(r, np.array([h]))[0]
        self.assertAlmostEqual(kv, self.calc_fac(h) * (1 - 1.5 * 0.5))

        # Case 2
        q = 1.5
        h = 2 / 3
        kv = k.evaluate(r, np.array([h]))[0]
        self.assertAlmostEqual(kv, self.calc_fac(h) * (0.25 * pow(2 - q, 3)))

        q = 2.0
        h = 0.5
        kv = k.evaluate(r, np.array([h]))[0]
        self.assertAlmostEqual(kv, self.calc_fac(h) * (0.25 * pow(2 - q, 3)))

        # Case 3
        q = 3.0
        h = 1 / 3
        kv = k.evaluate(r, np.array([h]))[0]
        self.assertAlmostEqual(kv, 0.0)

    def test_gradient(self):
        k = CubicSpline()
        r = np.array([2.0])
        x = np.array([3.0])

        # Case 1
        q = 0.5
        h = 4.0
        g = k.gradient(x, r, np.array([h]))[0]
        dwdq = self.calc_fac(h) * -3.0 * q * (1 - 0.75 * q)
        self.assertAlmostEqual(g, dwdq / (h * r[0]) * x[0])

        q = 1.0
        h = 2.0
        g = k.gradient(x, r, np.array([h]))[0]
        dwdq = self.calc_fac(h) * -0.75 * (2 - q) * (2 - q)
        self.assertAlmostEqual(g, dwdq / (h * r[0]) * x[0])
        
        # Case 2
        q = 2.0
        h = 1.0
        g = k.gradient(x, r, np.array([h]))[0]
        dwdq = self.calc_fac(h) * -0.75 * (2 - q) * (2 - q)
        self.assertAlmostEqual(g, dwdq / (h * r[0]) * x[0])

        # Case 3
        q = 3.0
        h = 2 / 3
        g = k.gradient(x, r, np.array([h]))[0]
        self.assertAlmostEqual(g, 0.0)

    def calc_fac(self, h: float) -> float:
        return 10 / (7 * np.pi * h * h)

if __name__ == "__main__":
    unittest.test()