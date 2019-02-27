# Add parent folder to path
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import unittest
import numpy as np
from src.Equations.KineticEnergy import KineticEnergy
from src.Common import particle_dtype

class test_kinetic_energy(unittest.TestCase):
    def test(self):
        num = 100
        pA = np.zeros(num, dtype=particle_dtype)
        kE = KineticEnergy(len(pA), pA)

        self.assertEqual(kE, 0)

        # Add some props
        pA['m'] = 1
        pA['vx'] = 1

        kE = KineticEnergy(len(pA), pA)
        self.assertEqual(kE, num * 0.5 * 1 * 1)


if __name__ == "__main__":
    test_kinetic_energy().test()