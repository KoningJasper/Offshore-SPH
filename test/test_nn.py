# Add parent folder to path; for directly running the file
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import unittest
import numpy as np
from src.Common import particle_dtype
from src.Tools.NearNeighbours import NearNeighboursCell

class test_nn(unittest.TestCase):
    def test(self):
        nn = NearNeighboursCell(alpha=2.0)
        
        pA = np.zeros(10, dtype=particle_dtype)
        pA['x'] = np.arange(10)
        pA['h'] = 2

        # Compute the neighbourhood
        nn.update(pA)

        # Check
        self.assertEqual(len(nn.pCells), 10)

        # Compute near neighbours for nr 0
        near_0 = nn.near(0)
        self.assertSetEqual(set(near_0), set([0, 1, 2, 3, 4, 5, 6, 7]))


if __name__ == "__main__":
    test_nn().test()