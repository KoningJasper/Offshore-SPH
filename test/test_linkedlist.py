# Add parent folder to path; for directly running the file
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import unittest
import numpy as np
from src.Common import particle_dtype
from src.Tools.NNLinkedList import NNLinkedList

class test_linkedlist(unittest.TestCase):
    def test(self):
        pA = self.create_particles()

        # Create NNLinkedList
        nn = NNLinkedList()

        # Check empty
        self.assertEqual(len(nn.shifts), 3)
        self.assertEqual(nn.scale, 2.0)

        # Update
        nn.update(pA)

        self.assertEqual(len(nn.nexts), len(pA))
        self.assertEqual(nn.cell_size, 2.0)

        

    def create_particles(self):
        N = 25

        # Create some particles
        xv = np.linspace(0, 25, N)
        yv = np.linspace(0, 25, N)
        x, y = np.meshgrid(xv, yv, indexing='ij')
        x = x.ravel()
        y = y.ravel()

        pA = np.zeros(len(x), dtype=particle_dtype)
        pA['x'] = x; pA['y'] = y
        pA['h'] = 1.0
        return pA


if __name__ == "__main__":
    test_linkedlist().test()