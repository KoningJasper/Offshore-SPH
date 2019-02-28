# Add parent folder to path; for directly running the file
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import unittest
import numpy as np
from time import perf_counter
from src.Common import particle_dtype
from src.Tools.NNLinkedList import NNLinkedList
from src.Tools.NNCellList import NNCellList

class test_nn_algos(unittest.TestCase):
    def test_linkedlist(self):
        print('==============')
        print('  LinkedList  ')
        print('==============')
        # Init some particles
        h = 1.0; pA = self.create_particles(h)
        print(f'{len(pA)} particles')

        # Create NNLinkedList
        nn = NNLinkedList(3.0, True)

        # Check empty
        self.assertEqual(len(nn.shifts), 3)
        self.assertEqual(nn.scale, 3.0)

        # Check number of cells
        # Find mins and maxes
        nn.xmin = pA['x'].min()
        nn.xmax = pA['x'].max()
        nn.ymin = pA['y'].min()
        nn.ymax = pA['y'].max()
        cell_size = nn.get_cell_size(pA)
        nn.cell_size = cell_size

        n_cells = nn.get_number_of_cells(pA)

        # Update
        nn.update(pA) # Compile
        count = 100; start = perf_counter()
        for i in range(count):
            nn.update(pA)
        end = perf_counter() - start
        print(f'Completed {count} updates in {end:f} [s]')
        print(f'Per-loop: {round(end / count * 1000, 2)} [ms]')

        self.assertEqual(len(nn.nexts), len(pA))

        # Check nearest bottom-left corner
        nn.near(0, pA) # Compile
        count = 1_000; start = perf_counter();
        for i in range(count):
            near_0 = nn.near(0, pA)
        end = perf_counter() - start
        print(f'Completed {count} near-finds in {end:f} [s]')
        print(f'Per-loop: {round(end / count * 1000, 2)} [ms]')

        self.assertGreater(len(near_0), 0)
        self.assertIn(0, near_0)  # Self: (0, 0)
        self.assertIn(1, near_0)  # Up (0, 1)
        self.assertIn(26, near_0) # Right (1, 0)
        self.assertIn(27, near_0) # Right-up (1, 1)

        # Check that all the nears fall in range
        for n in near_0[near_0 > -1]:
            x = pA[n]['x'] - pA[0]['x']; y = pA[n]['y'] - pA[0]['y']
            r = np.sqrt(x * x + y * y)
            q = r / h

            self.assertLessEqual(q, 3.0)

        # Check not any particles missed
        for n in range(len(pA)):
            x = pA[n]['x'] - pA[0]['x']; y = pA[n]['y'] - pA[0]['y']
            r = np.sqrt(x * x + y * y)
            q = r / h

            if q > 3.0:
                self.assertNotIn(n, near_0)
            else:
                self.assertIn(n, near_0)

    def test_celllist(self):
        print('==============')
        print('   CellList   ')
        print('==============')
        # Init some particles
        h = 1.0; pA = self.create_particles(h)
        print(f'{len(pA)} particles')
        nn = NNCellList(2.0, False)

        # Check empty
        self.assertEqual(len(nn.shifts), 3)
        self.assertEqual(nn.scale, 2.0)

        # Check number of cells
        # Find mins and maxes
        nn.xmin = pA['x'].min()
        nn.xmax = pA['x'].max()
        nn.ymin = pA['y'].min()
        nn.ymax = pA['y'].max()
        cell_size = nn.get_cell_size(pA)
        nn.cell_size = cell_size

        n_cells = nn.get_number_of_cells(pA)

        # Update
        nn.update(pA) # Compile
        count = 100; start = perf_counter()
        for i in range(count):
            nn.update(pA)
        end = perf_counter() - start
        print(f'Completed {count} updates in {end:f} [s]')
        print(f'Per-loop: {round(end / count * 1000, 2)} [ms]')

        self.assertEqual(len(nn.heads), len(pA))
        self.assertEqual(len(nn.boxes), n_cells)

        # Check nearest bottom-left corner
        nn.near(0, pA) # Compile
        count = 1_000; start = perf_counter();
        for i in range(count):
            near_0 = nn.near(0, pA)
        end = perf_counter() - start
        print(f'Completed {count} near-finds in {end:f} [s]')
        print(f'Per-loop: {round(end / count * 1000, 2)} [ms]')

        self.assertGreater(len(near_0), 0)
        self.assertIn(0, near_0)  # Self: (0, 0)
        self.assertIn(1, near_0)  # Up (0, 1)
        self.assertIn(26, near_0) # Right (1, 0)
        self.assertIn(27, near_0) # Right-up (1, 1)

        # Check that all the nears fall within an acceptable range,
        # Acceptable is 5.0 instead of 3.0
        acceptable = 5.0
        for n in near_0[near_0 > -1]:
            x = pA[n]['x'] - pA[0]['x']; y = pA[n]['y'] - pA[0]['y']
            r = np.sqrt(x * x + y * y)
            q = r / h

            self.assertLessEqual(q, acceptable)

        # Check not any particles missed
        for n in range(len(pA)):
            x = pA[n]['x'] - pA[0]['x']; y = pA[n]['y'] - pA[0]['y']
            r = np.sqrt(x * x + y * y)
            q = r / h

            if q > acceptable:
                self.assertNotIn(n, near_0)
            elif q <= 3.0:
                self.assertIn(n, near_0)
            else:
                # Doesn't matter if they're in or out
                continue

    def create_particles(self, h: float):
        N = 25

        # Create some particles
        xv = np.linspace(0, 25, N + 1)
        yv = np.linspace(0, 25, N + 1)
        x, y = np.meshgrid(xv, yv, indexing='ij')
        x = x.ravel()
        y = y.ravel()

        pA = np.zeros(len(x), dtype=particle_dtype)
        pA['x'] = x; pA['y'] = y
        pA['h'] = h
        return pA


if __name__ == "__main__":
    test_nn_algos().test_linkedlist()
    test_nn_algos().test_celllist()