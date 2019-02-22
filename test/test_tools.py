import unittest
import numpy as np
from src.Common import particle_dtype
from src.Tools.SolverTools import findActive, _assignProps, computeProps

class test_tools(unittest.TestCase):
    def test_findActive(self):
        no = np.zeros(10, dtype=particle_dtype)

        count, indexes = findActive(len(no), no)

        self.assertEqual(count, 10)
        self.assertEqual(len(indexes), 10)

        # Set number 5 active
        no['deleted'][5] = True

        count, indexes = findActive(len(no), no)
        self.assertEqual(count, 9)
        self.assertEqual(len(indexes), 10)
        self.assertFalse(indexes[5])


if __name__ == "__main__":
    test_tools().test()