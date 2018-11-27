import unittest
from src.Kernels.Gaussian import Gaussian
import numpy as np


class test_kernels_Gaussian(unittest.TestCase):
    """ Tests the Gaussian kernel implementation against known values of the gaussian kernel """

    def test_evaluate(self):
        # Arrange
        kernel = Gaussian()
        x = np.array([0., 0.])
        r = np.array([0.5])
        h = np.array([1.])

        # Act
        val = kernel.evaluate(x, r, h)[0]

        # Assert
        self.assertAlmostEqual(val, 0.2478999886193059)

    def test_derivative(self):
        # Arrange
        kernel = Gaussian()
        r = np.array([0.25])
        h = np.array([0.5])

        # Act
        val = kernel.derivative(r, h)[0]

        # Assert
        self.assertAlmostEqual(val, -0.9915999544772236)

    def test_gradient(self):
        # Arrange
        kernel = Gaussian()
        x = np.array([1., 1.])
        r = np.array([0.25])
        h = np.array([1.3 * 0.25])

        # Act
        grad = kernel.gradient(x, r, h)[0]

        # Assert
        self.assertAlmostEqual(grad[0], -31.576769410027108)
        self.assertAlmostEqual(grad[1], -31.576769410027108)

    def test_gradient_array(self):
        """ Verifies if it performs well on multiple entries. """
        # Arrange
        kernel = Gaussian()
        x = np.array([[1., 1.], [0.5, 0.5]])
        r = np.array([0.25, 0.25])
        h = np.array([1, 1]) * 1.3 * 0.25

        # Act
        grad = kernel.gradient(x, r, h)

        # Assert
        self.assertAlmostEqual(grad[0][0], -31.576769410027108)
        self.assertAlmostEqual(grad[0][1], -31.576769410027108)
        self.assertAlmostEqual(grad[1][0], -15.788384705013554)
        self.assertAlmostEqual(grad[1][1], -15.788384705013554)


if __name__ == '__main__':
    unittest.main()
