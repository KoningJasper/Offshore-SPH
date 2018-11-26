import unittest
import src.Kernels.Gaussian
import numpy as np

class test_kernels_Gaussian(unittest.TestCase):

    def test_alpha(self):
        self.assertAlmostEqual(src.Kernels.Gaussian.Gaussian().alpha, 0.31830988618379064)
        self.assertAlmostEqual(src.Kernels.Gaussian.Gaussian().alpha, 0.31830988618379064 / 4)
        self.assertAlmostEqual(src.Kernels.Gaussian.Gaussian().alpha, 0.31830988618379064 * 4)

    def test_evaluate(self):
        """ Evaluates distances 0.5, 1.0 and 2.0 against the kernel implementation. """
        kernel = src.Kernels.Gaussian.Gaussian()
        self.assertAlmostEqual(kernel.evaluate(x=None, r=np.array([0.5]), h=np.array([1.0]))[0], 0.2478999886193059)
        self.assertAlmostEqual(kernel.evaluate(x=None, r=np.array([1.0]), h=np.array([1.0]))[0], 0.11709966304863831)
        self.assertAlmostEqual(kernel.evaluate(x=None, r=np.array([2.0]), h=np.array([1.0]))[0], 0.005830048930056386)

        kernel = src.Kernels.Gaussian.Gaussian()
        self.assertAlmostEqual(kernel.evaluate(x=None, r=np.array([0.5]), h=np.array([0.5]))[0], 0.07475611627593091)
        self.assertAlmostEqual(kernel.evaluate(x=None, r=np.array([1.0]), h=np.array([0.5]))[0], 0.061974997154826475)
        self.assertAlmostEqual(kernel.evaluate(x=None, r=np.array([2.0]), h=np.array([0.5]))[0], 0.029274915762159577)

    def test_gradient(self):
        """ Evaluates distances 0.5, 1.0 and 2.0 against the kernel implementation. """
        kernel = src.Kernels.Gaussian.Gaussian()
        self.assertAlmostEqual(kernel.gradient(r=0.5), -0.3718499829289588)
        self.assertAlmostEqual(kernel.gradient(r=1.0), 0.0)
        self.assertAlmostEqual(kernel.gradient(r=2.0), 0.034980293580338315)

        kernel = src.Kernels.Gaussian.Gaussian()
        self.assertAlmostEqual(kernel.gradient(r=0.5), -0.07008385900868523)
        self.assertAlmostEqual(kernel.gradient(r=1.0), -0.04648124786611985)
        self.assertAlmostEqual(kernel.gradient(r=2.0), 0.0)

if __name__ == '__main__':
    unittest.main()