import unittest
import numpy as np
import scipy.spatial

# Add parent folder to path
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from src.Kernels.Gaussian import Gaussian

# PySPH implementation
from PySPH.Gaussian import Gaussian as pGaussian

class test_kernels_Gaussian(unittest.TestCase):
    """ Tests the Gaussian kernel implementation against known values of the gaussian kernel """

    def createParticles(self, N: int = 10) -> np.array:
        # Create some particles
        xv = np.linspace(0, 10, N)
        yv = np.linspace(0, 10, N)
        x, y = np.meshgrid(xv, yv, indexing='ij')
        x = x.ravel().reshape(N ** 2, 1)
        y = y.ravel().reshape(N ** 2, 1)
        r = np.hstack((x, y))

        return r

    def test_evaluate_many(self):
        kernel = Gaussian()
        pKernel = pGaussian()
        
        # Create particles
        r = self.createParticles()
        
        # Calculate distance
        dist = scipy.spatial.distance.cdist(r, r, 'euclidean')
        for i in range(len(r)):
            for hdx in np.arange(0.1, 1, 0.1):
                xij = r[:] - r[i]
                rij = dist[i, :]
                hij = hdx * np.ones(len(r))

                # Evaluate gradient
                k = kernel.evaluate(rij, hij)

                for ii in range(len(r)):
                    v = pKernel.kernel(xij[ii, :], rij[ii], hij[ii])

                    # Check values
                    self.assertAlmostEqual(k[ii], v)

    def test_derivative_many(self):
        kernel = Gaussian()
        pKernel = pGaussian()
        
        # Create particles
        r = self.createParticles()
        
        # Calculate distance
        dist = scipy.spatial.distance.cdist(r, r, 'euclidean')
        for i in range(len(r)):
            for hdx in np.arange(0.1, 1, 0.1):
                rij = dist[i, :]
                hij = hdx * np.ones(len(r))

                # Evaluate gradient
                k = kernel.derivative(rij, hij)

                for ii in range(len(r)):
                    v = pKernel.dwdq(rij[ii], hij[ii])

                    # Check values
                    self.assertAlmostEqual(k[ii], v)

    def test_gradient_many(self):
        kernel = Gaussian()
        pKernel = pGaussian()
        
        # Create particles
        r = self.createParticles()
        
        # Calculate distance
        dist = scipy.spatial.distance.cdist(r, r, 'euclidean')
        for i in range(len(r)):
            for hdx in np.arange(0.1, 1, 0.1):
                xij = r[:] - r[i]
                rij = dist[i, :]
                hij = hdx * np.ones(len(r))

                # Evaluate gradient
                dwij = np.empty_like(xij)
                dwij[:, 0] = kernel.gradient(xij[:, 0], rij, hij)
                dwij[:, 1] = kernel.gradient(xij[:, 1], rij, hij)

                for ii in range(len(r)):
                    grad = [0, 0, 0]
                    pKernel.gradient(xij[ii, :], rij[ii], hij[ii], grad)

                    # Check values
                    self.assertAlmostEqual(dwij[ii, 0], grad[0])
                    self.assertAlmostEqual(dwij[ii, 1], grad[1])

if __name__ == '__main__':
    unittest.main()
