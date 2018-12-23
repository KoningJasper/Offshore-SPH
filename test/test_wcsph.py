import unittest
import math
import numpy as np
import scipy.spatial

# Add parent folder to path
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from src.Particle import Particle
from src.Equations.WCSPH import WCSPH
from src.Kernels.Gaussian import Gaussian
from PySPH.Momentum import MomentumEquation


class test_wcsph(unittest.TestCase):
    def test_speed_of_sound(self):
        # Act
        wcsph = WCSPH(height=1)

        # Verify, following monaghan
        self.assertAlmostEqual(wcsph.co, 10.0 * math.sqrt(2 * 9.81 * 1.0))

    def test_B(self):
        co: float = 10.0 * math.sqrt(2 * 9.81 * 1.0)
        B: float = co ** 2 * 1000 / 7
        self.assertAlmostEqual(WCSPH(height=1).B, B)

    def test_heigh_initialization(self):
        pass

    def test_taiteos(self):
        pass

    def test_momentum_two_particles(self):
        # Arange
        wcsph = WCSPH(height=2.0)
        mass = 1.0
        h = np.ones(2)
        particles = [Particle('fluid', 0, 0, mass), Particle('fluid', 0.2, 0, mass)]

        r = np.array([p.r for p in particles])
        dist = scipy.spatial.distance.cdist(r, r, 'euclidean')

        # set the initial conditions
        for p in particles:
            wcsph.loop_initialize(p)
            p.p = 9.81 * 2 * 1000

        kernel = Gaussian()

        xij: np.array = r[:] - particles[0].r
        rij: np.array = dist[0, :]
        vij: np.array = np.zeros([2, 2])
        dwij: np.array = kernel.gradient(xij, rij, h)

        pressure = np.array([p.p for p in particles])
        rho = np.array([p.rho for p in particles])

        # Act
        wcsph.Momentum(mass, particles[0], pressure, rho, dwij)

        # PySPH implementation
        d_au = [0]
        d_av = [0]
        # Self
        MomentumEquation().loop(0, 0, rho, [], pressure, d_au, d_av, 0, [mass, mass], rho, [], pressure, vij, xij, h, [], [], 0, dwij[0, :], [], 0, 0)
        # Other
        MomentumEquation().loop(0, 1, rho, [], pressure, d_au, d_av, 0, [mass, mass], rho, [], pressure, vij, xij, h, [], [], 0, dwij[1, :], [], 0, 0)

        # Verify
        self.assertLess(particles[0].a[0], 0,
                        msg='Accelerating in the wrong direction.')
        self.assertAlmostEqual(particles[0].a[0], -d_au[0])
        self.assertAlmostEqual(particles[0].a[1], d_av[0])

    def test_momentum_many_particles(self):
        wcsph = WCSPH(height=2)
        kernel = Gaussian()

        # Create some particles
        mass = 1000.0
        xv = np.linspace(0, 2, 10)
        yv = np.linspace(0, 2, 10)
        x, y = np.meshgrid(xv, yv, indexing='ij')
        x = x.ravel()
        y = y.ravel()
        h = np.ones(len(x))

        particles = []
        for i in range(len(x)):
            particles.append(Particle('fluid', x[i], y[i], mass))

        # Init density
        for p in particles:
            wcsph.inital_condition(p)

        pi = particles[0]
        r = np.array([p.r for p in particles])
        dist = scipy.spatial.distance.cdist(r, r, 'euclidean')

        xij: np.array = r[:] - pi.r
        rij: np.array = dist[0, :]
        dwij: np.array = kernel.gradient(xij, rij, h)

        pressure = np.array([p.p for p in particles])
        rho = np.array([p.rho for p in particles])

        # Act
        wcsph.Momentum(mass, pi, pressure, rho, dwij)

        # Verify
        self.assertLess(pi.a[0], 0,
                        msg='Accelerating in the wrong x-direction.')
        self.assertLess(pi.a[1], 0,
                        msg='Accelerating in the wrong y-direction.')
        self.assertAlmostEqual(pi.a[0], -214.39581666)
        self.assertAlmostEqual(pi.a[1], -190.99652198)

    def test_continuity_two_particles(self):
        # Arange
        wcsph = WCSPH(height=2.0)
        mass = 5.0
        h = np.ones(2) * 0.5
        particles = [Particle('fluid', 0, 0, mass), Particle('fluid', 0.2, 0, mass)]

        r = np.array([p.r for p in particles])
        dist = scipy.spatial.distance.cdist(r, r, 'euclidean')

        # set the initial conditions
        for p in particles:
            p.p = 9.81 * 2 * 1000

        kernel = Gaussian()

        xij: np.array = r[:] - particles[0].r
        rij: np.array = dist[0, :]
        vij: np.array = np.ones([2, 2])
        dwij: np.array = kernel.gradient(xij, rij, h)

        # Act
        wcsph.Continuity(mass, particles[0], dwij, vij)

        # Verify
        self.assertLess(particles[0].drho, 0, msg='Negative density')
        self.assertAlmostEqual(particles[0].drho, -8.679865359297883)