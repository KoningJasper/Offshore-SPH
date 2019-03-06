# Add parent folder to path
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import numpy as np
import unittest
from src.Particle import Particle
from src.Common import ParticleType
from src.Helpers import Helpers

class test_helper(unittest.TestCase):
    def test(self):
        N = 10
        old = test_helper.create_particles(N=N, mass=1.0)
        new = test_helper.new_particles(N=N, mass=1.0)

        self.assertEqual(len(old), len(new))

        # Compare the fluid
        r = [f'{round(p.r[0], 0)};{round(p.r[1], 0)}' for p in old]
        for i in range(len(new)):
            rn = f"{round(new[i]['x'], 0)};{round(new[i]['y'], 0)}"
            self.assertIn(rn, r)

            # Find the index
            index = r.index(rn)
            self.assertEqual(new[i]['rho'], old[index].rho)
            self.assertEqual(new[i]['m'], old[index].m)

    @staticmethod
    def create_particles(N: int, mass: float):
        # Create some particles
        xv = np.linspace(0, 25, N)
        yv = np.linspace(0, 25, N)
        y, x = np.meshgrid(xv, yv, indexing='ij')
        x = x.ravel()
        y = y.ravel()

        particles = []
        for i in range(len(x)):
            particles.append(Particle('fluid', x[i], y[i], mass))
        
        r0     = 25 / N # Distance between boundary particles.
        rho_b  = 1000. # Density of the boundary [kg/m^3]
        mass_b = mass * 1.5 # Mass of the boundary [kg]

        # Maximum and minimum values of boundaries
        x_min = - 2.5 * r0
        x_max = 150
        y_min = - 2.5 * r0
        y_max = 30
        
        # Bottom & Top
        xv = np.linspace(x_min, x_max, np.ceil((x_max - x_min) / r0))
        yv = np.zeros(len(xv)) + y_min
        for i in range(len(xv)):
            particles.append(Particle('boundary', xv[i] + r0 / 2, yv[i], mass_b, rho_b))
            #particles.append(Particle('boundary', xv[i] - r0 / 2, yv[i] - r0, mass_b, rho_b))

        # Left & Right
        yv3 = np.linspace(y_min, y_max, np.ceil((y_max - y_min) / r0))    
        xv2 = np.zeros(len(yv3)) + x_min
        for i in range(len(yv3)):
            particles.append(Particle('boundary', xv2[i], yv3[i] + r0 / 2, mass_b, rho_b))
            #particles.append(Particle('boundary', xv2[i] - r0, yv3[i] - r0 / 2, mass_b, rho_b))
            
        # Temp-Boundary
        xvt = np.zeros(len(yv3)) + 25 - x_min
        for i in range(len(yv3)):
            particles.append(Particle('temp-boundary', xvt[i], yv3[i] + r0 / 2, mass_b, rho_b))
            #particles.append(Particle('temp-boundary', xvt[i] + r0, yv3[i] - r0 / 2, mass_b, rho_b))

        return particles

    @staticmethod
    def new_particles(N: int, mass: float):
        helper = Helpers(1000.0)
        new = helper.box(0, 25, 0, 25, N * N, hex=False, mass=mass)

        r0 = 25 / N # Distance between boundary particles.

        # Create boundaries
        bottom = helper.box(xmin=-2 * r0, xmax=150 + r0 / 2, ymin=-2.5*r0, ymax=-2.5*r0, r0=r0, hex=False, mass=1.5 * mass, type=ParticleType.Boundary)
        left   = helper.box(xmin=-2.5 * r0, xmax=-2.5 * r0, ymin=-2*r0, ymax=30 + r0 / 2, r0=r0, hex=False, mass=1.5 * mass, type=ParticleType.Boundary)
        temp   = helper.box(xmin=25 + 2.5 * r0, xmax=25 + 2.5 * r0, ymin=-2*r0, ymax=30 + r0 / 2, r0=r0, hex=False, mass=1.5 * mass, type=ParticleType.TempBoundary)

        return np.concatenate((new, bottom, left, temp))


if __name__ == "__main__":
    test_helper().test()