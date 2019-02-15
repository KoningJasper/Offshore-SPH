# Add parent folder to path
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np

from src.Solver import Solver
from src.Particle import Particle
from src.Methods.WCSPH import WCSPH
from src.Kernels.Gaussian import Gaussian
from src.Integrators.PEC import PEC

def create_particles(N: int, mass: float):
    # Create some particles
    xv = np.linspace(0, 25, N)
    yv = np.linspace(0, 25, N)
    x, y = np.meshgrid(xv, yv, indexing='ij')
    x = x.ravel()
    y = y.ravel()

    particles = []
    for i in range(len(x)):
        particles.append(Particle('fluid', x[i], y[i], mass))
    
    r0 = 25 / N / 2 # Distance between boundary particles.
    rho_b = 1000. # Density of the boundary [kg/m^3]
    mass_b = mass * 1.5 # Mass of the boundary [kg]

    # Maximum and minimum values of boundaries
    x_min = -3
    x_max = 80
    y_min = -3
    y_max = 40
    
    # Bottom & Top
    xv = np.arange(x_min, x_max, r0)
    yv = np.zeros(len(xv)) + y_min
    for i in range(len(xv)):
        particles.append(Particle('boundary', xv[i], yv[i], mass_b, rho_b))
        particles.append(Particle('boundary', xv[i] - r0 / 2, yv[i] - r0, mass_b, rho_b))

    # Left & Right
    yv3 = np.arange(y_min, y_max, r0)    
    xv2 = np.zeros(len(yv3)) + x_min
    for i in range(len(yv3)):
        particles.append(Particle('boundary', xv2[i], yv3[i], mass_b, rho_b))
        particles.append(Particle('boundary', xv2[i] - r0, yv3[i] - r0 / 2, mass_b, rho_b))
        
    return particles

def main():
    # Main parameters
    N = 10; rho0 = 1000.0; duration = 1.0

    # Create some particles
    dA = 25 * 25 / N ** 2 # Area per particle. [m^2]
    mass = dA * rho0
    particles = create_particles(N, mass)

    # Create the solver
    kernel = Gaussian()
    method = WCSPH(height=25.0, rho0=rho0, num_particles=len(particles))
    integrator = PEC()
    solver = Solver(method, integrator, kernel, duration, plot=True)

    # Add the particles
    solver.addParticles(particles)

    # Setup
    solver.setup()

    # Run it!
    solver.solve()

    # Output timing
    solver.timing()

    # Output
    solver.save()

if __name__ == '__main__':
    main()