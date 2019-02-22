# Add parent folder to path
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np

from src.Solver import Solver
from src.Particle import Particle
from src.Methods.WCSPH import WCSPH
from src.Kernels.CubicSpline import CubicSpline
from src.Integrators.PEC import PEC
from src.Post.Plot import Plot

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
    
    r0 = 25 / N # Distance between boundary particles.
    rho_b = 1000. # Density of the boundary [kg/m^3]
    mass_b = mass * 1.5 # Mass of the boundary [kg]

    # Maximum and minimum values of boundaries
    x_min = - 2.5 * r0
    x_max = 150
    y_min = - 2.5 * r0
    y_max = 30
    
    # Bottom & Top
    xv = np.arange(x_min, x_max, r0)
    yv = np.zeros(len(xv)) + y_min
    for i in range(len(xv)):
        particles.append(Particle('boundary', xv[i] + r0 / 2, yv[i], mass_b, rho_b))
        #particles.append(Particle('boundary', xv[i] - r0 / 2, yv[i] - r0, mass_b, rho_b))

    # Left & Right
    yv3 = np.arange(y_min, y_max, r0)    
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

def main():
    # Main parameters
    N = 40; rho0 = 1000.0; duration = 2.0
    XSPH = True; height = 25.0; plot = True

    # Create some particles
    dA = 25 * 25 / N ** 2 # Area per particle. [m^2]
    mass = dA * rho0
    particles = create_particles(N, mass)

    # Create the solver
    kernel     = CubicSpline()
    method     = WCSPH(height=height, rho0=rho0, num_particles=len(particles), useXSPH=XSPH)
    integrator = PEC(useXSPH=XSPH, strict=True)
    solver     = Solver(method, integrator, kernel, duration, quick=True)

    # Add the particles
    solver.addParticles(particles)

    # Setup
    solver.setup()

    # Run it!
    solver.run()

    # Output timing
    solver.timing()

    # Output
    exportPath = f'{sys.path[0]}\\dam-break-2d.hdf5'
    solver.save(exportPath)

    if plot == True:
        plt = Plot(exportPath, title=f'Dam Break (2D); {len(particles)} particles', xmin=-3, xmax=81, ymin=-3, ymax=41)
        plt.save(f'{sys.path[0]}\\dam-break-2d.mp4')

if __name__ == '__main__':
    main()