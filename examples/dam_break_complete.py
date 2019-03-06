# Add parent folder to path
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
from math import floor

from src.Solver import Solver
from src.Particle import Particle
from src.Methods.WCSPH import WCSPH
from src.Kernels.CubicSpline import CubicSpline
from src.Integrators.PEC import PEC
from src.Post.Plot import Plot
from src.Common import particle_dtype, get_label_code

def create_particles(N: int, mass: float):
    r0 = 25 / N # Distance between boundary particles.
    rho_b = 1000. # Density of the boundary [kg/m^3]
    mass_b = mass * 1.5 # Mass of the boundary [kg]

    # Create some particles
    xv = np.linspace(0, 25, N)
    yv = np.linspace(0, 25, N)
    y, x = np.meshgrid(xv, yv, indexing='ij')
    x = x.ravel()
    y = y.ravel()

    particles = []
    for i in range(len(x)):
        particles.append(Particle('fluid', x[i], y[i], mass))
    
    hexP = True # Use hex packing
    if hexP == True:
        # Iterate the rows of particles
        for i in range(floor(N / 2)):
            for p in particles[2*i*N : 2*i*N + N]:
                p.r[0] += r0 / 2

    # Maximum and minimum values of boundaries
    x_min = - 2 * r0
    x_max = 150
    y_min = - 2 * r0
    y_max = 30
    
    # Bottom & Top
    xv = np.arange(x_min, x_max, r0)
    yv = np.zeros(len(xv)) + y_min
    for i in range(len(xv)):
        particles.append(Particle('boundary', xv[i] + r0 / 2, yv[i], mass_b, rho_b))

    # Left & Right
    yv3 = np.arange(y_min, y_max, r0)    
    xv2 = np.zeros(len(yv3)) + x_min
    for i in range(len(yv3)):
        particles.append(Particle('boundary', xv2[i], yv3[i] + r0 / 2, mass_b, rho_b))
        
    # Temp-Boundary
    xvt = np.zeros(len(yv3)) + 25 - x_min
    for i in range(len(yv3)):
        particles.append(Particle('temp-boundary', xvt[i], yv3[i] + r0 / 2, mass_b, rho_b))

    return particles

def _convertParticles(particles: np.array):
    """ Convert the particle classes to a numpy array. """
    # Init empty array
    num_particles = len(particles)
    particleArray = np.zeros(num_particles, dtype=particle_dtype)
    for i, p in enumerate(particles):
        pA          = particleArray[i]
        pA['label'] = get_label_code(p.label)
        pA['m']     = p.m
        pA['rho']   = p.rho
        pA['p']     = p.p
        pA['x']     = p.r[0]
        pA['y']     = p.r[1]

    return particleArray

def main():
    # ----- Setup ----- #
    # Main parameters
    N = 50; rho0 = 1000.0; duration = 5.0
    XSPH = True; height = 25.0; plot = True

    # Create some particles
    dA        = 25 * 25 / N ** 2 # Area per particle. [m^2]
    mass      = dA * rho0
    particles = create_particles(N, mass)
    pA        = _convertParticles(particles)

    # Create the solver
    kernel     = CubicSpline()
    method     = WCSPH(height=height, rho0=rho0, num_particles=len(particles), useXSPH=XSPH)
    integrator = PEC(useXSPH=XSPH, strict=True)
    solver     = Solver(method, integrator, kernel, duration, quick=True, incrementalWriteout=False)

    # Add the particles
    solver.addParticles(pA)

    # Setup
    solver.setup()

    # ----- Run ----- #
    solver.run()

    # ----- Post ----- #
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