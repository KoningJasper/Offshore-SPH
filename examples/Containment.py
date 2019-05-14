# Add parent folder to path
import sys, os, math
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np

from src.Solver import Solver
from src.Helpers import Helpers
from src.Methods.WCSPH import WCSPH
from src.Kernels.CubicSpline import CubicSpline
from src.Integrators.PEC import PEC
from src.Post.Plot import Plot
from src.Common import ParticleType, particle_dtype

def create_particles(Nx: int, rho0: float, x_end: float, y_end: float):
    """
        Nx: int
            Number of particles in x-direction. Total number of particles is computed by assuming a constant spacing (r0).
    """
    r0   = x_end / Nx # Distance between particles.

    # Create some fluid particles
    fluid = Helpers.rect(xmin=0, xmax=x_end, ymin=0, ymax=y_end, r0=r0, mass=1.0, rho0=rho0, pack=True)

    # Compute the mass based on the average area
    dA         = x_end * y_end / len(fluid)
    mass       = dA * rho0
    fluid['m'] = mass

    # Maximum and minimum values of boundaries
    x_min = - r0
    x_max = x_end + r0
    y_min = - r0
    y_max = 1.3 * y_end + r0
    
    # Create the boundary
    bottom = Helpers.rect(xmin=x_min, xmax=x_max, ymin=y_min, ymax=y_min, r0=r0, mass=0., rho0=0., label=ParticleType.Boundary)
    left   = Helpers.rect(xmin=x_min, xmax=x_min, ymin=y_min, ymax=y_max, r0=r0, mass=0., rho0=0., label=ParticleType.Boundary)
    right  = Helpers.rect(xmin=x_max, xmax=x_max, ymin=y_min, ymax=y_max, r0=r0, mass=0., rho0=0., label=ParticleType.Boundary)

    return r0, np.concatenate((fluid, bottom, left, right))

def main():
    # ----- Setup ----- #
    # Simulation parameters
    Nx = 30; duration = 3.0
    height = 1.0; width = 1.0

    # Other parameters
    rho0 = 1000.0; XSPH = False 
    plot = True

    # Create some particles
    r0, pA = create_particles(Nx, rho0, width, height)

    # Create the solver
    kernel     = CubicSpline()
    method     = WCSPH(height=height, rho0=rho0, r0=r0, useXSPH=XSPH, Pb=0)
    integrator = PEC(useXSPH=XSPH, strict=False)
    solver     = Solver(method, integrator, kernel, duration, quick=False, incrementalWriteout=False, exportProperties=['x', 'y', 'p', 'vx', 'vy'])

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
    exportPath = f'{sys.path[0]}/containment.hdf5'
    solver.save(exportPath)

    if plot == True:
        plt = Plot(exportPath, title=f'Containment (2D); {len(pA)} particles', xmin=-0.2, xmax=1.2, ymin=-0.2, ymax=1.2)
        plt.save(f'{sys.path[0]}/containment.mp4')

if __name__ == '__main__':
    main()

