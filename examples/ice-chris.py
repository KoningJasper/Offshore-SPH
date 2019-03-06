# Add parent folder to path
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np

from src.Solver import Solver
from src.Helpers import Helpers
from src.Methods.WCSPH import WCSPH
from src.Kernels.CubicSpline import CubicSpline
from src.Integrators.PEC import PEC
from src.Post.Plot import Plot
from src.Common import ParticleType

def create_particles(Nx: int, rho0: float, x_end: float, y_end: float):
    """
        Nx: int
            Number of particles in x-direction. Total number of particles is computed by assuming a constant spacing (r0).
    """
    r0   = x_end / Nx # Distance between particles.

    # Create some fluid particles
    fluid = Helpers.rect(xmin=0, xmax=25, ymin=0, ymax=25, r0=r0, mass=1.0, rho0=rho0, pack=True)

    # Compute the mass based on the average area
    dA         = x_end * y_end / len(fluid)
    mass       = dA * rho0
    fluid['m'] = mass

    # Maximum and minimum values of boundaries
    x_min = - 2 * r0
    x_max = x_end + 2 * r0
    y_min = - 2 * r0
    y_max = y_end
    mass_b = mass * 1.5 # Mass of the boundary [kg]
    
    # Create the boundary
    bottom = Helpers.rect(xmin=x_min, xmax=x_max, ymin=y_min, ymax=y_min, r0=r0, mass=mass_b, rho0=rho0, label=ParticleType.Boundary)
    left   = Helpers.rect(xmin=x_min, xmax=x_min, ymin=y_min, ymax=y_max, r0=r0, mass=mass_b, rho0=rho0, label=ParticleType.Boundary)
    right  = Helpers.rect(xmin=x_max, xmax=x_max, ymin=y_min, ymax=y_max, r0=r0, mass=mass_b, rho0=rho0, label=ParticleType.Boundary)

    # TODO: Ship/plate moving boundary.

    return np.concatenate((fluid, bottom, left, right))

def main():
    # ----- Setup ----- #
    # Simulation parameters
    Nx = 30; duration = 1.0
    height = 25.0; width = 25.0

    # Other parameters
    rho0 = 1000.0; XSPH = True 
    plot = True

    # Create some particles
    pA = create_particles(Nx, rho0, width, height)

    # Create the solver
    kernel     = CubicSpline()
    method     = WCSPH(height=height, rho0=rho0, num_particles=len(pA), useXSPH=XSPH)
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
    exportPath = f'{sys.path[0]}\\ice-chris.hdf5'
    solver.save(exportPath)

    if plot == True:
        plt = Plot(exportPath, title=f'Ice Breaking (2D); {len(pA)} particles', xmin=-3, xmax=81, ymin=-3, ymax=41)
        plt.save(f'{sys.path[0]}\\ice-chris.mp4')

if __name__ == '__main__':
    main()

