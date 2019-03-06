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

def create_particles(N: int, mass: float):
    r0 = 25 / N # Distance between boundary particles.
    rho_b = 1000. # Density of the boundary [kg/m^3]
    mass_b = mass * 1.5 # Mass of the boundary [kg]

    # Create some fluid particles
    fluid = Helpers.rect(xmin=0, xmax=25, ymin=0, ymax=25, r0=r0, mass=mass, rho0=rho_b, pack=True)

    # Maximum and minimum values of boundaries
    x_min = - 2 * r0
    x_max = 150
    y_min = - 2 * r0
    y_max = 30
    
    # Create the boundary
    bottom = Helpers.rect(xmin=x_min, xmax=x_max, ymin=y_min, ymax=y_min, r0=r0, mass=mass_b, rho0=rho_b, label=ParticleType.Boundary)
    left   = Helpers.rect(xmin=x_min, xmax=x_min, ymin=y_min, ymax=y_max, r0=r0, mass=mass_b, rho0=rho_b, label=ParticleType.Boundary)
    temp   = Helpers.rect(xmin=25 - x_min, xmax=25 - x_min, ymin=y_min, ymax=y_max, r0=r0, mass=mass_b, rho0=rho_b, label=ParticleType.TempBoundary)

    return np.concatenate((fluid, bottom, left, temp))

def main():
    # ----- Setup ----- #
    # Main parameters
    N = 30; rho0 = 1000.0; duration = 1.0
    XSPH = True; height = 25.0; plot = True

    # Create some particles
    dA   = 25 * 25 / N ** 2 # Area per particle. [m^2]
    mass = dA * rho0
    pA   = create_particles(N, mass)

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
    exportPath = f'{sys.path[0]}\\dam-break-2d.hdf5'
    solver.save(exportPath)

    if plot == True:
        plt = Plot(exportPath, title=f'Dam Break (2D); {len(pA)} particles', xmin=-3, xmax=81, ymin=-3, ymax=41)
        plt.save(f'{sys.path[0]}\\dam-break-2d.mp4')

if __name__ == '__main__':
    main()