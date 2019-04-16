# Add parent folder to path
import sys, os, math
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np

from src.Solver import Solver
from src.Helpers import Helpers
from src.Methods.WCSPH import WCSPH
from src.Kernels.CubicSpline import CubicSpline
from src.Integrators.PEC import PEC
from src.Integrators.Euler import Euler
from src.Post.Plot import Plot
from src.Common import ParticleType, particle_dtype

def create_particles(N: int, mass: float, obs: bool):
    r0     = 25 / N     # Distance between boundary particles.
    rho_b  = 0.      # Density of the boundary [kg/m^3]
    mass_b = 0          # Mass of the boundary [kg]

    # Create some fluid particles
    fluid = Helpers.rect(xmin=0, xmax=25, ymin=0, ymax=25, r0=r0, mass=mass, rho0=1000.0, pack=True)

    # Maximum and minimum values of boundaries
    # Keep 1.5 spacing
    x_min = - 1 * r0
    x_max = 150
    y_min = - 1 * r0
    y_max = 30
    
    # Create the boundary
    bottom = Helpers.rect(xmin=x_min, xmax=x_max, ymin=y_min, ymax=y_min, r0=r0, mass=mass_b, rho0=rho_b, label=ParticleType.Boundary)
    left   = Helpers.rect(xmin=x_min, xmax=x_min, ymin=y_min, ymax=y_max, r0=r0, mass=mass_b, rho0=rho_b, label=ParticleType.Boundary)
    temp   = Helpers.rect(xmin=25 - x_min, xmax=25 - x_min, ymin=y_min, ymax=y_max, r0=r0, mass=mass_b, rho0=rho_b, label=ParticleType.TempBoundary)

    left['y'] = left['y'] - 0.5 * r0

    # Create the triangular thingy
    # I measured it at 2.5m * 2.5m, starts at 50
    if obs == True:
        side = 2.5
        Nt = math.ceil(side / (r0 / 8))
        x = np.linspace(50, 50 + side, Nt)
        y = np.linspace(0, side, Nt)
        
        triag = np.zeros(3 * Nt, dtype=particle_dtype)
        triag[0:Nt]['x'] = x; triag[0:Nt]['y'] = y # Diag
        triag[Nt:2*Nt]['x'] = 50 + side; triag[Nt:2*Nt]['y'] = y # Right
        triag[2*Nt:3*Nt]['x'] = x;         triag[2*Nt:3*Nt]['y'] = 0 # Bottom

        # General properties
        triag['m'] = mass_b; triag['rho'] = rho_b; triag['label'] = ParticleType.Boundary

        return r0, np.concatenate((fluid, bottom, left, temp, triag))
    else:
        return r0, np.concatenate((fluid, bottom, left, temp))

def main():
    # ----- Setup ----- #
    # Main parameters
    N = 50; rho0 = 1000.0; duration = 5.0
    XSPH = True; height = 25.0; plot = True

    # Create some particles
    dA     = 25 * 25 / N ** 2 # Area per particle. [m^2]
    mass   = dA * rho0
    r0, pA = create_particles(N, mass, obs=False)

    # Create the solver
    kernel     = CubicSpline()
    method     = WCSPH(height=height, r0=r0, rho0=rho0, useXSPH=XSPH)
    integrator = PEC()
    solver     = Solver(method, integrator, kernel, duration, quick=False, incrementalWriteout=False)

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
        plt = Plot(exportPath, title=f'Dam Break (2D); {len(pA)} particles', xmin=-3, xmax=151, ymin=-3, ymax=41)
        plt.save(f'{sys.path[0]}\\dam-break-2d.mp4')

if __name__ == '__main__':
    main()