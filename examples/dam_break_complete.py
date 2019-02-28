# Add parent folder to path
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np

from src.Solver import Solver
from src.Methods.WCSPH import WCSPH
from src.Kernels.CubicSpline import CubicSpline
from src.Integrators.PEC import PEC
from src.Post.Plot import Plot
from src.Helpers import Helpers
from src.Common import ParticleType

def create_particles(N: int, rho0: float):
    helper = Helpers(rho0)
    pA = helper.box(0, 25, 0, 25, N)
    mass = 25 * 25 * rho0 / len(pA)
    pA['m'] = mass

    # (initial) separation between fluid and boundaries
    r0 = pA['x'][1] - pA['x'][0]
    print(r0)
    sep = 2.5 * r0

    # Create some boundaries
    pA_bottom = helper.box(-sep, 150, -sep, -sep, r0=r0, type=ParticleType.Boundary)
    pA_left   = helper.box(-sep, -sep, -sep, 30, r0=r0, type=ParticleType.Boundary)
    pA_temp   = helper.box(25 + sep, 25 + sep, -sep, 30, r0=r0, type=ParticleType.TempBoundary)

    # Set boundary mass
    pA_bottom['m'] = mass * 1.5; pA_left['m'] = mass * 1.5; pA_temp['m'] = mass * 1.5

    # Return the giant particles
    return np.concatenate((pA, pA_bottom, pA_left, pA_temp))

def main():
    # Main parameters
    N = 2500; rho0 = 1000.0; duration = 1.0
    XSPH = True; height = 25.0; plot = False

    # Create some particles
    particles = create_particles(N, rho0)

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