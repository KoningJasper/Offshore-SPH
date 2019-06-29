import numpy as np
from math import ceil, floor
from src.Common import particle_dtype, ParticleType

class Helpers:
    @staticmethod
    def rect(xmin: float, xmax: float, ymin: float, ymax: float, r0: float, mass: float = 1.0, rho0: float = 1000.0, pack: bool = False, label: ParticleType = ParticleType.Fluid, strict: bool = False, packDirection: str = 'y') -> np.array:
        """
            Creates a particle array that represents a rectangle in a square configuration.

            Parameters
            ----------
            xmin: float
                Start of the rectangle
            xmax: float
                End of the rectangle
            ymin: float
                Start of the rectangle in y-direction
            ymax: float
                End of the rectangle in y-direction
            r0: float
                Separation between particles.
            pack: bool
                Should the particles be hex-packed, more energy efficient.
            label: ParticleType
                The particle type of the particle
            strict: bool
                Enforce strict boundary checking, hex packing shifts particles. This removes those shifted particles that shift out of the rectangle box
            
            Returns
            -------
            particles: np.array
                Particle array of type particle_dtype with x and y filled in with the rectangle values.

            Notes
            -----
            The number of actual particles can deviate from the set target if packing is set, the rectangle is not square, and strict boundary checking is not enforced.
        """

        # Ranges
        xrange = xmax - xmin
        yrange = ymax - ymin

        # Compute n
        Nx = max(1, ceil(xrange / r0))
        Ny = max(1, ceil(yrange / r0))

        # Create some particles
        xv = np.linspace(xmin, xmax, Nx)
        yv = np.linspace(ymin, ymax, Ny)
        x, y = np.meshgrid(xv, yv, indexing='ij')
        x = x.ravel(); y = y.ravel()

        # Create the particles
        pA = np.zeros(len(x), dtype=particle_dtype)
        pA['label'] = label
        pA['x']     = x
        pA['y']     = y
        pA['m']     = mass
        pA['rho']   = rho0

        # Use hex packing
        if (pack == True) and (xrange > 0):
            # Iterate the rows of particles
            if packDirection == 'y':
                for i in range(floor(Nx / 2)):
                    for p in pA[2*i*Ny : 2*i*Ny + Ny]:
                        p['y'] += r0 / 2
            elif packDirection == 'x':
                for i in range(floor(Ny / 2)):
                    for p in pA[2 * i :: Nx]:
                        p['x'] += r0 / 2
            else:
                raise Exception('Invalid packing direction.')

            if strict == True:
                # Check each particle
                saved = []
                for p in pA:
                    if p['x'] > xmax or p['x'] < xmin:
                        continue
                    if p['y'] > ymax or p['y'] < ymin:
                        continue
                    saved.append(p)

                pA = np.array(saved)
        return pA
