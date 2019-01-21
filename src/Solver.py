import numpy as np
from scipy.spatial.distance import cdist
from numba import prange, autojit
from typing import List
from src.Particle import Particle
from src.Methods.Method import Method
from src.Kernels.Kernel import Kernel
from src.Integrators.Integrator import Integrator


class Solver:
    """
        Solver interface.
    """

    method: Method
    integrator: Integrator
    kernel: Kernel
    duration: float
    dt: float

    # Particle information
    particles: List[Particle]
    num_particles: int

    def __init__(self, method: Method, integrator: Integrator, kernel: Kernel, duration: float, dt: float):
        """
            Parameters
            ----------

            method: The method to use for the solving of SPH particles (WCSPH/ICSPH).

            integrator: The integration method to be used by the solver (Euler/Verlet).

            kernel: The kernel function to use in the evaluation (Gaussian/CubicSpline).

            duration: The duration of the simulation in seconds.

            dt: timestep [s]
        """

        self.method     = method
        self.integrator = integrator
        self.kernel     = kernel
        self.duration   = duration
        
        # TODO: Remove
        self.dt = dt

    def addParticle(self, particles):
        self.particles.append(particles)

    def setup(self):
        self.num_particles: int = len(self.particles)
        time_step_guess: int = self.duration / self.dt

        # Empty time arrays
        self.x = np.zeros([self.num_particles, time_step_guess]) # X-pos
        self.y = np.zeros([self.num_particles, time_step_guess]) # Y-pos
        self.c = np.zeros([self.num_particles, time_step_guess]) # Pressure (p)
        self.u = np.zeros([self.num_particles, time_step_guess]) # Density (rho)
        self.drho = np.zeros([self.num_particles, time_step_guess]) # Density (rho)
        self.v = np.zeros([self.num_particles, 2, time_step_guess]) # Velocity both x and y
        self.a = np.zeros([self.num_particles, 2, time_step_guess]) # Acceleration both x and y

        # Initialization
        for p in self.particles:
            self.method.initialize(p)

        # Set 0-th time-step
        self.x[:, 0] = [p.r[0] for p in self.particles]
        self.y[:, 0] = [p.r[1] for p in self.particles]
        self.c[:, 0] = [p.p for p in self.particles]
        self.u[:, 0] = [p.rho for p in self.particles]

    def solve(self):
        """
            Solve the equations setup
        """
        t_step: int = 0   # Step
        t: float    = 0.0 # Current time
        
        # Keep integrating until simulation duration is reached.
        # TODO: Change giant-matrix size if wrong due to different time-steps.
        while t < self.duration:
            # TODO: Move this to separate class and function.
            # Distance and neighbourhood
            r = np.array([p.r for p in self.particles])
            dist = cdist(r, r, 'euclidean')

            # Distance of closest particle time 1.3
            # TODO: Move to separate class.
            h: np.array = 1.3 * np.ma.masked_values(dist, 0.0, copy=False).min(1)

            # Acceleration and force loop
            for i in range(self.num_particles):
                p = self.particles[i]

                # Run EOS
                p.p = self.method.compute_pressure(p)

                # Query neighbours
                h_i: np.array = 0.5 * (h[i] + h[:])
                q_i: np.array = dist[i, :] / (h_i)
                near_arr: np.array = np.flatnonzero(np.argwhere(q_i <= 3.0)) # Find neighbours and remove self (0).

                # Skip if got no neighbours
                # Keep same properties, no acceleration.
                if len(near_arr) == 0:
                    continue

                # Calc vectors
                xij: np.array = p.r - r[near_arr, :]
                rij: np.array = dist[i, near_arr]
                vij: np.array = p.v - self.v[near_arr, :, t_step]

                # Calc averaged properties
                hij: np.array = h_i[near_arr]
                cij: np.array = np.ones(len(near_arr)) * self.method.compute_speed_of_sound(p) # This is a constant (currently).

                # kernel calculations
                wij = self.kernel.evaluate(xij, rij, hij)
                dwij = self.kernel.gradient(xij, rij, hij)

                # Continuity
                p.drho = self.method.compute_density_change(p, vij, dwij)

                # Gradient 
                if p.label == 'fluid':
                    p.a = self.method.compute_acceleration(p, xij, rij, vij, self.c[near_arr, t_step], self.u[near_arr, t_step], hij, cij, wij, dwij)
                    p.v = self.method.compute_velocity(p)
                # end fluid
            # end acc. loop

            # Integration loop
            for i in range(self.num_particles):
                # Select the current particle
                p = self.particles[i]

                # Integrate
                self.integrator.integrate(self.dt, p)

                # Put into giant-matrix
                self.x[i, t_step + 1] = p.r[0]
                self.y[i, t_step + 1] = p.r[1]
                self.c[i, t_step + 1] = p.p
                self.u[i, t_step + 1] = p.rho
                self.v[i, :, t_step + 1] = p.v
                self.a[i, :, t_step + 1] = p.a
                self.drho[i, t_step + 1] = p.drho

            # End integration-loop
            t += self.dt
            t_step += 1

    def save(self):
        """
            Saves the output to a compressed export .npz file.
        """
        
        # Export compressed numpy-arrays
        np.savez_compressed(f'{sys.path[0]}/export', x=self.x, y=self.y, p=self.c, rho=self.u, v=self.v, a=self.a, drho=self.drho)
        print(f'Exported arrays to: "{sys.path[0]}/export.npz".')