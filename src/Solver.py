# Import python packages
import sys
import math

# Other packages
import numpy as np
import h5py
from colorama import Fore, Style
from prettytable import PrettyTable
from tqdm import tqdm
from scipy.spatial.distance import cdist
from numba import prange, jit, njit, cfunc
from typing import List, Tuple, Dict
from time import perf_counter

# Own components
from src.Common import particle_dtype, computed_dtype, ParticleType, get_label_code
from src.Particle import Particle
from src.Methods.Method import Method
from src.Kernels.Kernel import Kernel
from src.Integrators.Integrator import Integrator
from src.Equations.TimeStep import TimeStep
from src.Tools.NearNeighbours import NearNeighbours
from src.Tools.SolverTools import computeProps, findActive

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
    particles: List[Particle] = []
    particleArray: np.array
    num_particles: int = 0  # Number of active particles
    indexes: np.array       # Indexes of active particles

    # Timing information
    timing_data: Dict[str, float] = {}
    
    data: List[np.array] = [] # Complete data of last $incrementalFreq-items
    export: Dict[str, List[np.array]] = {} 

    t_step: int
    settleTime: float = 0.0
    
    dt_a: List[float] = []
    dt_c: List[float] = []
    dt_f: List[float] = []

    damping: float = 0.05 # Damping factor

    def __init__(self, method: Method, integrator: Integrator, kernel: Kernel, duration: float = 1.0, quick: bool = True, incrementalWriteout: bool = True, incrementalFile: str = "export", incrementalFreq: int = 1000, exportProperties: List[str] = ['x', 'y', 'p']):
        """
        Initializes a new solver object. The solver orchestrates the entire solving of the SPH simulation.

        Parameters
        ----------

        method: src.Methods.Method
            The method to use for the solving of SPH particles (WCSPH/ICSPH).

        integrator: src.Interators.Integrator
            The integration method to be used by the solver (Euler/Verlet).

        kernel: src.Kernels.Kernel
            The kernel function to use in the evaluation (Gaussian/CubicSpline).

        duration: float
            The duration of the simulation in seconds.

        quick: bool
            Makes larger timesteps, if an explosion occurs: winds back and retries with a smaller timestep

        incrementalWriteout: bool
            Write out the data incrementally

        incrementalFile: str
            File to write-out to.

        incrementalFreq: int
            Increment to write to storage in timesteps.

        exportProperties: List[str]
            Properties to include in the final export, more properties will require more storage.
        """

        # Initialize classes
        self.method     = method
        self.integrator = integrator
        self.kernel     = kernel
        self.nn         = NearNeighbours()

        # Set properties
        self.duration = duration
        self.quick    = quick

        # Incremental write-out
        self.incrementalWriteout = incrementalWriteout
        self.incrementalFreq = incrementalFreq
        self.incrementalFile = incrementalFile
        self.exportProperties = exportProperties

        # Initialize timing
        self.timing_data['total']                = 0.0
        self.timing_data['storage']              = 0.0
        self.timing_data['integrate_correction'] = 0.0
        self.timing_data['integrate_prediction'] = 0.0
        self.timing_data['compute']              = 0.0
        self.timing_data['time_step']            = 0.0
        self.timing_data['neighbour_hood']       = 0.0

    def load(self, file: str):
        """
            Continue the simulation based on a previous file.

        Parameters
        ----------

        file: str
            Path to the file, including .hdf5
        """

        with h5py.File(file, 'r') as h5f:
            self.particleArray = h5f['particleArray'][:]
            self.dt_a = h5f['dt_a'][:]
            self.dt_c = h5f['dt_c'][:]
            self.dt_f = h5f['dt_f'][:]

    def addParticles(self, particles: List[Particle]):
        """
            Adds the particles to the simulation.

            Parameters
            ----------

            particles: List[Particle]
                particles to add to the simulation, should be a list of particle objects.    
        """
        self.particles.extend(particles)

    def _convertParticles(self):
        """ Convert the particle classes to a numpy array. """
        # Init empty array
        self.num_particles = len(self.particles)
        self.particleArray = np.zeros(self.num_particles, dtype=particle_dtype)
        for i, p in enumerate(self.particles):
            pA          = self.particleArray[i]
            pA['label'] = get_label_code(p.label)
            pA['m']     = p.m
            pA['rho'] = p.rho
            pA['p'] = p.p
            pA['x'] = p.r[0]
            pA['y'] = p.r[1]

    def setup(self):
        """ Sets-up the solver, with the required parameters. """
        # Convert the particle array.
        self._convertParticles()

        # Find the active ones.
        self.num_particles, self.indexes = findActive(self.num_particles, self.particleArray)

        # Calc h
        (_, _, h) = self._nbrs()
        self.particleArray['h'][self.indexes] = h

        # Initialize particles
        self.particleArray[self.indexes] = self.method.initialize(self.particleArray[self.indexes])

        # Compute some initial properties
        self.particleArray['p'][self.indexes] = self.method.compute_pressure(self.particleArray[self.indexes])
        self.particleArray['c'][self.indexes] = self.method.compute_speed_of_sound(self.particleArray[self.indexes])

        # Set 0-th time-step
        self.data.append(self.particleArray[:])

        # Create export lists, and store 0-th timestep.
        for key in self.exportProperties:
            self.export[key] = []

        println(f'{Fore.GREEN}Setup complete.{Style.RESET_ALL}')

    @jit(fastmath=True, cache=True)
    def _minTimeStep(self) -> float:
        """ Compute the minimum timestep, based on courant and force criteria. """
        start = perf_counter()

        gamma_c = 0.4
        gamma_f = 0.25
        if self.quick == True:
            gamma_f = 0.6

        m, c, f = TimeStep.compute(self.num_particles, self.particleArray[self.indexes], gamma_c, gamma_f)

        # Store for later retrieval, making pretty plots or whatevs.
        self.dt_c.append(c); self.dt_f.append(f); self.dt_a.append(m)

        self.timing_data['time_step'] += perf_counter() - start
        return m

    @staticmethod
    @njit('float64[:](float64, int64, float64[:], float64[:])', fastmath=True, cache=True)
    def computeH(sigma: float, J: int, m: np.array, rho: np.array):
        """ Compute (dynamic) h size, based on Monaghan 2005. """
        d = 2 # 2 Dimensions (x, y)
        f = 1 / d
        h = np.zeros_like(m)
        for j in prange(J):
            h[j] = sigma * (m[j] / rho[j]) ** f
        
        return h

    @jit(fastmath=True, cache=True)
    def _nbrs(self):
        start = perf_counter()

        pA = self.particleArray[self.indexes]

        # Distance and neighbourhood
        r = np.transpose(np.vstack((pA['x'], pA['y'])))
        dist = cdist(r, r, 'euclidean')

        # Distance of closest particle time 1.3
        h = Solver.computeH(1.3, self.num_particles, pA['m'], pA['rho'])
        #h: np.array = 1.3 * np.ma.masked_values(dist, 0.0, copy=False).min(1)

        self.timing_data['neighbour_hood'] += perf_counter() - start
        return (r, dist, h)

    @staticmethod
    @njit(fastmath=True, parallel=True, cache=True)
    def _loop(h, dist, pA, evFunc, gradFunc, methodClass, nn):
        p = methodClass.compute_pressure(pA)
        c = methodClass.compute_speed_of_sound(pA)

        for i in prange(len(pA)):
            # Assign parameters
            pA[i]['p'] = p[i]
            pA[i]['c'] = c[i]

            # Find near neighbours and their h and q
            h_i, q_i, near_arr = nn.near(i, h, dist[i, :])

            # Skip if got no neighbours, early exit.
            # Keep same properties, no acceleration.
            if len(near_arr) == 0:
                continue

            # Create computed properties
            # Fill the props
            calcProps = computeProps(i, pA, near_arr, h_i, q_i, dist, evFunc, gradFunc)

            # Continuity
            pA[i]['drho'] = methodClass.compute_density_change(pA[i], calcProps)

            if pA[i]['label'] == ParticleType.Fluid:
                # Momentum
                [pA[i]['ax'], pA[i]['ay']] = methodClass.compute_acceleration(pA[i], calcProps)

                # XSPH
                [pA[i]['vx'], pA[i]['vy'], pA[i]['xsphx'], pA[i]['xsphy']] = methodClass.compute_velocity(pA[i], calcProps)
            # end fluid

        return pA

    def _compute(self):
        """ Compute the accelerations, velocities, etc. """
        start = perf_counter()

        # Neighbourhood
        (_, dist, h) = self._nbrs()

        # Set h
        self.particleArray['h'][self.indexes] = h

        # Re-set accelerations
        self.particleArray['ax'][self.indexes]   = 0.0
        self.particleArray['ay'][self.indexes]   = 0.0
        self.particleArray['drho'][self.indexes] = 0.0

        # Loop
        self.particleArray[self.indexes] = Solver._loop(h, dist, self.particleArray[self.indexes], self.kernel.evaluate, self.kernel.gradient, self.method, self.nn)

        self.timing_data['compute'] += perf_counter() - start

    def run(self):
        """
            Runs the SPH simulation
        """
        start_all = perf_counter()

        # Check particle length.
        if len(self.particles) == 0 or len(self.particles) != self.num_particles:
            raise Exception('No or invalid particles set!')

        t_step: int = 0   # Step
        t: float    = 0.0 # Current time
        println('Started solving...')
        settleSteps = 100
        sbar = tqdm(total=settleSteps, desc='Settling', unit='steps', leave=False)

        t_fail: float = 0.0 # Time at failure
        t_stepback: int = 10
        dt_threshold: float = 0.05 # Explosion treshold
        dt_scale: float = 0.5 # Scale with half after explosion.

        while t < self.duration:
            # Compute time step.
            self.dt = self._minTimeStep()

            # Check explosion
            if self.dt > dt_threshold:
                println(f'{Fore.YELLOW}Warning: {Style.RESET_ALL} suspected explosion at t = {t} [s]; Check time-step.')

                if self.quick == True:
                    println(f'Rolling back {t_stepback} timesteps')
                    t_fail = t

                    # Restore previous data
                    self.particleArray = self.data[-t_stepback]

            if t_fail > 1e-12:
                # Failed
                if t >= t_fail:
                    # Unfailed
                    t_fail = 0.0
                else:
                    # Scale dt
                    self.dt = self.dt * dt_scale

            if self.integrator.isMultiStage() == True:
                # Start with eval
                self._compute()
            
            # Predict
            start = perf_counter()
            self.particleArray[self.indexes] = self.integrator.predict(self.dt, self.particleArray[self.indexes], self.damping)
            self.timing_data['integrate_prediction'] += perf_counter() - start

            # Compute the accelerations
            self._compute()

            # Correct
            start = perf_counter()
            self.particleArray[self.indexes] = self.integrator.correct(self.dt, self.particleArray[self.indexes], self.damping)
            self.timing_data['integrate_correction'] += perf_counter() - start

            # Store to data
            start = perf_counter()
            self._store(t_step)
            self.timing_data['storage'] += perf_counter() - start

            # End integration-loop
            if self.damping == 0:
                # Only move forward if damping
                t += self.dt
            t_step += 1

            if t_step > settleSteps:
                if self.damping > 0:
                    sbar.update(n=1)
                    sbar.close()
                    println(f'{Fore.GREEN}Settling Complete.{Style.RESET_ALL}')

                    # Create progress bar.
                    tbar = tqdm(total=self.duration, desc='Time-stepping', unit='s', leave=False)

                    # Remove temp-boundary
                    inds = []
                    for i in range(self.num_particles):
                        p = self.particleArray[i]
                        if p['label'] == ParticleType.TempBoundary:
                            inds.append(i)
                            p['deleted'] = True
                    self.num_particles, self.indexes = findActive(self.num_particles, self.particleArray)

                    # Set the pressure to -1000.0 for deleted particles.
                    self.particleArray['p'][np.array(inds)] = -1e15

                    # Set settling time
                    self.settleTime = t
                    
                # Stop damping after 100-th time steps.
                self.damping = 0.0

            # Only keep self.incrementalFreq in memory.
            if len(self.data) > self.incrementalFreq:
                self.data.pop(0)

            # Update tbar
            if self.damping == 0:
                tbar.update(self.dt)
            else:
                sbar.update(n=1)
        # End while

        tbar.close()
        self.timing_data['total'] = perf_counter() - start_all

        total = self.timing_data['total']
        self.t_step = t_step
        println(f'{Fore.GREEN}Solved!{Style.RESET_ALL}')
        println(f'Solved {Fore.YELLOW}{self.num_particles}{Style.RESET_ALL} particles for {Fore.YELLOW}{self.duration:f}{Style.RESET_ALL} [s].')
        println(f'Completed solve in {Fore.YELLOW}{total:f}{Style.RESET_ALL} [s] and {Fore.YELLOW}{t_step}{Style.RESET_ALL} steps')

    def _store(self, t_step: int):
        self.data.append(np.copy(self.particleArray))

        # Add the export properties
        for key in self.exportProperties:
            self.export[key].append(np.copy(self.particleArray[key]))
        
        # Incremental writeout
        if self.incrementalWriteout and t_step % self.incrementalFreq == 0:
            self.save(f'{self.incrementalFile}-{t_step}.hdf5', printLocation=False)


    def timing(self):
        t = PrettyTable(['Name', 'Time [s]', 'Percentage [%]'])
        for k, v in self.timing_data.items():
            t.add_row([k, round(v, 3), round(v / self.timing_data['total'] * 100, 2)])

        println('Detailed timing statistics:')
        println(t)

    def save(self, location: str, printLocation: bool = True):
        """
            Saves the output to a compressed export .hdf5 file.

            Parameters
            ----------

            location: str
                Path to export arrays to, should include .hdf5
        """
        # Export hdf5 file
        with h5py.File(location, 'w') as h5f:
            h5f.create_dataset('particleArray', data=self.particleArray, shuffle=True, compression="gzip")
            h5f.create_dataset('dt_a', data=self.dt_a, shuffle=True, compression="gzip")
            h5f.create_dataset('dt_c', data=self.dt_c, shuffle=True, compression="gzip")
            h5f.create_dataset('dt_f', data=self.dt_f, shuffle=True, compression="gzip")
            h5f.create_dataset('settleTime', data=np.array(self.settleTime))

            for key in self.exportProperties:
                h5f.create_dataset(key, data=np.stack(self.export[key]), shuffle=True, compression="gzip")

        if printLocation == True:
            println(f'Exported arrays to: "{location}".')
    
def println(text: str):
    print(f'\n{text}')