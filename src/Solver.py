# Import python packages
import sys, math
from typing import List, Tuple, Dict
from time import perf_counter

# Other packages
import numpy as np
import h5py
from colorama import Fore, Style
from prettytable import PrettyTable
from tqdm import tqdm
from numba import prange, jit, njit

# Own components
from src.Common import particle_dtype, computed_dtype, ParticleType
from src.Methods.Method import Method
from src.Kernels.Kernel import Kernel
from src.Integrators.Integrator import Integrator
from src.Equations.TimeStep import TimeStep
from src.Equations.KineticEnergy import KineticEnergy
from src.Equations import BoundaryForce
from src.Tools.NNLinkedList import NNLinkedList
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
    particleArray: np.array = None
    num_particles: int = 0  # Number of active particles
    indexes: np.array       # Indexes of active particles
    f_indexes: np.array     # Indexes of active fluid-particles

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

    def __init__(self, method: Method, integrator: Integrator, kernel: Kernel, duration: float = 1.0, quick: bool = True, incrementalWriteout: bool = True, incrementalFile: str = "export", incrementalFreq: int = 1000, exportProperties: List[str] = ['x', 'y', 'p'], kE: float = 0.8, maxSettle: int = 500):
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

        kE: float
            Kinetic energy factor when to stop settling, as fraction of initial kinetic energy. Default: 0.8
        
        maxSettle: int
            Maximum number of timesteps to take during settling.
        """

        # Initialize classes
        self.method     = method
        self.integrator = integrator
        self.kernel     = kernel
        self.nn         = NNLinkedList(scale=2.0)

        # Set properties
        self.duration  = duration
        self.quick     = quick
        self.kEF       = kE
        self.maxSettle = maxSettle

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

    def addParticles(self, particles: np.array):
        """
            Adds the particles to the simulation.
            Parameters
            ----------
            particles: np.array
                particles to add to the simulation, should be a numpy array with particle_dtype as dtype.    
        """
        if self.particleArray == None:
            self.particleArray = particles
        else:
            self.particleArray = np.concatenate(self.particleArray, particles)

    def setup(self):
        """ Sets-up the solver, with the required parameters. """
        println('Starting setup.')

        # Find the active ones.
        self.num_particles, self.indexes = findActive(self.num_particles, self.particleArray)
        self.f_indexes = (self.particleArray['label'] == ParticleType.Fluid) & (self.particleArray['deleted'] == False)
        self.fluid_count = np.sum(self.f_indexes)
        println(f'{Fore.YELLOW}{self.num_particles}{Style.RESET_ALL} total particles, {Fore.YELLOW}{self.fluid_count}{Style.RESET_ALL} fluid particles.')

        # Calc h
        h = Solver.computeH(1.3, self.fluid_count, self.particleArray[self.f_indexes]['m'], self.particleArray[self.f_indexes]['rho'])
        self.particleArray['h'][self.f_indexes] = h

        # Initialize particles
        self.particleArray[self.f_indexes] = self.method.initialize(self.particleArray[self.f_indexes])

        # Compute some initial properties
        self.particleArray['p'][self.f_indexes] = self.method.compute_pressure(self.particleArray[self.f_indexes])
        self.particleArray['c'][self.f_indexes] = self.method.compute_speed_of_sound(self.particleArray[self.f_indexes])

        # Set 0-th time-step
        self.data.append(self.particleArray[:])

        # Compute initial kinetic energy
        self.kE = KineticEnergy(self.fluid_count, self.particleArray[self.f_indexes]) * self.kEF

        # Create export lists, and store 0-th timestep.
        for key in self.exportProperties:
            self.export[key] = []

        println(f'{Fore.GREEN}Setup complete.{Style.RESET_ALL}')

    @jit(fastmath=True)
    def _minTimeStep(self) -> float:
        """ Compute the minimum timestep, based on courant and force criteria. """
        start = perf_counter()

        gamma_c = 0.4
        gamma_f = 0.25
        if self.quick == True:
            # Determined by pure guess work.
            gamma_f = 0.6

        m, c, f = TimeStep.compute(self.fluid_count, self.particleArray[self.f_indexes], gamma_c, gamma_f)
        
        # Floor at 5th decimal
        # m = round(math.floor(m / 0.00001) * 0.00001, 5)

        # Store for later retrieval, making pretty plots or whatevs.
        self.dt_c.append(c); self.dt_f.append(f); self.dt_a.append(m)

        self.timing_data['time_step'] += perf_counter() - start
        return m

    @staticmethod
    @njit('float64[:](float64, int64, float64[:], float64[:])', fastmath=True)
    def computeH(sigma: float, J: int, m: np.array, rho: np.array):
        """ Compute (dynamic) h size, based on Monaghan 2005. """
        d = 2 # 2 Dimensions (x, y)
        f = 1 / d
        h = np.zeros_like(m)
        
        # Outside of compute loop so prange can be used.
        for j in prange(J):
            h[j] = sigma * (m[j] / rho[j]) ** f
        
        return h

    @staticmethod
    @njit(fastmath=True)
    def _loop(pA, evFunc, gradFunc, methodClass, nn):
        p = methodClass.compute_pressure(pA)
        c = methodClass.compute_speed_of_sound(pA)

        for i in prange(len(pA)):
            if pA[i]['label'] != ParticleType.Fluid:
                continue;

            # Assign parameters
            pA[i]['p'] = p[i]
            pA[i]['c'] = c[i]

            # Find near neighbours and their h and q
            h_i, q_i, dist, near_arr = nn.near(i, pA)

            # Skip if got no neighbours, early exit.
            # Keep same properties, no acceleration.
            if len(near_arr) == 0:
                continue

            # Create computed properties
            # Fill the props
            calcProps = computeProps(i, pA, near_arr, h_i, q_i, dist, evFunc, gradFunc)

            # Continuity
            pA[i]['drho'] = methodClass.compute_density_change(pA[i], calcProps)

            # Momentum
            [a_x, a_y] = methodClass.compute_acceleration(pA[i], calcProps)
            
            # Compute boundary forces
            [b_x, b_y] = BoundaryForce.BoundaryForce(methodClass.r0, methodClass.D, methodClass.p1, methodClass.p2, pA[i], calcProps)

            # Total acceleration
            pA[i]['ax'] = a_x + b_x
            pA[i]['ay'] = a_y + b_y

            # XSPH
            [pA[i]['vx'], pA[i]['vy']] = methodClass.compute_velocity(pA[i], calcProps)

        return pA

    def _compute(self):
        """ Compute the accelerations, velocities, etc. """
        start = perf_counter()

        # Neighbourhood
        start = perf_counter()
        self.nn.update(self.particleArray[self.indexes])
        self.timing_data['neighbour_hood'] += perf_counter() - start

        # Set h
        h = Solver.computeH(1.3, self.fluid_count, self.particleArray[self.f_indexes]['m'], self.particleArray[self.f_indexes]['rho'])
        self.particleArray['h'][self.f_indexes] = h

        # Re-set accelerations
        self.particleArray['ax'][self.indexes]   = 0.0
        self.particleArray['ay'][self.indexes]   = 0.0
        self.particleArray['drho'][self.indexes] = 0.0

        # Loop
        self.particleArray[self.indexes] = Solver._loop(self.particleArray[self.indexes], self.kernel.evaluate, self.kernel.gradient, self.method, self.nn)

        self.timing_data['compute'] += perf_counter() - start

    def run(self):
        """
            Runs the SPH simulation
        """
        start_all = perf_counter()

        # Check particle length.
        if len(self.particleArray) == 0 or len(self.particleArray) != self.num_particles:
            raise Exception('No or invalid particles set!')

        println('Started solving...')

        t_step: int = 0   # Step
        t: float    = 0.0 # Current time
        
        # Create progress bar.
        println('Settling particles...')
        settled: bool = False
        sbar = tqdm(total=self.maxSettle, desc='Settling', leave=False)

        while t < self.duration:
            # Compute time step.
            self.dt = self._minTimeStep()

            if self.integrator.isMultiStage() == True:
                # Start with eval
                self._compute()
            
            # Predict
            start = perf_counter()
            self.particleArray[self.f_indexes] = self.integrator.predict(self.dt, self.particleArray[self.f_indexes], self.damping)
            self.timing_data['integrate_prediction'] += perf_counter() - start

            # Compute the accelerations
            self._compute()

            # Correct
            start = perf_counter()
            self.particleArray[self.f_indexes] = self.integrator.correct(self.dt, self.particleArray[self.f_indexes], self.damping)
            self.timing_data['integrate_correction'] += perf_counter() - start

            # Store to data
            start = perf_counter()
            self._store(t_step)
            self.timing_data['storage'] += perf_counter() - start

            # End integration-loop
            if settled == True:
                # Only move forward if damping
                t += self.dt
            t_step += 1

            if settled == False and t_step > 1:
                # Compute kinetic energy
                ke = KineticEnergy(self.fluid_count, self.particleArray[self.f_indexes])
                if ke < self.kE or t_step > self.maxSettle:
                    if t_step > self.maxSettle:
                        println(f'{Fore.YELLOW}WARNING!{Style.RESET_ALL} Maximum settle steps reached, check configuration, maybe increase spacing between wall and particles.')
                    # Remove the settling bar
                    sbar.close()

                    # Remove temp-boundary
                    inds = []
                    for i in range(self.num_particles):
                        p = self.particleArray[i]
                        if p['label'] == ParticleType.TempBoundary:
                            inds.append(i)
                            p['deleted'] = True

                    # Get the new indexes
                    self.num_particles, self.indexes = findActive(self.num_particles, self.particleArray)
                    self.f_indexes = (self.particleArray['label'] == ParticleType.Fluid) & (self.particleArray['deleted'] == False)
                    self.fluid_count = np.sum(self.f_indexes)

                    # Set the pressure to - a lot for deleted particles.
                    self.particleArray['p'][inds] = -1e15

                    # Set settling time
                    self.settleTime = sum(self.dt_a)
                    
                    # Stop damping after reaching settling kinetic energy
                    self.damping = 1e-4

                    settled = True
                    println(f'{Fore.GREEN}Settling Complete.{Style.RESET_ALL}')

                    # Create progress bar.
                    tbar = tqdm(total=self.duration, desc='Time-stepping', unit='s', leave=False)
                else:
                    sbar.update(1)
            # End settled

            # Only keep self.incrementalFreq in memory.
            if len(self.data) > self.incrementalFreq:
                self.data.pop(0)

            # Update tbar
            if settled == True:
                tbar.update(self.dt)
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
            h5f.create_dataset('settleTime', data=self.settleTime)

            for key in self.exportProperties:
                h5f.create_dataset(key, data=np.stack(self.export[key]), shuffle=True, compression="gzip")

        if printLocation == True:
            println(f'Exported arrays to: "{location}".')
    
def println(text: str):
    print(f'\n{text}')