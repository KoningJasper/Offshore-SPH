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
from src.Equations.BoundaryForce import BoundaryForce
from src.Equations.SummationDensity import SummationDensity
from src.Tools.NNLinkedList import NNLinkedList
from src.Tools.SolverTools import computeProps, findActive, computeH, _loop

class Solver:

    # Particle information
    particleArray = None
    num_particles = 0  # Number of active particles

    # Timing information
    timing_data = {}
    
    data = [] # Complete data of last $incrementalFreq-items
    export = {} 

    settleTime = 0.0
    
    dt_a = []
    dt_c = []
    dt_f = []

    damping = 0.05 # Damping factor

    def __init__(self, method: Method, integrator: Integrator, kernel: Kernel, duration: float = 1.0, quick: bool = True, incrementalWriteout: bool = True, incrementalFile: str = "export", incrementalFreq: int = 1000, exportProperties: List[str] = ['x', 'y', 'p'], kE: float = 0.8, maxSettle: int = 500, timeStep: float = None, h: float = None, coupling = None, couplingIntegrator = None, couplingProperties = None, customSettle = None, damping: float = 0.05):
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

        timeStep: float
            Fixed time-step to use, leave None for dynamic timestep.
        
        h: float
            Fixed smoothing length, leave None for dynamic smoothing length (Monaghan, 2005).

        coupling: Function(pA: particleArray, solver: Solver) -> particleArray
            Coupling function.
        
        couplingIntegrator:
            Integrator to be used for the coupling
        
        couplingProperties: dict
            Properties for the coupling

        customSettle: Function(pA: particleArray, solver: Solver) -> bool
            A custom settling function to determine if the simulation has settled.
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
        self.timeStep  = timeStep
        self.h         = h
        self.ts_error  = 0
        self.damping   = damping
        
        # Coupling function
        self.coupling = coupling
        self.couplingIntegrator = couplingIntegrator
        self.couplingProperties = couplingProperties
        self.customSettle = customSettle

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
        self.timing_data['coupling']             = 0.0

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
        self.c_indexes = (self.particleArray['label'] == ParticleType.Coupled) & (self.particleArray['deleted'] == False)
        self.fluid_count = np.sum(self.f_indexes)
        println('{0}{1}{2} total particles, {0}{3}{2} fluid particles.'.format(Fore.YELLOW, self.num_particles, Style.RESET_ALL, self.fluid_count))

        # Calc h
        if self.h == None: 
            h = computeH(1.3, self.fluid_count, self.particleArray[self.f_indexes]['m'], self.particleArray[self.f_indexes]['rho'])
        else:
            h = np.ones_like(self.particleArray['h'][self.f_indexes]) * self.h
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

        println('{0}Setup complete.{1}'.format(Fore.GREEN, Style.RESET_ALL))

    def _minTimeStep(self) -> float:
        """ Compute the minimum timestep, based on courant and force criteria. """
        start = perf_counter()

        # CFL & Force coeffs
        gamma_c = 0.25; gamma_f = 0.25

        m, c, f = TimeStep().compute(self.fluid_count, self.particleArray[self.f_indexes], gamma_c, gamma_f)

        # Check the time-step.
        if (m < 1e-6):
            # Only show warning is non sequential.
            # if (len(self.dt_a) != self.ts_error + 1):
                # println(f'{Fore.YELLOW}WARNING! {Style.RESET_ALL}Time-step has become very small; might indicate an error.')
            self.ts_error = len(self.dt_a)

        # Store for later retrieval, making pretty plots or whatevs.
        self.dt_c.append(c); self.dt_f.append(f); self.dt_a.append(m)

        self.timing_data['time_step'] += perf_counter() - start
        return m

    def _compute(self):
        """ Compute the accelerations, velocities, etc. """
        start = perf_counter()

        # Neighbourhood
        start = perf_counter()
        self.nn.update(self.particleArray[self.indexes])
        self.timing_data['neighbour_hood'] += perf_counter() - start

        # Set h
        if self.h == None:
            h = computeH(1.3, self.fluid_count, self.particleArray[self.f_indexes]['m'], self.particleArray[self.f_indexes]['rho'])
        else:
            h = np.ones_like(self.particleArray['h'][self.f_indexes]) * self.h
        self.particleArray['h'][self.f_indexes] = h

        # Re-set accelerations for fluids only
        self.particleArray[self.f_indexes]['ax']   = 0.0
        self.particleArray[self.f_indexes]['ay']   = 0.0
        self.particleArray[self.f_indexes]['drho'] = 0.0

        # Loop
        self.particleArray[self.indexes] = _loop(self.particleArray[self.indexes], self.kernel.evaluate, self.kernel.gradient, self.method, self.nn)

        self.timing_data['compute'] += perf_counter() - start

    def run2(self):
        """ 
            Experimental fully compiled method.
            
            Remarks
            -------
            Do not use!
        """
        # Execute
        if self.timeStep == None:
            self.timeStep = -1
        data = Solver.executeRun2(self.particleArray, self.duration, self.timeStep, self.fluid_count, self.indexes, self.f_indexes, self.integrator, self.nn, self.kernel.evaluate, self.kernel.gradient, self.method)

        # Set the export
        self.export['x'] = data[:, 0, :]
        self.export['y'] = data[:, 0, :]
        self.export['p'] = data[:, 0, :]

    @staticmethod
    @njit(fastmath=True)
    def executeRun2(pA: np.array, duration: float, timeStep: float, fluid_count: int, indexes: np.array, f_indexes: np.array, integrator, nn, evFunc, gradFunc, methodClass):
        """ 
            Experimental fully compiled method.
            
            Remarks
            -------
            Do not use!
        """
        t = 0.0; step = 0
        size = 10000; data = np.zeros((len(pA), 3, size)); ts = np.zeros((3, size))
        damping = 0.0; gamma_c = 0.25; gamma_f = 0.25
        while t < duration:
            # -- Time step -- #
            c = 0.0; f = 0.0
            if timeStep == -1:
                m, c, f = TimeStep().compute(fluid_count, pA[f_indexes], gamma_c, gamma_f)
                dt = m
            else:
                dt = timeStep
            ts[0, step] = dt; ts[1, step] = c; ts[2, step] = f

            # -- Predict -- #
            pA[f_indexes] = integrator.predict(dt, pA[f_indexes], damping)

            # -- Compute -- #
            nn.update(pA[indexes])

            pA['h'][f_indexes] = computeH(1.3, fluid_count, pA[f_indexes]['m'], pA[f_indexes]['rho'])

            pA['ax'][f_indexes]   = 0.0
            pA['ay'][f_indexes]   = 0.0
            pA['drho'][f_indexes] = 0.0

            # Loop
            pA[indexes] = _loop(pA[indexes], evFunc, gradFunc, methodClass, nn)

            # -- Correct -- #
            pA[f_indexes] = integrator.correct(dt, pA[f_indexes], damping)

            # -- Store -- #
            # Resize
            if step >= size:
                print('Resizing!')
                copy = np.copy(data)
                ts_c = np.copy(ts)

                data = np.zeros((len(pA), 3, size * 2))
                ts = np.zeros((3, size * 2))

                data[:, :, 0:size-1] = copy
                ts[:, 0:size-1] = ts

                size = 2 * size
            
            # Write data
            data[:, 0, step] = np.copy(pA['x'])
            data[:, 1, step] = np.copy(pA['y'])
            data[:, 2, step] = np.copy(pA['p'])

            # Advance loop
            if step % 1000 == 0:
                print(t, ' @ ', step)
            t += dt; step += 1

        # Return the slice
        return data[:, :, 0:step]
        

    def run(self):
        """
            Runs the SPH simulation
        """
        start_all = perf_counter()

        # Check particle length.
        if len(self.particleArray) == 0 or len(self.particleArray) != self.num_particles:
            raise Exception('No or invalid particles set!')

        println('Started solving...')

        t_step = 0   # Step
        self.t = 0.0 # Current time
        
        # Create progress bar.
        println('Settling particles...')
        self.settled = False
        sbar = tqdm(total=self.maxSettle, desc='Settling', leave=False)

        while self.t < self.duration:
            # Compute time step.
            if self.timeStep == None:
                self.dt = self._minTimeStep()
            else:
                self.dt_a.append(self.timeStep) # Add to the time-step series.
                self.dt = self.timeStep

            if self.integrator.isMultiStage() == True:
                # Start with eval
                self._compute()
            
            # Predict
            start = perf_counter()
            self.particleArray[self.f_indexes] = self.integrator.predict(self.dt, self.particleArray[self.f_indexes], self.damping)
            if self.coupling != None:
                self.particleArray[self.c_indexes] = self.couplingIntegrator.predict(self.dt, self.particleArray[self.c_indexes], self.damping)
            self.timing_data['integrate_prediction'] += perf_counter() - start

            # Compute the accelerations
            self._compute()

            # Coupling to user-defined function.
            if self.coupling != None:
                start = perf_counter()
                self.particleArray = self.coupling(self.particleArray, self)
                self.timing_data['coupling'] += perf_counter() - start

            # Correct
            start = perf_counter()
            self.particleArray[self.f_indexes] = self.integrator.correct(self.dt, self.particleArray[self.f_indexes], self.damping)
            if self.coupling != None:
                self.particleArray[self.c_indexes] = self.couplingIntegrator.correct(self.dt, self.particleArray[self.c_indexes], self.damping)
            self.timing_data['integrate_correction'] += perf_counter() - start

            # Store to data
            start = perf_counter()
            self._store(t_step)
            self.timing_data['storage'] += perf_counter() - start

            # End integration-loop
            if self.settled == True:
                # Only move forward if damping
                self.t += self.dt
            t_step += 1

            if self.settled == False and t_step > 1:
                ke = 1e12; cs = False
                if self.customSettle == None:
                    # Compute kinetic energy
                    ke = KineticEnergy(self.num_particles, self.particleArray[self.f_indexes])
                else:
                    cs = self.customSettle(self.particleArray, self)

                if (ke < self.kE) or (t_step > self.maxSettle) or (cs == True):
                    if (t_step > self.maxSettle):
                        # Print extra warning.
                        println('{0}WARNING!{1} Maximum settle steps reached, check configuration, maybe increase spacing between wall and particles.'.format(Fore.YELLOW, Style.RESET_ALL))
                        
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
                    self.damping = 0.0

                    self.settled = True
                    println('{0}Settling Complete.{1}'.format(Fore.GREEN, Style.RESET_ALL))
                    
                    # Create progress bar.
                    tbar = tqdm(total=self.duration, desc='Time-stepping', unit='s', leave=False)
                else:
                    sbar.update(1)
            # End self.settled

            # Only keep self.incrementalFreq in memory.
            if len(self.data) > self.incrementalFreq:
                self.data.pop(0)

            # Update tbar
            if self.settled == True:
                tbar.update(self.dt)
        # End while

        tbar.close()
        self.timing_data['total'] = perf_counter() - start_all

        total = self.timing_data['total']
        self.t_step = t_step
        println('{0}Solved!{1}'.format(Fore.GREEN, Style.RESET_ALL))
        println('Solved {0}{2}{1} particles for {0}{3:f}{1} [s].'.format(Fore.YELLOW, Style.RESET_ALL, self.num_particles, self.duration))
        println('Completed solve in {0}{2:f}{1} [s] and {0}{3}{1} steps'.format(Fore.YELLOW, Style.RESET_ALL, total, t_step))

    def _store(self, t_step: int):
        self.data.append(np.copy(self.particleArray))

        # Add the export properties
        for key in self.exportProperties:
            self.export[key].append(np.copy(self.particleArray[key]))
        
        # Incremental writeout
        if self.incrementalWriteout and t_step % self.incrementalFreq == 0:
            self.save('{0}-{1}.hdf5'.format(self.incrementalFile, t_step), printLocation=False)


    def timing(self):
        t = PrettyTable(['Name', 'Time [s]', 'Percentage [%]'])
        for k, v in self.timing_data.items():
            t.add_row([k, round(v, 3), round(v / self.timing_data['total'] * 100, 2)])

        println('Detailed timing statistics:')
        println(t)

    def save(self, location: str, printLocation: bool = True, extraProperties: dict = None):
        """
            Saves the output to a compressed export .hdf5 file.

            Parameters
            ----------

            location: str
                Path to export arrays to, should include .hdf5
            printLocation: bool
                Should the location of the file be printed?
            extraProperties: dict
                Extra properties to be exported to the file.
        """
        println('Starting file export.')

        # Export hdf5 file
        with h5py.File(location, 'w') as h5f:
            h5f.create_dataset('particleArray', data=self.particleArray, shuffle=True, compression="gzip")
            h5f.create_dataset('dt_a', data=self.dt_a, shuffle=True, compression="gzip")
            h5f.create_dataset('dt_c', data=self.dt_c, shuffle=True, compression="gzip")
            h5f.create_dataset('dt_f', data=self.dt_f, shuffle=True, compression="gzip")
            h5f.create_dataset('settleTime', data=self.settleTime)

            println('Starting key export.')

            for key in self.exportProperties:
                h5f.create_dataset(key, data=np.stack(self.export[key]), shuffle=True, compression="gzip")

            if extraProperties:
                for key, value in extraProperties.items():
                    h5f.create_dataset(key, data=value)

        if printLocation == True:
            println('Exported arrays to: "{0}".'.format(location))
    
def println(text: str):
    print('\n{0}'.format(text))