# Import python packages
import sys
import tempfile
import subprocess
import math

# Other packages
import numpy as np
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
from src.ColorBar import ColorBar
from src.Equations.Courant import Courant

# Plotting
import pyqtgraph as pg
import pyqtgraph.exporters
from pyqtgraph.Qt import QtCore
from pyqtgraph.graphicsItems import TextItem
from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

class Solver:
    """
        Solver interface.
    """

    method: Method
    integrator: Integrator
    kernel: Kernel
    duration: float
    dt: float
    plot: bool

    # Particle information
    particles: List[Particle] = []
    particleArray: np.array
    num_particles: int = 0

    # Timing information
    timing_data: Dict[str, float] = {}

    data: List[np.array] = []

    t_step: int
    dts: List[float] = []
    damping: float = 0.05 # Damping factor

    def __init__(self, method: Method, integrator: Integrator, kernel: Kernel, duration: float = 1.0, plot: bool = False):
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

        plot: bool
            Should a plot be created?
        """

        self.method     = method
        self.integrator = integrator
        self.kernel     = kernel
        self.duration   = duration
        self.plot       = plot

        # Initialize timing
        self.timing_data['total']                = 0.0
        self.timing_data['storage']              = 0.0
        self.timing_data['export']               = 0.0
        self.timing_data['integrate_correction'] = 0.0
        self.timing_data['integrate_prediction'] = 0.0
        self.timing_data['compute']              = 0.0
        self.timing_data['time_step']            = 0.0
        self.timing_data['neighbour_hood']       = 0.0

    def addParticles(self, particles: List[Particle]):
        self.particles.extend(particles)

    def _convertParticles(self):
        """ Convert the particle classes to a numpy array. """
        # Init empty array
        self.num_particles = len(self.particles)
        self.particleArray = np.zeros(self.num_particles, dtype=particle_dtype)
        for i, p in enumerate(self.particles):
            pA = self.particleArray[i]
            pA['label'] = get_label_code(p.label)
            pA['m'] = p.m
            pA['rho'] = p.rho
            pA['p'] = p.p
            pA['x'] = p.r[0]
            pA['y'] = p.r[1]

    def setup(self):
        """ Sets-up the solver, with the required parameters. """
        # Convert the particle array.
        self._convertParticles()

        # Calc h
        (_, _, h) = self._nbrs()
        self.particleArray['h'] = h

        # Initialize particles
        self.particleArray      = self.method.initialize(self.particleArray)

        # Compute some initial properties
        self.particleArray['p'] = self.method.compute_pressure(self.particleArray)
        self.particleArray['c'] = self.method.compute_speed_of_sound(self.particleArray)

        # Set 0-th time-step
        self.data.append(self.particleArray)

        if self.plot == True:
            self.init_plot()
            self.export(0)

        print(f'{Fore.GREEN}Setup complete.{Style.RESET_ALL}')

    @jit
    def _minTimeStep(self) -> float:
        start = perf_counter()

        dt = Courant(0.4, self.particleArray['h'], self.particleArray['c']) * (1 / 1.3)
        self.dts.append(dt)

        self.timing_data['time_step'] += perf_counter() - start
        return dt

    @jit
    def _nbrs(self):
        start = perf_counter()

        # Distance and neighbourhood
        r = np.transpose(np.vstack((self.particleArray['x'], self.particleArray['y'])))
        dist = cdist(r, r, 'euclidean')

        # Distance of closest particle time 1.3
        h: np.array = 1.3 * np.ma.masked_values(dist, 0.0, copy=False).min(1)

        self.timing_data['neighbour_hood'] += perf_counter() - start
        return (r, dist, h)

    @staticmethod
    @njit(fastmath=True, parallel=True)
    def _loop(h, dist, pA, evFunc, gradFunc, methodClass):
        p = methodClass.compute_pressure(pA)
        c = methodClass.compute_speed_of_sound(pA)

        for i in prange(len(pA)):
            # Assign parameters
            pA[i]['p'] = p[i]
            pA[i]['c'] = c[i]

            # Find near neighbours and their h and q
            h_i, q_i, near_arr = _nearNbrs(i, h, dist[i, :])

            # Skip if got no neighbours, early exit.
            # Keep same properties, no acceleration.
            if len(near_arr) == 0:
                continue

            # Create computed properties
            # Fill the props
            calcProps = _assignProps(i, pA, near_arr, h_i, q_i, dist[i, :])

            # Kernel values
            w = evFunc(calcProps['r'], calcProps['h'])
            dw_x = gradFunc(calcProps['x'], calcProps['r'], calcProps['h'])
            dw_y = gradFunc(calcProps['y'], calcProps['r'], calcProps['h'])

            for j in prange(len(calcProps)):
                calcProps[j]['w'] = w[j] # Not needed for density change
                calcProps[j]['dw_x'] = dw_x[j]
                calcProps[j]['dw_y'] = dw_y[j]

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
        (r, dist, h) = self._nbrs()

        # Set h
        self.particleArray['h'] = h

        # Re-set accelerations
        self.particleArray['ax'] = 0.0
        self.particleArray['ay'] = 0.0
        self.particleArray['drho'] = 0.0

        # Loop
        self.particleArray = Solver._loop(h, dist, self.particleArray, self.kernel.evaluate, self.kernel.gradient, self.method)

        self.timing_data['compute'] += perf_counter() - start

    def solve(self):
        """
            Solve the equations setup
        """
        # Check particle length.
        if len(self.particles) == 0 or len(self.particles) != self.num_particles:
            raise Exception('No or invalid particles set!')

        # Keep integrating until simulation duration is reached.

        start_all = perf_counter()

        # TODO: Change giant-matrix size if wrong due to different time-steps.
        t_step: int = 0   # Step
        t: float    = 0.0 # Current time
        print('Started solving...')
        settleSteps = 100
        sbar = tqdm(total=settleSteps, desc='Settling', unit='steps', leave=False)
        with tqdm(total=self.duration, desc='Time-stepping', unit='s', leave=False) as tbar:
            while t < self.duration:
                # Compute time step.
                self.dt = self._minTimeStep()

                # Check explosion
                if self.dt > 0.1:
                    print(f'{Fore.YELLOW}Warning: {Style.RESET_ALL} suspected explosion. Check time-step.')

                if self.integrator.isMultiStage() == True:
                    # Start with eval
                    self._compute()
                
                # Predict
                start = perf_counter()
                self.particleArray = self.integrator.predict(self.dt, self.particleArray, self.damping)
                self.timing_data['integrate_prediction'] += perf_counter() - start

                # Compute the accelerations
                self._compute()

                # Correct
                start = perf_counter()
                self.particleArray = self.integrator.correct(self.dt, self.particleArray, self.damping)
                self.timing_data['integrate_correction'] += perf_counter() - start

                # Store to data
                start = perf_counter()
                self.data.append(self.particleArray)
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
                        print(f'{Fore.GREEN}Settling Complete.{Style.RESET_ALL}')
                        
                    # Stop damping after 100-th time steps.
                    self.damping = 0.0

                    # Remove temp-boundary
                    inds = []
                    for i in range(self.num_particles):
                        p = self.particleArray[i]
                        if p['label'] == ParticleType.TempBoundary:
                            inds.append(i)
                    self.particleArray = np.delete(self.particleArray, np.array(inds), axis=0)
                    self.num_particles = len(self.particleArray)

                # Update and export plot
                start = perf_counter()
                if self.plot == True:
                    self.update_frame(t_step, t)
                    self.export(t_step)
                self.timing_data['export'] += perf_counter() - start

                # Update tbar
                if self.damping == 0:
                    tbar.update(self.dt)
                else:
                    sbar.update(n=1)
                    tbar.update(0)
            # End while
        # End-with
        self.timing_data['total'] = perf_counter() - start_all

        total = self.timing_data['total']
        self.t_step = t_step
        print(f'{Fore.GREEN}Solved!{Style.RESET_ALL}')
        print(f'Solved {Fore.YELLOW}{self.num_particles}{Style.RESET_ALL} particles for {Fore.YELLOW}{self.duration:f}{Style.RESET_ALL} [s].')
        print(f'Completed solve in {Fore.YELLOW}{total:f}{Style.RESET_ALL} [s] and {Fore.YELLOW}{t_step}{Style.RESET_ALL} steps')

    def timing(self):
        t = PrettyTable(['Name', 'Time [s]', 'Percentage [%]'])
        for k, v in self.timing_data.items():
            t.add_row([k, round(v, 3), round(v / self.timing_data['total'] * 100, 2)])

        print('Detailed timing statistics:')
        print(t)

    def save(self):
        """
            Saves the output to a compressed export .npz file.
        """
        
        # Export movie if relevant.
        if self.plot == True:
            self.export_mp4()

        # Export compressed numpy-arrays
        np.savez_compressed(f'{sys.path[0]}/export', data=np.array(self.data), dts=np.array(self.dts))
        print(f'Exported arrays to: "{sys.path[0]}/export.npz".')

        # Export dts
        np.savez_compressed(f'{sys.path[0]}/dts', dts=np.array(self.dts))
        print(f'Exported dts to: "{sys.path[0]}/export.npz".')

    # Plotting functions
    def export(self, frame: int):
        # Export
        self.exporter = pg.exporters.ImageExporter(self.pw.plotItem)
        self.exporter.params.param('width').setValue(700, blockSignal=self.exporter.widthChanged)
        self.exporter.params.param('height').setValue(500, blockSignal=self.exporter.heightChanged)

        frame_str = "{:06}".format(frame)
        self.exporter.export(f'{self.tempdir.name}/export_{frame_str}.png')

    def update_frame(self, frame: int, time: float) -> None:
        self.pl.setData(x=self.particleArray['x'], y=self.particleArray['y'], symbolBrush=self.cm.map(self.particleArray['p'] / 1000, 'qcolor'), symbolSize=self.sZ)
        self.txtItem.setText(f't = {time:f} [s]')

    def init_plot(self):
        """ Initialize the plot. """
        start: float = perf_counter()

        # use less ink
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        pg.setConfigOptions(antialias=True)

        # Plot
        # TODO: Change the titles.
        # TODO: Set range automatically, or from parameter; at least not hard-coded.
        self.pw = pg.plot(title='Dam break (2D)', labels={'left': 'Y [m]', 'bottom': 'X [m]', 'top': 'Dam Break (2D)'})
        self.pw.setXRange(-3, 81)
        self.pw.setYRange(-3, 41)

        # Text
        self.txtItem = pg.TextItem(text = f't = 0.00000 [s]')
        self.pw.scene().addItem(self.txtItem)
        [[xmin, xmax], [_, _]] = self.txtItem.getViewBox().viewRange()
        xrange = xmax - xmin
        self.txtItem.translate(350.0 - xrange, 90.0)

        # make colormap
        c_range = np.linspace(0, 1, num=10)
        colors = []
        for cc in c_range:
            colors.append([cc, 0.0, 1 - cc, 1.0]) # Blue to red color spectrum
        stops = np.round(c_range * (np.max(self.particleArray['p'])) / 1000, 0)
        self.cm = pg.ColorMap(stops, np.array(colors))
        
        # make colorbar, placing by hand
        cb = ColorBar(cmap=self.cm, width=10, height=200, label='Pressure [kPa]')
        self.pw.scene().addItem(cb)
        cb.translate(610.0, 90.0)

        # Initial points
        self.sZ = (40 * (2 / (self.num_particles ** 0.5)))
        self.pl = self.pw.plot(self.particleArray['x'], self.particleArray['y'], pen=None, symbol='o', symbolBrush=self.cm.map(self.particleArray['p'] / 1000, 'qcolor'), symbolPen=None, symbolSize=self.sZ)

        # Frame 0 export.
        self.tempdir = tempfile.TemporaryDirectory()
        print(f'Exporting frames to directory: {self.tempdir.name}')
        self.export(0)

        # Timing
        self.timing_data['export'] += perf_counter() - start

    def export_mp4(self):
        # Export to mp4
        fps = np.round(self.t_step / self.duration / 8) # Frames per second
        subprocess.run(f'ffmpeg -hide_banner -loglevel panic -y -framerate {fps} -i "{self.tempdir.name}\\export_%06d.png" -s:v 700x500 -c:v libx264 \
-profile:v high -crf 20 -pix_fmt yuv420p {sys.path[0]}\\export.mp4"')
        print('Export complete!')
        print(f'Exported to "{sys.path[0]}\\export.mp4".')

        # Cleanup export dir
        self.tempdir.cleanup()

# Moved to outside of class for numba
@njit(fastmath=True)
def _assignProps(i: int, particleArray: np.array, near_arr: np.array, h_i: np.array, q_i: np.array, dist: np.array):
    J = len(near_arr)

    # Create empty array
    calcProps = np.zeros(J, dtype=computed_dtype)

    # Fill based on existing data.
    for j in prange(J):
        global_i = near_arr[j]
        pA = particleArray[global_i]

        # From self properties
        calcProps[j]['p']   = pA['p']
        calcProps[j]['m']   = pA['m']
        #calcProps[near_i]['c']   = self.method.compute_speed_of_sound(pA)
        calcProps[j]['rho'] = pA['rho']

        # Pre-calculated properties
        calcProps[j]['h'] = h_i[global_i] # average h, precalculated
        calcProps[j]['q'] = q_i[global_i] # dist / h, precalculated
        calcProps[j]['r'] = dist[global_i] # distance, precalculated

        # Positional values
        calcProps[j]['x']  = particleArray[i]['x'] - pA['x']
        calcProps[j]['y']  = particleArray[i]['y'] - pA['y']
        calcProps[j]['vx'] = particleArray[i]['vx'] - pA['vx']
        calcProps[j]['vy'] = particleArray[i]['vy'] - pA['vy']
    # END_LOOP
    return calcProps

@njit(fastmath=True)
def _nearNbrs(i: int, h: np.array, dist: np.array):
    # Create empty complete matrices
    q_i = np.zeros_like(h)
    h_i = np.zeros_like(h)
    near = [] # indices of near particles

    # Check each particle.
    J = len(h)
    for j in prange(J):
        h_i[j] = 0.5 * (h[i] + h[j]) # averaged h.
        q_i[j] = dist[j] / h_i[j] # q (norm-dist)

        if q_i[j] <= 3.0:
            near.append(j)
    return (h_i, q_i, np.array(near))