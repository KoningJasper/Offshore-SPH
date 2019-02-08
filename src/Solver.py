# Import python packages
import sys
import tempfile
import subprocess
import math

# Other packages
import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm
from scipy.spatial.distance import cdist
from numba import prange, jit, njit
from typing import List
from time import perf_counter

# Own components
from src.Common import particle_dtype, computed_dtype, types, get_label_code
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
    nbrHTime: float = 0.0
    nbrFTime: float = 0.0 
    momTime: float = 0.0 # Momentum
    conTime: float = 0.0 # Continuity
    preTime: float = 0.0 # Prediction
    corTime: float = 0.0 # Correction
    propTime: float = 0.0 # calculation of properties.
    propATime: float = 0.0
    propKTime: float = 0.0
    storTime: float = 0.0 # storing properties

    def __init__(self, method: Method, integrator: Integrator, kernel: Kernel, duration: float, plot: bool):
        """
            Parameters
            ----------

            method: The method to use for the solving of SPH particles (WCSPH/ICSPH).

            integrator: The integration method to be used by the solver (Euler/Verlet).

            kernel: The kernel function to use in the evaluation (Gaussian/CubicSpline).

            duration: The duration of the simulation in seconds.

            dt: timestep [s]

            plot: Should a plot be created?
        """

        self.method     = method
        self.integrator = integrator
        self.kernel     = kernel
        self.duration   = duration
        self.plot       = plot

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

        # Initial guess for the timestep.
        self.dt = 1e-5;
        time_step_guess: int = math.ceil(self.duration / self.dt) + 1

        # Empty time arrays
        self.x = np.zeros([self.num_particles, time_step_guess]) # X-pos
        self.y = np.zeros([self.num_particles, time_step_guess]) # Y-pos
        self.c = np.zeros([self.num_particles, time_step_guess]) # Pressure (p)
        self.u = np.zeros([self.num_particles, time_step_guess]) # Density (rho)
        self.drho = np.zeros([self.num_particles, time_step_guess]) # Density (rho)
        self.v = np.zeros([self.num_particles, 2, time_step_guess]) # Velocity both x and y
        self.a = np.zeros([self.num_particles, 2, time_step_guess]) # Acceleration both x and y

        # Initialization
        # Calc h
        r = np.transpose(np.vstack((self.particleArray['x'], self.particleArray['y'])))
        dist = cdist(r, r, 'euclidean')
        h: np.array = 1.3 * np.ma.masked_values(dist, 0.0, copy=False).min(1)
        self.particleArray['h'] = h

        for i in range(self.num_particles):
            self.particleArray[i] = self.method.initialize(self.particleArray[i])
            self.particleArray[i]['p'] = self.method.compute_pressure(self.particleArray[i])
            self.particleArray[i]['c'] = self.method.compute_speed_of_sound(self.particleArray[i])

        # Set 0-th time-step
        self.x[:, 0] = [p.r[0] for p in self.particles]
        self.y[:, 0] = [p.r[1] for p in self.particles]
        self.c[:, 0] = [p.p for p in self.particles]
        self.u[:, 0] = [p.rho for p in self.particles]

        if self.plot == True:
            self.init_plot()
            self.export(0)

        print('Setup complete.')

    def _calcProps(self, i: int, near_arr: np.array, h_i: np.array, q_i: np.array, dist: np.array) -> np.array:
        start = perf_counter()

        # Fill the props
        startA = perf_counter()
        calcProps = _assignProps(i, self.particleArray, near_arr, h_i, q_i, dist)
        self.propATime += perf_counter() - startA

        # Speed of sound
        for p in calcProps:
            p['c'] = self.method.compute_speed_of_sound(p)

        # Kernel values
        startK = perf_counter()
        calcProps['w'] = self.kernel.evaluate(calcProps['r'], calcProps['h'])
        calcProps['dw_x'] = self.kernel.gradient(calcProps['x'], calcProps['r'], calcProps['h'])
        calcProps['dw_y'] = self.kernel.gradient(calcProps['y'], calcProps['r'], calcProps['h'])
        self.propKTime += perf_counter() - startK

        self.propTime += perf_counter() - start
        return calcProps

    def _minTimeStep(self) -> float:
        h_min = 10e10
        c_max = -1e10
        for p in self.particleArray:
            # Only for fluids, the other ones don't move
            if p['label'] != get_label_code('fluid'):
                continue

            if p['h'] < h_min:
                h_min = p['h']
            if p['c'] > c_max:
                c_max = p['c']

        dt = Courant().calc(h_min, c_max)
        return dt

    def _nbrs(self):
        start = perf_counter()

        # Distance and neighbourhood
        r = np.transpose(np.vstack((self.particleArray['x'], self.particleArray['y'])))
        dist = cdist(r, r, 'euclidean')

        # Distance of closest particle time 1.3
        h: np.array = 1.3 * np.ma.masked_values(dist, 0.0, copy=False).min(1)

        self.nbrHTime += perf_counter() - start
        return (r, dist, h)

    @jit
    def _findNbrs(self, i: int, h: np.array, dist: np.array):
        start = perf_counter()

        # Query neighbours
        h_i, q_i, near_arr = _nearNbrs(i, h, dist[i, :])

        self.nbrFTime += perf_counter() - start
        return (h_i, q_i, near_arr)

    def _compute(self):
        """ Compute the accelerations, velocities, etc. """
        # Neighbourhood
        (r, dist, h) = self._nbrs()

        # Re-set accelerations
        self.particleArray['ax'] = 0.0
        self.particleArray['ay'] = 0.0
        self.particleArray['drho'] = 0.0

        # Pbar
        with tqdm(total=self.num_particles, desc='Acceleration eval', unit='particle', leave=False) as pbar:
            # Acceleration and force loop
            for i in range(self.num_particles):
                # Find near neighbours and their h and q
                (h_i, q_i, near_arr) = self._findNbrs(i, h, dist)

                # Skip if got no neighbours, early exit.
                # Keep same properties, no acceleration.
                if len(near_arr) == 0:
                    continue

                # Create computed properties
                calcProps = self._calcProps(i, near_arr, h_i, q_i, dist[i, :])

                # Assign particle
                p = self.particleArray[i]

                # Calc speed of sound.
                p['c'] = self.method.compute_speed_of_sound(p)

                # Compute pressure, EOS
                p['p'] = self.method.compute_pressure(p)
                
                # Continuity
                start = perf_counter()
                p['drho'] = self.method.compute_density_change(p, calcProps)
                self.conTime += perf_counter() - start

                # Momentum
                if p['label'] == get_label_code('fluid'):
                    start = perf_counter()

                    [p['ax'], p['ay']] = self.method.compute_acceleration(p, calcProps)
                    [p['vx'], p['vy']] = self.method.compute_velocity(p, calcProps)

                    self.momTime += perf_counter() - start
                # end fluid

                # Increment with one particle
                pbar.update(1)
            # end acc. loop
        # end with pbar

    def solve(self):
        """
            Solve the equations setup
        """
        # Check particle length.
        if len(self.particles) == 0 or len(self.particles) != self.num_particles:
            raise Exception('No or invalid particles set!')

        # Keep integrating until simulation duration is reached.
        # TODO: Change giant-matrix size if wrong due to different time-steps.
        t_step: int = 0   # Step
        t: float    = 0.0 # Current time
        print('Started solving...')
        with tqdm(total=self.duration, desc='Time-stepping', unit='s') as tbar:
            while t < self.duration:
                # Compute time step.
                self.dt = self._minTimeStep()

                if self.integrator.isMultiStage() == True:
                    # Start with eval
                    self._compute()
                
                # Predict
                start = perf_counter()
                for p in self.particleArray:
                    p = self.integrator.predict(self.dt, p)
                self.preTime += perf_counter() - start

                # Compute the accelerations
                self._compute()

                # Correct
                start = perf_counter()
                for p in self.particleArray:
                    p = self.integrator.correct(self.dt, p)
                self.corTime += perf_counter() - start

                # Store-loop
                start = perf_counter()
                for i in range(self.num_particles):
                    # Put into giant-matrix
                    p = self.particleArray[i] # Easier
                    self.x[i, t_step + 1] = p['x']
                    self.y[i, t_step + 1] = p['y']
                    self.c[i, t_step + 1] = p['p']
                    self.u[i, t_step + 1] = p['rho']
                    self.v[i, :, t_step + 1] = np.array([p['vx'], p['vy']])
                    self.a[i, :, t_step + 1] = np.array([p['ax'], p['ay']])
                    self.drho[i, t_step + 1] = p['drho']
                self.storTime += perf_counter() - start

                # End integration-loop
                t += self.dt
                t_step += 1

                # Update and export plot
                if self.plot == True:
                    self.update_frame(t_step)
                    self.export(t_step)

                # Update tbar
                tbar.update(self.dt)
            # End while
        # End-with

    def timing(self):
        total = sum([self.conTime, self.momTime, self.propTime, self.nbrFTime, self.nbrHTime, self.storTime, self.preTime, self.corTime])

        nbrTime = self.nbrFTime + self.nbrHTime

        t = PrettyTable(['Name', 'Time [s]', 'Percentage [%]'])
        t.add_row(['Continuity', round(self.conTime, 3), round(self.conTime / total * 100, 2)])
        t.add_row(['Momentum', round(self.momTime, 3), round(self.momTime / total * 100, 2)])
        t.add_row(['Properties - Assigning', round(self.propATime, 3), round(self.propATime / total * 100, 2)])
        t.add_row(['Properties - Kernel', round(self.propKTime, 3), round(self.propATime / total * 100, 2)])
        t.add_row(['Properties - Total', round(self.propTime, 3), round(self.propTime / total * 100, 2)])
        t.add_row(['Neighbours - Hood', round(self.nbrHTime, 3), round(self.nbrHTime / total * 100, 2)])
        t.add_row(['Neighbours - Find', round(self.nbrFTime, 3), round(self.nbrFTime / total * 100, 2)])
        t.add_row(['Neighbours - Total', round(nbrTime, 3), round(nbrTime / total * 100, 2)])
        t.add_row(['Storage', round(self.storTime, 3), round(self.storTime / total * 100, 2)])
        t.add_row(['Predictor', round(self.preTime, 3), round(self.preTime / total * 100, 2)])
        t.add_row(['Corrector', round(self.corTime, 3), round(self.corTime / total * 100, 2)])

        print(t)

    def save(self):
        """
            Saves the output to a compressed export .npz file.
        """
        
        # Export movie if relevant.
        if self.plot == True:
            self.export_mp4()

        # Export compressed numpy-arrays
        np.savez_compressed(f'{sys.path[0]}/export', x=self.x, y=self.y, p=self.c, rho=self.u, v=self.v, a=self.a, drho=self.drho)
        print(f'Exported arrays to: "{sys.path[0]}/export.npz".')

    # Plotting functions
    def export(self, frame: int):
        # Export
        self.exporter = pg.exporters.ImageExporter(self.pw.plotItem)
        self.exporter.params.param('width').setValue(700, blockSignal=self.exporter.widthChanged)
        self.exporter.params.param('height').setValue(500, blockSignal=self.exporter.heightChanged)

        frame_str = "{:06}".format(frame)
        self.exporter.export(f'{self.tempdir.name}/export_{frame_str}.png')

    def update_frame(self, frame: int) -> None:
        self.pl.setData(x=self.x[:, frame], y=self.y[:, frame], symbolBrush=self.cm.map(self.c[:, frame], 'qcolor'), symbolSize=self.sZ)
        time: int = frame * self.dt
        self.txtItem.setText(f't = {time:f} [s]')

    def init_plot(self):
        """ Initialize the plot. """
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
        stops = np.round(c_range * (np.max(self.c[:, 0])) / 1000, 0)
        self.cm = pg.ColorMap(stops, np.array(colors))
        
        # make colorbar, placing by hand
        cb = ColorBar(self.cm, 10, 200, label='Pressure [kPa]')#, [0., 0.5, 1.0])
        self.pw.scene().addItem(cb)
        cb.translate(610.0, 90.0)

        # Initial points
        self.sZ = (40 * (2 / (self.num_particles ** 0.5)))
        self.pl = self.pw.plot(self.x[:, 0], self.y[:, 0], pen=None, symbol='o', symbolBrush=self.cm.map(self.c[:, 0] / 1000, 'qcolor'), symbolPen=None, symbolSize=self.sZ)

        # Frame 0 export.
        self.tempdir = tempfile.TemporaryDirectory()
        print(f'Exporting frames to directory: {self.tempdir.name}')
        self.export(0)

    def export_mp4(self):
        # Export to mp4
        fps = np.round(0.5 / self.dt) # Frames per second
        subprocess.run(f'ffmpeg -hide_banner -loglevel panic -y -r {fps} -i "{self.tempdir.name}/export_%06d.png" -c:v libx264 -vf fps=10 -pix_fmt yuv420p "{sys.path[0]}/export.mp4"')
        print('Export complete!')
        print(f'Exported to "{sys.path[0]}/export.mp4".')

        # Cleanup export dir
        self.tempdir.cleanup()

# Moved to outside of class for numba
@njit
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
        calcProps[j]['x'] = particleArray[i]['x'] - pA['x']
        calcProps[j]['y'] = particleArray[i]['y'] - pA['y']
        calcProps[j]['vx'] = particleArray[i]['vx'] - pA['vx']
        calcProps[j]['vy'] = particleArray[i]['vy'] - pA['vy']
    # END_LOOP
    return calcProps

@njit
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