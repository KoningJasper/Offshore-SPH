# Import python packages
import sys
import tempfile
import subprocess
import math

# Other packages
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from numba import prange, autojit
from typing import List

# Own components
from src.Common import particle_dtype, computed_dtype, types
from src.Particle import Particle
from src.Methods.Method import Method
from src.Kernels.Kernel import Kernel
from src.Integrators.Integrator import Integrator
from src.ColorBar import ColorBar

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

    def __init__(self, method: Method, integrator: Integrator, kernel: Kernel, duration: float, dt: float, plot: bool):
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
        
        # TODO: Remove
        self.dt = dt

    def addParticles(self, particles: List[Particle]):
        self.particles.extend(particles)

    def _convertParticles(self):
        """ Convert the particle classes to a numpy array. """
        self.particleArray = np.array([(types[p.label], p.m, p.rho, p.p, 0, 0, p.x, p.y, 0, 0, 0, 0) for p in self.particles])
        self.num_particles = len(self.particleArray)

    def setup(self):
        """ Sets-up the solver, with the required parameters. """
        # Convert the particle array.
        self._convertParticles()

        # Initial guess for the timestep.
        time_step_guess: int = math.ceil(self.duration / self.dt)

        # Empty time arrays
        self.x = np.zeros([self.num_particles, time_step_guess]) # X-pos
        self.y = np.zeros([self.num_particles, time_step_guess]) # Y-pos
        self.c = np.zeros([self.num_particles, time_step_guess]) # Pressure (p)
        self.u = np.zeros([self.num_particles, time_step_guess]) # Density (rho)
        self.drho = np.zeros([self.num_particles, time_step_guess]) # Density (rho)
        self.v = np.zeros([self.num_particles, 2, time_step_guess]) # Velocity both x and y
        self.a = np.zeros([self.num_particles, 2, time_step_guess]) # Acceleration both x and y

        # Initialization
        self.particleArray
        for i in range(self.num_particles):
            self.particles[i]   = self.method.initialize(self.particles[i])
            self.particles[i].p = self.method.compute_pressure(self.particles[i])

        # Set 0-th time-step
        self.x[:, 0] = [p.r[0] for p in self.particles]
        self.y[:, 0] = [p.r[1] for p in self.particles]
        self.c[:, 0] = [p.p for p in self.particles]
        self.u[:, 0] = [p.rho for p in self.particles]

        if self.plot == True:
            self.init_plot()
            self.export(0)

        print('Setup complete.')

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
                # TODO: Move this to separate class and function.
                # Distance and neighbourhood
                r = np.array([p.r for p in self.particles])
                dist = cdist(r, r, 'euclidean')

                # Distance of closest particle time 1.3
                # TODO: Move to separate class.
                h: np.array = 1.3 * np.ma.masked_values(dist, 0.0, copy=False).min(1)

                # Pbar
                with tqdm(total=self.num_particles, desc='Acceleration eval', unit='particle', leave=False) as pbar:
                    # Acceleration and force loop
                    for i in range(self.num_particles):
                        # For convenience, copy to p variable.
                        p = self.particles[i]

                        # Run EOS
                        p.p = self.method.compute_pressure(p)

                        # Query neighbours
                        # TODO: Move to separate class.
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
                            p.a = self.method.compute_acceleration(i, p, xij, rij, vij, self.c[near_arr, t_step], self.u[near_arr, t_step], hij, cij, wij, dwij)
                            p.v = self.method.compute_velocity(i, p)
                        # end fluid

                        # Assign back
                        self.particles[i] = p

                        # Increment with one particle
                        pbar.update(1)
                    # end acc. loop
                # end with pbar

                # Integration loop
                for i in range(self.num_particles):
                    # Integrate
                    self.particles[i] = self.integrator.integrate(self.dt, self.particles[i])

                    # Put into giant-matrix
                    p = self.particles[i] # Easier
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

                # Update and export plot
                if self.plot == True:
                    self.update_frame(t_step)
                    self.export(t_step)

                # Update tbar
                tbar.update(self.dt)
            # End while
        # End-with

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
        self.exporter = pyqtgraph.exporters.ImageExporter(self.pw.plotItem)
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