# Add parent folder to path
import sys, os, tempfile, subprocess
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import scipy.spatial
from tqdm import tqdm

# Plotting
from ColorBar import ColorBar
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import pyqtgraph.exporters

# Own classes
from src.Kernels.Gaussian import Gaussian
from src.Particle import Particle
from src.Equations.WCSPH import WCSPH
from src.Integrators.EulerIntegrater import EulerIntegrater

from Momentum import MomentumEquation

def create_particles(wcsph, N: int, mass: float):
    # Create some particles
    xv = np.linspace(0, 2, N)
    yv = np.linspace(0, 2, N)
    x, y = np.meshgrid(xv, yv, indexing='ij')
    x = x.ravel()
    y = y.ravel()

    particles = []
    for i in range(len(x)):
        particles.append(Particle('fluid', x[i], y[i], mass))
    
    r0 = 1 / N # Distance between boundary particles.
    rho_b = 1000. # Density of the boundary [kg/m^3]
    mass_b = mass * 1.5 # Mass of the boundary [kg]

    # Maximum and minimum values of boundaries
    x_min = -1.0
    x_max = 10.1
    y_min = -1.0
    y_max = 5
    
    # Bottom & Top
    xv = np.arange(x_min, x_max, r0)
    yv = np.zeros(len(xv)) + y_min
    #yv2 = np.zeros(len(xv)) + 10
    for i in range(len(xv)):
        particles.append(Particle('boundary', xv[i], yv[i], mass_b, rho_b))
        particles.append(Particle('boundary', xv[i] - r0 / 2, yv[i] - r0, mass_b, rho_b))
        particles.append(Particle('boundary', xv[i], yv[i] - 2 * r0, mass_b, rho_b))
        particles.append(Particle('boundary', xv[i] - r0 / 2, yv[i] - 3 * r0, mass_b, rho_b))

    # Left & Right
    yv3 = np.arange(y_min, y_max, r0)    
    xv2 = np.zeros(len(yv3)) + x_min
    xv3 = np.zeros(len(yv3)) + x_max
    for i in range(len(yv3)):
        particles.append(Particle('boundary', xv2[i], yv3[i], mass_b, rho_b))
        particles.append(Particle('boundary', xv2[i] - r0, yv3[i] - r0 / 2, mass_b, rho_b))
        particles.append(Particle('boundary', xv3[i], yv3[i], mass_b, rho_b))
        particles.append(Particle('boundary', xv3[i] + r0, yv3[i] - r0 / 2, mass_b, rho_b))
        
    return particles

def export(frame: int):
    # Export
    exporter = pyqtgraph.exporters.ImageExporter(pw.plotItem)
    exporter.params.param('width').setValue(700, blockSignal=exporter.widthChanged)
    exporter.params.param('height').setValue(500, blockSignal=exporter.heightChanged)

    frame_str = "{:06}".format(frame)
    exporter.export(f'{tempdir.name}/export_{frame_str}.png')

def update_frame(frame: int) -> None:
    pl.setData(x=x[:, frame], y=y[:, frame], symbolBrush=cm.map(u[:, frame], 'qcolor'), symbolSize=sZ)

if __name__ == '__main__':

    # WCSPH method
    wcsph = WCSPH(height=2.0)

    # Generate some particles
    #N = int(input('Number of particles (has to be a rootable number)?\n'))
    N = 10
    mass = 2 * 2 * wcsph.rho0 / (N ** 2) # Per particle
    particles = create_particles(wcsph, N, mass)
    mm = np.ones(len(particles)) * mass
    fluid_particles = [p for p in particles if p.label == 'fluid']

    # Get fluid indices
    fluid_ind = []
    for i in range(len(particles)):
        if particles[i].label == 'fluid':
            fluid_ind.append(i)
    fluid_ind = np.array(fluid_ind)

    # Solving properties
    kernel = Gaussian()
    integrater = EulerIntegrater()
    t_max = 1.5
    dt = 0.01
    t = np.arange(0, t_max, dt)
    t_n = len(t)
    H = 2 # water column height
    hdx = 1.3 * np.sqrt(2 * (2 * 2 / (N ** 2)))
    h = np.ones(len(particles)) * hdx

    gx = 0.
    gy = -9.81

    # Initialize the loop
    for p in particles:
        wcsph.inital_condition(p)

    # Time-stepping
    x = np.zeros([len(particles), t_n]) # X-pos
    y = np.zeros([len(particles), t_n]) # Y-pos
    c = np.zeros([len(particles), t_n]) # Pressure (p)
    u = np.zeros([len(particles), t_n]) # Density (rho)
    v = np.zeros([len(particles), 2, t_n]) # Velocity both x and y

    # Initialization
    x[:, 0] = [p.r[0] for p in particles]
    y[:, 0] = [p.r[1] for p in particles]
    c[:, 0] = [p.p for p in particles]
    u[:, 0] = [p.rho for p in particles]

    # Initial plot
    # use less ink
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')

    pw = pg.plot(title='Dam break (2D)', labels={'left': 'Y [m]', 'bottom': 'X [m]'})
    pw.setXRange(-1, 11)
    pw.setYRange(-1, 5)

    # make colormap
    # TODO: Auto generate these steps
    stops = np.r_[0.25, 0.5, 0.75, 1.0] * (max(u[fluid_ind, 0]) - wcsph.rho0) + wcsph.rho0
    colors = np.array([[0, 0, 1, 0.7], [0, 1, 0, 0.2], [0, 0, 0, 0.8], [1, 0, 0, 1.0]])
    cm = pg.ColorMap(stops, colors)
    
    # make colorbar, placing by hand
    cb = ColorBar(cm, 10, 200, label='Density [kg/m^3]')#, [0., 0.5, 1.0])
    pw.scene().addItem(cb)
    cb.translate(610.0, 90.0)

    # Initial points
    sZ = (20*(2/N))
    pl = pw.plot(x[:, 0], y[:, 0], pen=None, symbol='o', symbolBrush=cm.map(u[:, 0], 'qcolor'), symbolSize=sZ)

    # Frame 0 export.
    tempdir = tempfile.TemporaryDirectory()
    print(f'Exporting frames to directory: {tempdir.name}')
    export(0)

    i: int = 0
    for t_step in tqdm(range(t_n - 1), desc='Time-stepping'):
        # Distance and neighbourhood
        r = np.array([p.r for p in particles])
        dist = scipy.spatial.distance.cdist(r, r, 'euclidean')
        hood = scipy.spatial.cKDTree(r)
        
        # Force/Acceleration evaluation loop
        i: int = 0
        #for p in tqdm(particles, desc='Evaluating equations', leave=False):
        for p in particles:
            if p.label == 'boundary':
                i += 1
                continue
            
            # Initialize particle, reset acceleration and density.
            wcsph.loop_initialize(p)
            
            # Query neighbours
            r_dist: float  = 3 * h[0] # Goes to zero when q > 3
            near_ind: list = hood.query_ball_point(p.r, r_dist)
            near_ind.remove(i) # Delete self
            near_arr: np.array = np.array(np.sort(near_ind))
            
            # Skip if got no neighbours
            # Keep same properties, no acceleration.
            if len(near_arr) == 0:
                i += 1
                continue
            
            # Calculate some re-usable properties
            xij: np.array = r[near_arr] - p.r
            rij: np.array = dist[i, near_arr]
            vij: np.array = v[near_arr, :, t_step] - p.v
            dwij: np.array = kernel.gradient(xij, rij, h[near_arr])
            
            # Evaluate the equations
            wcsph.TaitEOS(p)
            wcsph.Continuity(mass, p, dwij, vij, len(near_arr))

            # Should be no need to re-set it, but whatever.
            # Fucking numpy.
            p.a = wcsph.Momentum(mass, p, c[near_arr, t_step], u[near_arr, t_step], dwij)

            # Add gravity
            wcsph.Gravity(p, gx, gy)
            
            # Next Particle
            i += 1
        
        # Integration loop
        i: int = 0
        for p in particles:
            # Integrate the thingies
            integrater.integrate(dt, p)
            
            # Put into giant-matrix
            x[i, t_step + 1] = p.r[0]
            y[i, t_step + 1] = p.r[1]
            c[i, t_step + 1] = p.p
            u[i, t_step + 1] = p.rho
            v[i, :, t_step + 1] = p.v
            
            i += 1

        # Update and export plot
        update_frame(t_step + 1)
        export(t_step + 1)

    # Export to mp4
    fps = 4 # Frames per second
    subprocess.run(f'ffmpeg -hide_banner -loglevel panic -y -r {fps} -i "{tempdir.name}/export_%06d.png" -c:v libx264 -vf fps=10 -pix_fmt yuv420p "{sys.path[0]}/export.mp4"')
    print('Export complete!')
    print(f'Exported to "{sys.path[0]}/export.mp4"')

    # Cleanup export dir
    tempdir.cleanup()