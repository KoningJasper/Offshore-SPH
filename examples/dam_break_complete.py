# Add parent folder to path
import sys, os, tempfile, subprocess
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import numpy as np
import scipy.spatial
from tqdm import tqdm

# Plotting
import pyqtgraph as pg
import pyqtgraph.exporters
from ColorBar import ColorBar
from pyqtgraph.Qt import QtCore
from pyqtgraph.graphicsItems import TextItem
from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

# Own classes
from src.Kernels.Gaussian import Gaussian
from src.Particle import Particle
from src.Equations.WCSPH import WCSPH
from src.Integrators.EulerIntegrater import EulerIntegrater

def create_particles(wcsph, N: int, mass: float):
    # Create some particles
    xv = np.linspace(0, 25, N)
    yv = np.linspace(0, 25, N)
    x, y = np.meshgrid(xv, yv, indexing='ij')
    x = x.ravel()
    y = y.ravel()

    particles = []
    for i in range(len(x)):
        particles.append(Particle('fluid', x[i], y[i], mass))
    
    r0 = 25 / N / 2 # Distance between boundary particles.
    rho_b = 1000. # Density of the boundary [kg/m^3]
    mass_b = mass * 1.5 # Mass of the boundary [kg]

    # Maximum and minimum values of boundaries
    x_min = -3
    x_max = 80
    y_min = -3
    y_max = 40
    
    # Bottom & Top
    xv = np.arange(x_min, x_max, r0)
    yv = np.zeros(len(xv)) + y_min
    #yv2 = np.zeros(len(xv)) + 10
    for i in range(len(xv)):
        particles.append(Particle('boundary', xv[i], yv[i], mass_b, rho_b))
        particles.append(Particle('boundary', xv[i] - r0 / 2, yv[i] - r0, mass_b, rho_b))
        # particles.append(Particle('boundary', xv[i], yv[i] - 2 * r0, mass_b, rho_b))
        # particles.append(Particle('boundary', xv[i] - r0 / 2, yv[i] - 3 * r0, mass_b, rho_b))

    # Left & Right
    yv3 = np.arange(y_min, y_max, r0)    
    xv2 = np.zeros(len(yv3)) + x_min
    xv3 = np.zeros(len(yv3)) + x_max
    for i in range(len(yv3)):
        particles.append(Particle('boundary', xv2[i], yv3[i], mass_b, rho_b))
        particles.append(Particle('boundary', xv2[i] - r0, yv3[i] - r0 / 2, mass_b, rho_b))
        # particles.append(Particle('boundary', xv3[i], yv3[i], mass_b, rho_b))
        # particles.append(Particle('boundary', xv3[i] + r0, yv3[i] - r0 / 2, mass_b, rho_b))
        
    return particles

def export(frame: int):
    # Export
    exporter = pyqtgraph.exporters.ImageExporter(pw.plotItem)
    exporter.params.param('width').setValue(700, blockSignal=exporter.widthChanged)
    exporter.params.param('height').setValue(500, blockSignal=exporter.heightChanged)

    frame_str = "{:06}".format(frame)
    exporter.export(f'{tempdir.name}/export_{frame_str}.png')

def update_frame(frame: int) -> None:
    pl.setData(x=x[:, frame], y=y[:, frame], symbolBrush=cm.map(c[:, frame], 'qcolor'), symbolSize=sZ)
    time: int = frame * dt
    txtItem.setText(f't = {time:f} [s]')

if __name__ == '__main__':

    # WCSPH method
    wcsph = WCSPH(height=25.0)

    # Generate some particles
    #N = int(input('Number of particles (has to be a rootable number)?\n'))
    N = 20
    mass = 25 * 25 * wcsph.rho0 / (N ** 2) # Per particle
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
    t_max = 1.0
    #dt = 4.52 * 10 ** -4
    dt = 5e-3
    t = np.arange(0, t_max, dt)
    t_n = len(t)
    H = 25 # water column height

    # Gravity
    gx = 0.
    gy = -9.81

    # WCSPH parameters
    alpha = 0.5
    beta = 0.0

    # Initialize the loop
    for p in particles:
        wcsph.inital_condition(p)

    # Time-stepping
    x = np.zeros([len(particles), t_n]) # X-pos
    y = np.zeros([len(particles), t_n]) # Y-pos
    c = np.zeros([len(particles), t_n]) # Pressure (p)
    u = np.zeros([len(particles), t_n]) # Density (rho)
    v = np.zeros([len(particles), 2, t_n]) # Velocity both x and y
    a = np.zeros([len(particles), 2, t_n]) # Acceleration both x and y
    drho = np.zeros([len(particles), t_n]) # Density (rho)

    # Initialization
    x[:, 0] = [p.r[0] for p in particles]
    y[:, 0] = [p.r[1] for p in particles]
    c[:, 0] = [p.p for p in particles]
    u[:, 0] = [p.rho for p in particles]

    # Initial plot
    # use less ink
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    pg.setConfigOptions(antialias=True)

    # Plot
    pw = pg.plot(title='Dam break (2D)', labels={'left': 'Y [m]', 'bottom': 'X [m]', 'top': 'Dam Break (2D)'})
    pw.setXRange(-3, 81)
    pw.setYRange(-3, 41)

    # Text
    t_step = 0.0
    txtItem = pg.TextItem(text = f't = 0.00000 [s]')
    pw.scene().addItem(txtItem)
    [[xmin, xmax], [ymin, ymax]] = txtItem.getViewBox().viewRange()
    xrange = xmax - xmin
    txtItem.translate(350.0 - xrange, 90.0)
    
    #pw.TextItem(f't = 0.000 s')

    # make colormap
    # TODO: Auto generate these steps
    gradient = 'spectrum'
    n_stops = 10
    c_range = np.linspace(0, 1, num=n_stops)

    # Blue to red color spectrum
    colors = []
    for cc in c_range:
        colors.append([cc, 0.0, 1 - cc, 1.0])
    colors = np.array(colors)
    stops = np.round(c_range * (np.max(c[fluid_ind, 0])) / 1000, 0)
    cm = pg.ColorMap(stops, colors)
    
    # make colorbar, placing by hand
    cb = ColorBar(cm, 10, 200, label='Pressure [kPa]')#, [0., 0.5, 1.0])
    pw.scene().addItem(cb)
    cb.translate(610.0, 90.0)

    # Initial points
    sZ = (40 * (2 / N))
    pl = pw.plot(x[:, 0], y[:, 0], pen=None, symbol='o', symbolBrush=cm.map(c[:, 0] / 1000, 'qcolor'), symbolPen=None, symbolSize=sZ)

    # Frame 0 export.
    tempdir = tempfile.TemporaryDirectory()
    print(f'Exporting frames to directory: {tempdir.name}')
    export(0)
    
    # hdx = 1.3 * np.sqrt(2 * (2 * 2 / (N ** 2)))
    # h = np.ones(len(particles)) * hdx

    # Loopy-loop
    i: int = 0
    for t_step in tqdm(range(t_n - 1), desc='Time-stepping'):
        # Distance and neighbourhood
        r = np.array([p.r for p in particles])
        dist = scipy.spatial.distance.cdist(r, r, 'euclidean')
        # hood = scipy.spatial.cKDTree(r)
        
        # Calculate H
        # J.J. Monaghan (2002), p. 1722
        # d = 2
        # h = 1.3 * np.power(mass / u[:, t_step], 1 / d)

        # Distance of closest particle time 1.3
        h = np.ma.masked_values(dist, 0.0, copy=False).min(1)

        # Force/Acceleration evaluation loop
        i: int = 0
        for p in particles:
            # Initialize particle, reset acceleration and density.
            wcsph.loop_initialize(p)
            
            # Run EOS
            p.p = wcsph.TaitEOS(p)

            # Query neighbours
            h_i: np.array = 0.5 * (h[i] + h[:])
            q_i: np.array = dist[i, :] / (h_i)
            near_arr: np.array = np.flatnonzero(np.argwhere(q_i <= 3.0)) # Find neighbours and remove self (0).

            # Skip if got no neighbours
            # Keep same properties, no acceleration.
            if len(near_arr) == 0:
                i += 1
                continue

            # Set zero variables
            _au = 0.0; _au_d = 0.0; _av = 0.0; _av_d = 0.0
            _arho = 0.0; _xsphx = 0.0; _xsphy = 0.0

            # Calc vectors
            xij: np.array = p.r - r[near_arr, :]
            rij: np.array = dist[i, near_arr]
            vij: np.array = p.v - v[near_arr, :, t_step]

            # Calc averaged properties
            hij: np.array   = h_i[near_arr]
            rhoij: np.array = 0.5 * (p.rho + u[near_arr, t_step])
            cij: np.array   = np.ones(len(near_arr)) * wcsph.co # This is a constant (currently).

            # kernel calculations
            wij = kernel.evaluate(xij, rij, hij)
            dwij = kernel.gradient(xij, rij, hij)

            # Continuity
            vijdotwij = np.sum(vij * dwij, axis=1) # row by row dot product
            _arho = np.sum(mass * vijdotwij, axis=0)

            # Gradient 
            if p.label == 'fluid':
                tmp = p.p / (p.rho * p.rho) + c[near_arr, t_step] / (u[near_arr, t_step] * u[near_arr, t_step])
                _au = np.sum(- mass * tmp * dwij[:, 0], axis=0)
                _av = np.sum(- mass * tmp * dwij[:, 1], axis=0)

                # Diffusion
                dot = np.sum(vij * xij, axis=1) # Row by row dot product
                piij = np.zeros(len(near_arr))

                # Perform diffusion for masked entities
                mask       = dot < 0
                if any(mask):
                    muij       = hij[mask] * dot[mask] / (rij[mask] * rij[mask] + 0.01 * hij[mask] * hij[mask])
                    muij       = muij
                    piij[mask] = muij * (beta * muij - alpha * cij[mask])
                    piij[mask] = piij[mask] / rhoij[mask]

                _au_d = np.sum(- mass * piij * dwij[:, 0])
                _av_d = np.sum(- mass * piij * dwij[:, 1])

                # XSPH
                _xsphtmp = mass / rhoij * wij

                _xsphx = np.sum(_xsphtmp * vij[:, 0], axis=0)
                _xsphy = np.sum(_xsphtmp * vij[:, 1], axis=0)
            # end fluid

            # Store the new properties
            if p.label == 'fluid':
                p.a = np.array([_au + _au_d + gx, _av + _av_d + gy])
                p.v[0] = p.v[0] - _xsphx
                p.v[1] = p.v[1] - _xsphy
            p.drho = _arho

            # Next Particle
            i += 1
        # end force loop
        
        # Integration loop
        i: int = 0
        for p in particles:
            # Integrate the thingies
            integrater.integrate(dt, p, False)
            
            # Set limit density.
            if p.rho < wcsph.rho0:
                p.rho = wcsph.rho0

            # Put into giant-matrix
            x[i, t_step + 1] = p.r[0]
            y[i, t_step + 1] = p.r[1]
            c[i, t_step + 1] = p.p
            u[i, t_step + 1] = p.rho
            v[i, :, t_step + 1] = p.v
            a[i, :, t_step + 1] = p.a
            drho[i, t_step + 1] = p.drho
            
            i += 1

        # Update and export plot
        update_frame(t_step + 1)
        export(t_step + 1)

    # Export to mp4
    fps = np.round(0.5 / dt) # Frames per second
    subprocess.run(f'ffmpeg -hide_banner -loglevel panic -y -r {fps} -i "{tempdir.name}/export_%06d.png" -c:v libx264 -vf fps=10 -pix_fmt yuv420p "{sys.path[0]}/export.mp4"')
    print('Export complete!')
    print(f'Exported to "{sys.path[0]}/export.mp4".')

    # Cleanup export dir
    tempdir.cleanup()

    # Export compressed numpy-arrays
    np.savez_compressed(f'{sys.path[0]}/export', x=x, y=y, c=c, u=u, v=v, a=a, drho=drho)
    print(f'Exported arrays to: "{sys.path[0]}/export.npz".')