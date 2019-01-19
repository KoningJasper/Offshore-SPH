# External libraries
from tqdm import tqdm
import scipy.spatial
import numpy as np

# Add parent folder to path
import sys
sys.path.append("..")

# Own stuff
from src.Kernels.Gaussian import Gaussian
from src.Particle import Particle
from src.Equations.WCSPH import WCSPH
from src.Animation import Animation
from src.Integrators.EulerIntegrater import EulerIntegrater

def create_particles(wcsph, N, rho_b: float = 1005.):
    """ Create some particles """
    xv = np.linspace(0, 2, N)
    yv = np.linspace(0, 2, N)
    x, y = np.meshgrid(xv, yv, indexing='ij')
    x = x.ravel()
    y = y.ravel()

    particles = []
    for i in range(len(x)):
        particles.append(Particle('fluid', x[i], y[i]))

    r0 = 0.1  # Distance between boundary particles.

    # Maximum and minimum values of boundaries
    x_min = -1
    x_max = 10
    y_min = -1
    y_max = 5

    # Bottom & Top
    xv = np.arange(x_min, x_max, r0)
    yv = np.zeros(len(xv)) + y_min
    for i in range(len(xv)):
        particles.append(Particle('boundary', xv[i], yv[i], rho_b))
        particles.append(
            Particle('boundary', xv[i] - r0 / 2, yv[i] - r0, rho_b))
        particles.append(Particle('boundary', xv[i], yv[i] - 2 * r0, rho_b))
        particles.append(
            Particle('boundary', xv[i] - r0 / 2, yv[i] - 3 * r0, rho_b))
        #particles.append(Particle('boundary', xv[i], yv2[i] + r0, rho_b))

    # Left & Right
    yv3 = np.arange(y_min, y_max, r0)
    xv2 = np.zeros(len(yv3)) + x_min
    xv3 = np.zeros(len(yv3)) + x_max
    for i in range(len(yv3)):
        particles.append(Particle('boundary', xv2[i], yv3[i], rho_b))
        particles.append(
            Particle('boundary', xv2[i] - r0, yv3[i] - r0 / 2, rho_b))
        particles.append(Particle('boundary', xv3[i], yv3[i], rho_b))
        particles.append(
            Particle('boundary', xv3[i] + r0, yv3[i] - r0 / 2, rho_b))

    return particles


def run(rho_b: float = 1005.):
    # Runs a (2D) dambreak example
    # WCSPH method
    wcsph = WCSPH(height=2.0)

    # Generate some particles
    # N * N (water-particles).
    print('Creating Particles.')
    particles = create_particles(wcsph, 10, rho_b)
    fluid_particles = [p for p in particles if p.label == 'fluid']
    mass = 2 * 2 / len(fluid_particles)
    print(f'Created {len(fluid_particles)} fluid-particles.')
    print(f'Created {len(particles) - len(fluid_particles)} boundary-particles.')
    print(f'For a total of {len(particles)} particles.')

    # Solving properties
    kernel = Gaussian()
    integrater = EulerIntegrater()

    # Time-step properties
    t_max = 1.5 # [s]
    dt = 0.02   # [s]
    t = np.arange(0, t_max, dt)
    t_n = len(t)

    # water column height
    H = 2 # [m]

    # H-parameter
    hdx = 1.3 * np.sqrt(H * 2 * 2 / len(fluid_particles))
    h = np.ones(len(particles)) * hdx

    # Gravity
    gx = 0.
    gy = -9.81

    # Initialize the loop
    for p in particles:
        wcsph.inital_condition(p)

    # Time-stepping
    x = np.zeros([len(particles), t_n])  # X-pos
    y = np.zeros([len(particles), t_n])  # Y-pos
    c = np.zeros([len(particles), t_n])  # Pressure (p)
    u = np.zeros([len(particles), t_n])  # Density (rho)
    v = np.zeros([len(particles), 2, t_n])  # Velocity both x and y

    # Initialization
    x[:, 0] = [p.r[0] for p in particles]
    y[:, 0] = [p.r[1] for p in particles]
    c[:, 0] = [p.p for p in particles]
    u[:, 0] = [p.rho for p in particles]

    # Integration loop
    i: int = 0
    for t_step in tqdm(range(t_n - 1), desc='Time-stepping'):
        # Distance and neighbourhood
        r = np.array([p.r for p in particles])
        dist = scipy.spatial.distance.cdist(r, r, 'euclidean')
        hood = scipy.spatial.cKDTree(r)

        # Force/Acceleration evaluation loop
        i: int = 0
        for p in tqdm(particles, desc='Evaluating equations', leave=False):
            if p.label == 'boundary':
                continue

            wcsph.loop_initialize(p)

            # Query neighbours
            r_dist: float = 3.01 * h[0]  # Goes to zero when q > 3
            near_ind: list = hood.query_ball_point(p.r, r_dist)
            near_arr: np.array = np.array(near_ind)

            # Calculate some general properties
            xij: np.array = r[near_arr] - p.r
            rij: np.array = dist[i, near_arr]
            vij: np.array = v[near_arr, :, t_step] - p.v
            dwij: np.array = kernel.gradient(xij, rij, h[near_arr])

            # Evaluate the equations
            wcsph.TaitEOS(p)
            wcsph.Continuity(mass, p, xij, rij, dwij, vij)
            wcsph.Momentum(mass, p, c[near_arr, t_step],
                           u[near_arr, t_step], dwij)
            wcsph.Gravity(p, gx, gy)

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
    print('Integration complete.')
    # END Integration loop

    # Export
    print('Starting export.')
    fluid_ind = []
    solid_ind = []
    i: int = 0
    for p in particles:
        if p.label == 'fluid':
            fluid_ind.append(i)
        else:
            solid_ind.append(i)
        i += 1
    Animation(x=x[fluid_ind], y=y[fluid_ind], r=3.0, c=c[fluid_ind], fps=20, xlim=[-1, 11],
              ylim=[-1, 11], xsolid=x[solid_ind], ysolid=y[solid_ind]).export(f'dam_break_simple_{rho_b}.mp4')
    print('Export complete.')


# Run from cmd
if __name__ == "__main__":
    rho_b = 1010.
    run(rho_b=rho_b)
