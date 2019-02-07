import numpy as np
from src.Common import get_label_code
from src.Integrators.Integrator import Integrator

class EulerIntegrator(Integrator):
    """ Stupidly simple Euler Integrator """
    @classmethod
    def integrate(self, dt: float, p: np.array) -> np.array:
        """
        Parameters
        ----------

        dt: time-step

        p: Particle to integrate
        """

        # Only move fluid particles.
        if p['label'] == get_label_code('fluid'):
            p['vx'] = p['vx'] + dt * p['ax']
            p['vy'] = p['vy'] + dt * p['ay']

            p['x'] = p['x'] + dt * p['vx'] + 0.5 * dt * dt * p['ax']
            p['y'] = p['y'] + dt * p['vy'] + 0.5 * dt * dt * p['ay']

        p['rho'] = p['rho'] + dt * p['drho']

        return p