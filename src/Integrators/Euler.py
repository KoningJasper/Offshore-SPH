import numpy as np
from src.Common import ParticleType
from src.Integrators.Integrator import Integrator

class Euler(Integrator):
    """ Stupidly simple Euler Integrator """

    def isMultiStage(self) -> bool:
        return False

    def predict(self, dt: float, p: np.array) -> np.array:
        """ Predict does nothing in eurler-integrator. """
        return p

    def correct(self, dt: float, p: np.array) -> np.array:
        """
        Parameters
        ----------

        dt: time-step

        p: Particle to integrate
        """

        # Only move fluid particles.
        if p['label'] == ParticleType.Fluid:
            p['x'] = p['x'] + dt * p['vx'] + 0.5 * dt * dt * p['ax']
            p['y'] = p['y'] + dt * p['vy'] + 0.5 * dt * dt * p['ay']

            p['vx'] = p['vx'] + dt * p['ax']
            p['vy'] = p['vy'] + dt * p['ay']


        p['rho'] = p['rho'] + dt * p['drho']

        return p