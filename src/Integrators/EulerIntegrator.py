from src.Particle import Particle
from src.Integrators.Integrator import Integrator

class EulerIntegrator(Integrator):
    """ Stupidly simple Euler Integrator """
    @classmethod
    def integrate(self, dt: float, p: Particle) -> Particle:
        """
        Parameters
        ----------

        dt: time-step

        p: Particle to integrate
        """

        # Only move fluid particles.
        if p.label == 'fluid':
            p.v = p.v + dt * p.a
            p.r = p.r + dt * p.v

        p.rho = p.rho + dt * p.drho

        return p