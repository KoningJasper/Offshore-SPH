from src.Particle import Particle


class EulerIntegrater():
    """ Stupidly simple Euler Integrator """

    @classmethod
    def integrate(self, dt: float, p: Particle):
        """
        Parameters:
        dt: time-step
        p: Particle to integrate
        """

        # Only move fluid particles.
        if p.label == 'fluid':
            p.v += dt * p.a
            p.r += dt * p.v

        p.rho += dt * p.drho
