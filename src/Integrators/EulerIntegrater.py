from src.Particle import Particle


class EulerIntegrater():
    """ Stupidly simple Euler Integrator """

    def integrate(self, dt: float, p: Particle):
        """
        Parameters:
        dt: time-step
        p: Particle to integrate
        """

        # Don't move boundary particles.
        if p.label == 'boundary':
            return
        p.v += dt * p.a
        p.r += dt * p.v
        p.rho += dt * p.drho
