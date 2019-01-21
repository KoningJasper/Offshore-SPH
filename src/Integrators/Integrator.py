import abc
from src.Particle import Particle

class Integrator():
    @abc.abstractmethod
    def integrate(self, dt: float, p: Particle):
        pass