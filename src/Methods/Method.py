import abc
import numpy as np

class Method():
    """ 
        Generic method.
        
        Will include all possible variables to the underlying functions so they can decide what to use.
    """
    @abc.abstractmethod
    def initialize(self, pA: np.array) -> np.array:
        raise Exception('No implemented!')

    @abc.abstractmethod
    def compute_speed_of_sound(self, pA: np.array) -> np.array:
        raise Exception('No implemented!')

    @abc.abstractmethod
    def compute_pressure(self, pA: np.array) -> np.array:
        raise Exception('No implemented!')

    @abc.abstractmethod
    def compute_density_change(self, p: Particle, vij: np.array, dwij: np.array) -> float:
        raise Exception('No implemented!')

    @abc.abstractmethod
    def compute_acceleration(self, i: int, p: Particle, xij: np.array, rij: np.array, vij: np.array, pressure: np.array, rho: np.array, hij: np.array, cij: np.array, wij: np.array, dwij: np.array) -> np.array:
        raise Exception('No implemented!')

    @abc.abstractmethod
    def compute_velocity(self, i: int, p: Particle) -> np.array:
        raise Exception('No implemented!')