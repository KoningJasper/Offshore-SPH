import abc
import numpy as np
from typing import List

class Method():
    """ 
        Generic method.
        
        Will include all possible variables to the underlying functions so they can decide what to use.
    """
    @abc.abstractmethod
    def density():
        raise Exception('No implemented!')
        
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
    def compute_density_change(self, p: np.array, comp: np.array) -> float:
        raise Exception('No implemented!')

    @abc.abstractmethod
    def compute_acceleration(self, p: np.array, comp: np.array) -> List[float]:
        raise Exception('No implemented!')

    @abc.abstractmethod
    def compute_velocity(self, p: np.array, comp: np.array) -> List[float]:
        raise Exception('No implemented!')