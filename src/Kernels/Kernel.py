import abc
import numpy as np

class Kernel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def evaluate(self, x1: np.array, x2: np.array) -> float:
        pass

    @abc.abstractmethod
    def gradient(self, x1: np.array, x2: np.array) -> float:
        pass