import abc
import numpy as np


class Kernel(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def evaluate(self, r: np.array, h: np.array) -> np.array:
        pass

    def derivative(self, r: np.array, h: np.array) -> np.array:
        pass

    @abc.abstractmethod
    def gradient(self, x: np.array, r: np.array, h: np.array) -> np.array:
        pass
