import abc
import numpy as np


class Kernel(metaclass=abc.ABCMeta):
    @abc.abstractstaticmethod
    def evaluate(r: np.array, h: np.array) -> np.array:
        pass

    @abc.abstractstaticmethod
    def gradient(x: np.array, r: np.array, h: np.array) -> np.array:
        pass
