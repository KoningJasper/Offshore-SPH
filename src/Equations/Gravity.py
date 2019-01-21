import numpy as np


class Gravity():
    g: np.array
    def __init__(self, gx: float, gy: float):
        self.g = np.array([gx, gy])

    def calc(self, a: np.array) -> np.array:
        return a + self.g