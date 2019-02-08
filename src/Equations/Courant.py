class Courant():
    alpha: float

    def __init__(self, alpha = 0.4):
        self.alpha = alpha

    def calc(self, h_min: float, c_max: float):
        return self.alpha * h_min / c_max