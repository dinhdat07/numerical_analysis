import numpy as np

class ODESystem:
    def __init__(self, f_vec, y0, x0, h, x_end, label=""):
        self.f_vec = f_vec  # callable: f_vec(x, Y) → np.ndarray
        self.y0 = np.array(y0, dtype=float)
        self.x0 = x0
        self.h = h
        self.x_end = x_end
        self.label = label

    def n_steps(self):
        return int((self.x_end - self.x0) / self.h)

    def __str__(self):
        return (
            f"{self.label}: dY/dx = f_vec(x, Y), "
            f"(x0, Y0) = ({self.x0}, {self.y0}), h = {self.h}, x_end = {self.x_end}"
        )

    def __repr__(self):
        return self.__str__()
