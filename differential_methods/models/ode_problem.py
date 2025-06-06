from sympy import Float


class ODEProblem:
    def __init__(self, f, x0, y0, h, y_exact, x_end, label=""):
        self.f = f
        self.x0 = x0
        self.y0 = y0
        self.h = h
        self.y_exact = y_exact
        self.x_end = x_end    
        self.label = label

    def __str__(self):
        return (
            f"{self.label}: dy/dt = {self.f}, y_exact = {self.y_exact}, "
            f"(x0, y0) = ({self.x0}, {self.y0}), h = {self.h}, x_end = {self.x_end}"
        )

    def __repr__(self):
        return self.__str__()
