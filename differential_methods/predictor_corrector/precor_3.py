import numpy as np
from .precor_base import PredictorCorrectorBase


class PredictorCorrector3(PredictorCorrectorBase):
    order = 3

    def predict(self, y_vals, f_vals, h):
        return y_vals[-1] + h * (23*f_vals[-1] - 16*f_vals[-2] + 5*f_vals[-3]) / 12

    def correct(self, y_vals, f_vals, h, x_next, y_pred, max_iter=5, tol=1e-8):
        y_corr = y_pred
        for _ in range(max_iter):
            f_next = self._f(x_next, y_corr)
            y_new = y_vals[-1] + h * (5*f_next + 8*f_vals[-1] - f_vals[-2]) / 12
            error = np.linalg.norm(y_new - y_corr) if isinstance(y_new, np.ndarray) else abs(y_new - y_corr)
            if error < tol:
                break
            y_corr = y_new
        return y_corr
