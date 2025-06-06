from .precor_base import PredictorCorrectorBase


class PredictorCorrector4(PredictorCorrectorBase):
    order = 4

    def predict(self, y_vals, f_vals, h):
        return y_vals[-1] + h * (55*f_vals[-1] - 59*f_vals[-2] + 37*f_vals[-3] - 9*f_vals[-4]) / 24

    def correct(self, y_vals, f_vals, h, x_next, y_pred, max_iter=5, tol=1e-8):
        y_corr = y_pred
        for _ in range(max_iter):
            f_next = self._f(x_next, y_corr)
            y_new = y_vals[-1] + h * (9*f_next + 19*f_vals[-1] - 5*f_vals[-2] + f_vals[-3]) / 24
            if abs(y_new - y_corr) < tol:
                break
            y_corr = y_new
        return y_corr
