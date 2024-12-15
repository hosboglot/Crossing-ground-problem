import numpy as np
import numpy.typing as npt


class HermiteInterpolation:
    def __init__(self, coefficients, offset: float = 0) -> None:
        self._coefs = np.asarray(coefficients)
        self._offset = offset

    def __call__(self, t: npt.ArrayLike) -> npt.ArrayLike:
        f1 = self._coefs
        f2 = np.pow((t - self._offset), np.arange(0, len(self._coefs)))
        return np.inner(f1, f2)

    @property
    def d(self):
        return HermiteInterpolation(
            (self._coefs * np.arange(0, len(self._coefs)))[1:],
            self._offset
            )

    @classmethod
    def from_2p2v(cls,
                  t1, x1, dx1,
                  t2, x2, dx2
                  ):
        """ 2 points, 2 values each (value and derivative)"""
        h = t2 - t1
        interpolator = HermiteInterpolation(
            [
                x1,
                dx1,
                3 * (x2 - x1) / h**2 - (dx2 + 2*dx1) / h,
                2 * (x1 - x2) / h**3 + (dx1 + dx2) / h**2
            ],
            t1)

        return interpolator
