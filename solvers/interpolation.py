import numpy as np
import numpy.typing as npt


class InterpolatorBase:
    def __init__(self):
        raise NotImplementedError()

    def __call__(self, t: npt.ArrayLike) -> npt.ArrayLike:
        raise NotImplementedError()

    @property
    def d(self) -> 'InterpolatorBase':
        raise NotImplementedError()


class HermiteInterpolator(InterpolatorBase):
    def __init__(self, coefficients, offset: float = 0) -> None:
        self._coefs = np.asarray(coefficients)
        self._offset = offset

    def __call__(self, t: npt.ArrayLike) -> npt.ArrayLike:
        f1 = self._coefs
        f2 = np.pow((t - self._offset), np.arange(0, len(self._coefs)))
        return np.inner(f1, f2)

    @property
    def d(self):
        return HermiteInterpolator(
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
        interpolator = HermiteInterpolator(
            [
                x1,
                dx1,
                3 * (x2 - x1) / h**2 - (dx2 + 2*dx1) / h,
                2 * (x1 - x2) / h**3 + (dx1 + dx2) / h**2
            ],
            t1)

        return interpolator


class LagrangeInterpolator(InterpolatorBase):
    def __init__(self, coefficients, offsets):
        self._coefs = np.asarray(coefficients)
        self._offsets = np.asarray(offsets)

    def __call__(self, t: npt.ArrayLike) -> npt.ArrayLike:
        t = np.asarray(t)
        offset_matrix = t - self._offsets[:, None]
        eye = np.identity(len(self._offsets))
        f0 = offset_matrix.T[:, None]
        f1 = f0 * (1 - eye) + eye
        f2 = np.prod(f1, axis=-1)
        if self._coefs.ndim > 1:
            f3 = np.sum(f2[:, :, None] * self._coefs, axis=1)
        else:
            f3 = np.sum(f2 * self._coefs, axis=1)
        return f3

    @classmethod
    def from_points(cls, ts: npt.ArrayLike, xs: npt.ArrayLike):
        """
            n points, m values at each
            ts.shape == (n,), xs.shape == (n,) or (n, m)
        """
        ts = np.asarray(ts)
        xs = np.asarray(xs)
        offset_matrix = (ts + np.identity(len(ts))).T
        offset_matrix -= ts
        if xs.ndim > 1:
            coefs = xs / np.prod(offset_matrix, axis=1)[:, None]
        else:
            coefs = xs / np.prod(offset_matrix, axis=1)
        return LagrangeInterpolator(coefs, ts)


class LocalPolynomialInterpolator(InterpolatorBase):
    """
        Local polynomial interpolation,
        based on Lagrange interpolator of 3rd order
    """
    def __init__(self, interpolator_dict: dict[float, LagrangeInterpolator]):
        self._intp_dict = interpolator_dict

    def __call__(self, ts: np.float64) -> npt.ArrayLike:
        if np.ndim(ts) == 0:
            nodes = sorted(self._intp_dict.keys())
            if ts < nodes[0]:
                return self._intp_dict[nodes[0]](ts)[0]
            for j in range(len(nodes)):
                if ts < nodes[j]:
                    return self._intp_dict[nodes[j-1]](ts)[0]
            else:
                return self._intp_dict[nodes[-1]](ts)[0]

        ts = np.asarray(ts)
        res = []
        nodes = sorted(self._intp_dict.keys())
        for i in range(len(ts)):
            if ts[i] < nodes[0]:
                res.append(self._intp_dict[nodes[0]](ts[i])[0])
                continue
            for j in range(1, len(nodes)):
                if ts[i] < nodes[j]:
                    res.append(self._intp_dict[nodes[j-1]](ts[i])[0])
                    break
            else:
                res.append(self._intp_dict[nodes[-1]](ts[i])[0])
        return np.asarray(res)

    @property
    def d(self):
        res_dict = {}
        for node, intp in self._intp_dict.items():
            res_dict[node] = intp.d
        return LocalPolynomialInterpolator(res_dict)

    @classmethod
    def from_points(cls, ts: npt.ArrayLike, xs: npt.ArrayLike):
        """
            n points, m values at each
            ts.shape == (n,), xs.shape == (n,) or (n, m)
        """
        ts = np.asarray(ts)
        xs = np.asarray(xs)
        if len(ts) <= 4:
            return LocalPolynomialInterpolator(
                {0: LagrangeInterpolator.from_points(ts, xs)})

        intp_dict = {}
        intp_dict[ts[0]] = LagrangeInterpolator.from_points(ts[:3], xs[:3])
        intp_dict[ts[-1]] = LagrangeInterpolator.from_points(ts[-3:], xs[-3:])
        for i in range(1, len(ts)-1):
            intp_dict[ts[i]] = LagrangeInterpolator.from_points(
                ts[i-1:i+3], xs[i-1:i+3])

        return LocalPolynomialInterpolator(intp_dict)


# class LeastSquaresApproximator(InterpolatorBase):
#     def __init__(self, coefficients):
#         self._coefs = coefficients

#     @classmethod
#     def from_points(cls, ts: npt.ArrayLike, xs: npt.ArrayLike, order=4):
#         from solvers.root_finding import NewtonSolver
#         minimizer = NewtonSolver(
#             lambda t: (t - )
#         )
