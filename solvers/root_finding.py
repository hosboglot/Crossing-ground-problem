from typing import Callable
import numpy as np


class RootSolverBase:
    def solution(self):
        raise NotImplementedError()

    def step(self):
        raise NotImplementedError()


class SecantSolver(RootSolverBase):
    def __init__(self,
                 func: Callable[[float], float],
                 t1: float, t2: float):
        self._f = func

        self._fs = [self._f(t1), self._f(t2)]
        self._ts = [t1, t2]

    def solution(self):
        return np.asarray(self._ts)

    def step(self):
        diff = (self._fs[-1] - self._fs[-2]) / \
            (self._ts[-1] - self._ts[-2])
        t = self._ts[-2] - self._fs[-2] / diff

        self._ts.append(t)
        self._fs.append(self._f(t))
        return self._ts[-2:]


class NewtonSolver(RootSolverBase):
    def __init__(self,
                 func: Callable[[float], float],
                 func_d: Callable[[float], float],
                 t_last: float):
        self._f = func
        self._f_d = func_d

        self._fs = [self._f(t_last)]
        self._fs_d = [self._f_d(t_last)]
        self._ts = [t_last]

    def solution(self):
        return np.asarray(self._ts)

    def step(self):
        t = self._ts[-1] - self._fs[-1] / self._fs_d[-1]
        self._ts.append(t)
        self._fs.append(self._f(t))
        self._fs_d.append(self._f_d(t))
        return self._ts[-2:]


class HalleySolver(RootSolverBase):
    def __init__(self,
                 func: Callable[[float], float],
                 func_d: Callable[[float], float],
                 func_d2: Callable[[float], float],
                 t_last: float):
        self._f = func
        self._f_d = func_d
        self._f_d2 = func_d2

        self._fs = [self._f(t_last)]
        self._fs_d = [self._f_d(t_last)]
        self._fs_d2 = [self._f_d2(t_last)]
        self._ts = [t_last]

    def solution(self):
        return np.asarray(self._ts)

    def step(self):
        t = self._ts[-1] - (2 * self._fs[-1] * self._fs_d[-1]) / \
            (2 * self._fs_d[-1]**2 - self._fs[-1] * self._fs_d2[-1])

        self._ts.append(t)
        self._fs.append(self._f(t))
        self._fs_d.append(self._f_d(t))
        self._fs_d2.append(self._f_d2(t))
        return self._ts[-2:]
