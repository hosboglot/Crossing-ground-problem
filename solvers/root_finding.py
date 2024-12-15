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
        self._ts = [t1, t2]

    def solution(self):
        return np.asarray(self._ts)

    def step(self):
        diff = (self._f(self._ts[-1]) - self._f(self._ts[-2])) / \
                    (self._ts[-1] - self._ts[-2])
        self._ts.append(
            self._ts[-2] -
            self._f(self._ts[-2]) / diff
            )
        return self._ts[-2:]


class NewtonSolver(RootSolverBase):
    def __init__(self,
                 func: Callable[[float], float],
                 func_d: Callable[[float], float],
                 t_last: float):
        self._f = func
        self._f_d = func_d
        self._ts = [t_last]

    def solution(self):
        return np.asarray(self._ts)

    def step(self):
        self._ts.append(
            self._ts[-1] -
            self._f(self._ts[-1]) / self._f_d(self._ts[-1])
            )
        return self._ts[-2:]
