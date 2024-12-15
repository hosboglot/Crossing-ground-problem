from typing import Callable
import numpy as np


class IvpSolverBase:
    def __init__(self, rhs: Callable[[float, np.ndarray], np.ndarray],
                 t_0: float, x_0: np.ndarray, step: float = 1e-2):
        raise NotImplementedError()

    def solution(self):
        raise NotImplementedError()

    def step(self):
        raise NotImplementedError()

    def solve(self, stop_condition: Callable[[], bool]):
        raise NotImplementedError()


class RKSolver(IvpSolverBase):
    """Explicit RK method of 4-th order"""
    def __init__(self, rhs: Callable[[float, np.ndarray], np.ndarray],
                 t_0: float, x_0: np.ndarray, step: float = 1e-2):
        self._step = step
        self._rhs = rhs
        self._eq_n = len(x_0)
        self._ts = [t_0]
        self._xs = [np.asarray(x_0)]
        self._ks = np.zeros((4, self._eq_n))

    def solution(self):
        return np.asarray(self._ts), np.asarray(self._xs).T

    def step(self):
        self._ks[0] = self._rhs(self._ts[-1],
                                self._xs[-1])
        self._ks[1] = self._rhs(self._ts[-1] + self._step / 2,
                                self._xs[-1] + (self._step / 2) * self._ks[0])
        self._ks[2] = self._rhs(self._ts[-1] + self._step / 2,
                                self._xs[-1] + (self._step / 2) * self._ks[1])
        self._ks[3] = self._rhs(self._ts[-1] + self._step,
                                self._xs[-1] + self._step * self._ks[1])
        self._ts.append(self._ts[-1] + self._step)
        self._xs.append(self._xs[-1] + (self._step / 6.) *
                        (self._ks[0] + 2*self._ks[1] +
                         2*self._ks[2] + self._ks[3]))
        return self._ts[-1], self._xs[-1]

    def solve(self, stop_condition: Callable[[], bool]):
        '''
        Solve until stop_condition evaluates to True

        stop_condition takes args t, x
        '''
        while not stop_condition(self._ts[-1], self._xs[-1]):
            self.step()
        return self.solution()


def solveRK(rhs: Callable[[float, np.ndarray], np.ndarray],
            t_0: float, x_0: np.ndarray,
            stop_condition: Callable[[], bool], step: float = 1e-2):
    solver = RKSolver(rhs, t_0, x_0, step)
    solver.solve(stop_condition)
    return solver.solution()


class AdamsSolver(IvpSolverBase):
    """
    Implicit Adams method of 4-th order
    with explicit Adams prediction
    """
    def __init__(self, rhs: Callable[[float, np.ndarray], np.ndarray],
                 t_0: float, x_0: np.ndarray, step: float = 1e-2):
        self._step = step
        self._rhs = rhs
        self._eq_n = len(x_0)
        self._ts = [t_0]
        self._xs = [np.asarray(x_0)]
        self._fs = [rhs(t_0, x_0)]

        self._start_solver = RKSolver(rhs, t_0, x_0, step)

    def solution(self):
        return np.asarray(self._ts), np.asarray(self._xs).T

    def step(self):
        if len(self._xs) < 3:
            t, x = self._start_solver.step()
        else:
            t, x = self._correct(self._predict())
        self._ts.append(t)
        self._xs.append(x)
        self._fs.append(self._rhs(t, x))

        return self._ts[-1], self._xs[-1]

    def solve(self, stop_condition: Callable[[], bool]):
        '''
        Solve until stop_condition evaluates to True

        stop_condition takes args t, x
        '''
        while not stop_condition(self._ts[-1], self._xs[-1]):
            self.step()
        return self.solution()

    def _predict(self) -> np.ndarray:
        """Explicit Adams prediction"""
        return self._xs[-1] + self._step * \
            np.dot((5, -16, 23), self._fs[-3:]) / 12

    def _correct(self, prediction: np.ndarray) -> tuple[float, np.ndarray]:
        t = self._ts[-1] + self._step
        return t, self._xs[-1] + self._step * \
            np.dot(
                (1, -5, 19, 9),
                self._fs[-3:] + [self._rhs(t, prediction)]
                ) / 24


def solveAdams(rhs: Callable[[float, np.ndarray], np.ndarray],
               t_0: float, x_0: np.ndarray,
               stop_condition: Callable[[], bool], step: float = 1e-2):
    solver = AdamsSolver(rhs, t_0, x_0, step)
    solver.solve(stop_condition)
    return solver.solution()
