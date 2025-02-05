from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import scipy.sparse as ssparse

from solvers.ivp import IvpSolverBase, RKSolver, AdamsSolver


@dataclass
class Conditions:
    x_0: np.ndarray[float]
    v_0: np.ndarray[float]
    w: np.ndarray[float]
    'Self frequencies of springs, n values'
    k_l: np.ndarray[float]
    'Coefficients between i and i-1 springs, n-1 values'
    k_r: np.ndarray[float]
    'Coefficients between i and i+1 springs, n-1 values'
    t_0: float = 0
    'Start time'
    t_end: float | None = None
    'Time until run evaluations, None for infinite'


class BoundSpringsSolver:
    def __init__(self, step: float = 1e-2,
                 ode_method: Literal['RK', 'Adams'] = 'Adams',
                 conditions: Conditions | None = None) -> None:
        self.step = step
        self.ode_method = ode_method
        self._conditions = conditions

        self._solution: tuple[np.NDArray[float], np.NDArray[float]] = None

    @property
    def conditions(self):
        return self._conditions

    @conditions.setter
    def conditions(self, new: Conditions):
        if self._conditions != new:
            self._conditions = new

    @property
    def solution(self):
        return self._solution[0], self._solution[1][:self.conditions.x_0.shape[0]]

    @property
    def velocities(self):
        return self._solution[0], self._solution[1][self.conditions.x_0.shape[0]:]

    def solve(self):
        if self._conditions is None:
            raise ValueError('Conditions are not set')
        if self._conditions.t_end is None:
            raise ValueError('End time is not set')

        self._solution = self._solve_ode()

    def make_steps(self):
        if self._conditions is None:
            raise ValueError('Conditions are not set')
        
        solver = self._make_solver()

        while True:
            t_last, x_last = solver.step()
            yield t_last, x_last[:self.conditions.x_0.shape[0]]

    def _solve_ode(self):
        solver = self._make_solver()

        t_last = self.conditions.t_0
        while t_last < self.conditions.t_end:
            t_last, x_last = solver.step()

        return solver.solution()

    def _make_solver(self):
        n = self.conditions.x_0.shape[0]
        local_K = ssparse.diags_array(
            (
                self.conditions.w**2 - np.hstack((self.conditions.k_r, 0)) - np.hstack((0, self.conditions.k_l)),
                self.conditions.k_r,
                self.conditions.k_l,
            ),
            offsets=(0, 1, -1)
        )
        K = ssparse.block_array((
            (None, ssparse.eye_array(n)),
            (local_K, None)
        ))

        def rhs(t: float, x: np.ndarray):
            return K @ x

        t_last, x_last = self.conditions.t_0, np.hstack((self.conditions.x_0, self.conditions.v_0))
        match self.ode_method.lower():
            case 'rk':
                solver = RKSolver(rhs, t_last, x_last, self.step)
            case 'adams':
                solver = AdamsSolver(rhs, t_last, x_last, self.step)

        return solver

def main():
    n = 5000
    # x_0 = np.hstack((1, np.zeros(n-1)))
    x_0 = np.hstack((np.sin(np.linspace(0, (n // 6) * 2*np.pi, n // 6)), np.zeros(n-n//6))) * 10
    # x_0 = np.sin(np.linspace(0, 2*np.pi, n)) * 10
    cond = Conditions(
        x_0=x_0,
        v_0=np.zeros(n),
        t_0=0,
        w=np.hstack((0, np.zeros(n-2), 0)),
        k_r=10 * n*np.hstack((0, np.ones(n-2))),
        k_l=10 * n*np.hstack((np.ones(n-2), 0))
        )
    solver = BoundSpringsSolver(0.001, conditions=cond)

    import time
    import pyqtgraph as pg
    
    app = pg.mkQApp()
    x_range = np.arange(len(x_0))
    plot = pg.plot(x_range, x_0)
    y_limit = np.max(np.abs(x_0))
    plot.setYRange(-y_limit, y_limit)

    steps = solver.make_steps()
    start_time = time.time()
    def update():
        while True:
            t, xs = next(steps)
            if t >= (time.time() - start_time):
                break
        plot.plotItem.dataItems[0].setData(x_range, xs)
        plot.setWindowTitle(f'Time: {t}')

    timer = pg.QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(10)
    app.exec()


if __name__ == '__main__':
    main()

