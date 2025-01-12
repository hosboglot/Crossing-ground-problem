from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from solvers.ivp import RKSolver, AdamsSolver
from solvers.root_finding import (
    SecantSolver, NewtonSolver, HalleySolver
)
from solvers.interpolation import HermiteInterpolator


@dataclass
class Conditions:
    x_0: tuple[float, float, float, float]
    m: float
    k: float
    mu: Callable[[float], float]
    mu_d: Callable[[float], float] | None = None
    mu_d2: Callable[[float], float] | None = None
    t_0: float = 0
    g: float = 9.8
    b: float = 0


class CrossingGroundSolver:
    def __init__(self,
                 step: float = 1e-2,
                 ode_method: Literal['RK', 'Adams'] = 'Adams',
                 root_method: Literal['Secant', 'Newton', 'Halley'] = 'Secant',
                 conditions: Conditions | None = None,
                 ensure_start: bool = False):
        self.step = step
        self.ode_method = ode_method
        self.root_method = root_method
        self.conditions = conditions
        self.ensure_start = ensure_start

        self._ode_solution: tuple[np.NDArray[float], np.NDArray[float]] = None
        self._root_solution: tuple[np.NDArray[float], np.NDArray[float]] = None
        self._crossing: tuple[float, float, float] = None

    def __repr__(self) -> str:
        return "CGSolver(" \
              f"step={self.step!r}, " \
              f"ode_method={self.ode_method!r}, " \
              f"root_method={self.root_method!r}" \
               ")"

    @property
    def ode_solution(self):
        '''Returns ts, xs'''
        return self._ode_solution

    @property
    def root_solution(self):
        '''Returns ts, xs'''
        return self._root_solution

    @property
    def solution(self):
        '''Returns ts, xs'''
        return np.hstack((self._ode_solution[0], self._root_solution[0])), \
            np.hstack((self._ode_solution[1], self._root_solution[1]))

    @property
    def crossing(self) -> tuple[float, float, float]:
        '''Returns t, x1, x2'''
        return self._crossing

    def solve(self):
        if self.conditions is None:
            raise ValueError("No conditions given")

        self._ode_solution = self._solve_ode()
        if len(self._ode_solution[0]) == 1:
            return False
        self._root_solution = self._solve_root()
        self._crossing = (self._root_solution[0][-1],     # t
                          self._root_solution[1][0][-1],  # x1
                          self._root_solution[1][1][-1])  # x2

        return True

    def _solve_ode(self):
        def rhs(t: float, x: np.ndarray):
            return np.asarray([
                x[2],
                x[3],
                -(self.conditions.k / self.conditions.m) * x[2],
                -self.conditions.g - (
                    self.conditions.k / self.conditions.m) * x[3]
                ])

        t_last, x_last = self.conditions.t_0, np.asarray(self.conditions.x_0)
        match self.ode_method.lower():
            case 'rk':
                solver = RKSolver(rhs, t_last, x_last, self.step)
            case 'adams':
                solver = AdamsSolver(rhs, t_last, x_last, self.step)

        start_flag = self.ensure_start
        while start_flag or self.conditions.mu(x_last[0]) <= x_last[1]:
            start_flag = False
            t_last, x_last = solver.step()

        return solver.solution()

    def _solve_root(self):
        ts, xs = self.ode_solution
        ts, xs = ts[-2:], xs[:, -2:]

        interp1 = HermiteInterpolator.from_2p2v(ts[0], xs[0, 0], xs[2, 0],
                                                ts[1], xs[0, 1], xs[2, 1])
        interp2 = HermiteInterpolator.from_2p2v(ts[0], xs[1, 0], xs[3, 0],
                                                ts[1], xs[1, 1], xs[3, 1])

        t1, t2 = ts[0], ts[1]
        xs = [np.asarray([xs[0, 0], xs[1, 0], xs[2, 0], xs[3, 0]]),
              np.asarray([xs[0, 1], xs[1, 1], xs[2, 1], xs[3, 1]])]

        def f(t: float):
            return interp2(t) - self.conditions.mu(interp1(t))

        # match self._root_method.lower(), self.conditions.mu_d:
        if self.root_method.lower() == 'newton':
            def f_d(t: float):
                return interp2.d(t) - \
                    self.conditions.mu_d(interp1(t)) * interp1.d(t)
            solver = NewtonSolver(f, f_d, t2)
        elif self.root_method.lower() == 'halley':
            def f_d(t: float):
                return interp2.d(t) - \
                    self.conditions.mu_d(interp1(t)) * interp1.d(t)

            def f_d2(t: float):
                return interp2.d.d(t) - \
                    self.conditions.mu_d2(interp1(t)) * (interp1.d(t) ** 2) - \
                    self.conditions.mu_d(interp1(t)) * interp1.d.d(t)
            solver = HalleySolver(f, f_d, f_d2, t2)
        else:  # root_method == 'secant'
            solver = SecantSolver(f, t1, t2)

        while abs(t2 - t1) > self.step ** 4:
            t1, t2 = solver.step()
            xs.append(np.asarray([interp1(t2), interp2(t2),
                                  interp1.d(t2), interp2.d(t2)]))

        return solver.solution(), np.asarray(xs).T


class BouncingCGSolver:
    def __init__(self,
                 step: float = 1e-2,
                 ode_method: Literal['RK', 'Adams'] = 'Adams',
                 root_method: Literal['Secant', 'Newton', 'Halley'] = 'Secant',
                 conditions: Conditions | None = None):
        self.step = step
        self._conditions = None
        self._ode_method = ode_method
        self._root_method = root_method
        self.conditions = conditions

        self._solution: tuple[np.NDArray[float], np.NDArray[float]] = (
            np.empty((0,)), np.empty((4, 0))
        )
        # self._crossing: tuple[float, float, float] = None

    @property
    def conditions(self):
        return self._conditions

    @conditions.setter
    def conditions(self, new: Conditions):
        if self._conditions != new:
            self._conditions = new

    @property
    def solution(self):
        return self._solution

    def solve(self):
        if self._conditions is None:
            raise ValueError("No conditions given")

        cond = self.conditions
        counter = 0
        last_t, last_x = cond.t_0, np.asarray(cond.x_0)
        while True:
            solver = CrossingGroundSolver(
                self.step,
                self._ode_method,
                self._root_method,
                cond,
                ensure_start=True
            )
            if not solver.solve() or \
                    np.allclose(last_x[:2], solver.solution[1][:2, -1]) or \
                    counter == 1000:
                break
            counter += 1
            self._solution = (
                np.hstack((self._solution[0], solver.solution[0][:-1])),
                np.hstack((self._solution[1], solver.solution[1][:, :-1]))
                )

            last_t, last_x = solver.solution
            last_t, last_x = last_t[-1], last_x[:, -1]

            n = np.array([-self.conditions.mu_d(last_x[0]), 1])
            n = n / np.linalg.norm(n)
            v = np.array([last_x[2], last_x[3]])
            v = (v - 2 * np.dot(v, n) * n) * (1 - cond.b) ** 0.5
            cond.x_0 = [last_x[0], last_x[1],
                        v[0], v[1]]

            cond.t_0 = last_t


def main():
    cond = Conditions(
        x_0=[0, 10, 5, 2],
        m=4,
        k=3,
        mu=lambda t: np.sin(t),
        mu_d=lambda t: np.cos(t)
        )
    solver = BouncingCGSolver(0.01)
    solver.conditions = cond
    solver.solve()


if __name__ == '__main__':
    main()
