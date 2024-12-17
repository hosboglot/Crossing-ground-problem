from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from solvers.ivp import RKSolver, AdamsSolver
from solvers.root_finding import SecantSolver, NewtonSolver
from solvers.interpolation import HermiteInterpolator


@dataclass
class Conditions:
    x_0: tuple[float, float, float, float]
    m: float
    k: float
    mu: Callable[[float], float]
    mu_d: Callable[[float], float] | None = None
    g: float = 9.8


class CrossingGroundSolver:
    def __init__(self,
                 step: float = 1e-2,
                 ode_method: Literal['RK', 'Adams'] = 'Adams',
                 root_method: Literal['Secant', 'Newton'] = 'Newton',
                 conditions: Conditions | None = None):
        self.step = step
        self._conditions = None
        self._ode_method = ode_method
        self._root_method = root_method
        self.conditions = conditions

        self._ode_solution: tuple[np.NDArray[float], np.NDArray[float]] = None
        self._root_solution: tuple[np.NDArray[float], np.NDArray[float]] = None
        self._crossing: tuple[float, float, float] = None

    def __repr__(self) -> str:
        return "CGSolver(" \
              f"step='{self.step}', " \
              f"ode_method='{self._ode_method}', " \
              f"root_method='{self._root_method}'" \
               ")"

    @property
    def conditions(self):
        return self._conditions

    @conditions.setter
    def conditions(self, new: Conditions):
        if self._conditions != new:
            self._conditions = new

    @property
    def ode_solution(self):
        '''Returns ts, xs'''
        return self._ode_solution

    @property
    def root_solution(self):
        '''Returns ts, xs'''
        return self._root_solution

    @property
    def crossing(self) -> tuple[float, float, float]:
        '''Returns t, x1, x2'''
        return self._crossing

    def solve(self):
        if self._conditions is None:
            raise ValueError("No conditions given")

        self._ode_solution = self._solve_ode()
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

        t_last, x_last = 0, np.asarray(self.conditions.x_0)
        match self._ode_method.lower():
            case 'rk':
                solver = RKSolver(rhs, t_last, x_last, self.step)
            case 'adams':
                solver = AdamsSolver(rhs, t_last, x_last, self.step)

        while self.conditions.mu(x_last[0]) <= x_last[1]:
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
        xs = [np.asarray([xs[0, 0], xs[1, 0]]),
              np.asarray([xs[0, 1], xs[1, 1]])]

        def f(t: float):
            return interp2(t) - self.conditions.mu(interp1(t))

        match self._root_method.lower(), self.conditions.mu_d:
            case 'secant', _:
                solver = SecantSolver(f, t1, t2)
            case 'newton', mu_d if mu_d is not None:
                def f_d(t: float):
                    return interp2.d(t) - mu_d(interp1(t)) * interp1.d(t)
                solver = NewtonSolver(f, f_d, t2)
            case _:
                solver = SecantSolver(f, t1, t2)

        while abs(t2 - t1) > self.step ** 4:
            t1, t2 = solver.step()
            xs.append(np.asarray([interp1(t2), interp2(t2)]))

        return solver.solution(), np.asarray(xs).T


def main():
    cond = Conditions(
        x_0=[0, 10, 5, 2],
        m=4,
        k=3,
        mu=lambda t: np.sin(t)
        )
    solver = CrossingGroundSolver()
    solver.conditions = cond
    solver.solve()


if __name__ == '__main__':
    main()
