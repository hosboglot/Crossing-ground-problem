from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from solvers.ivp import RKSolver, AdamsSolver


@dataclass
class Conditions:
    x_0: np.ndarray[float]
    v_0: np.ndarray[float]
    t_end: float
    'Time until stop evaluations'
    w: np.ndarray[float]
    'Self frequencies of springs, n values'
    k: np.ndarray[float]
    'Coefficients between i and i+1 springs, n-1 values'
    t_0: float = 0


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
        return self._solution[0][:self.conditions.x_0.shape[0]], self._solution[1][:self.conditions.x_0.shape[0]]

    @property
    def velocities(self):
        return self._solution[0][self.conditions.x_0.shape[0]:], self._solution[1][self.conditions.x_0.shape[0]:]

    def solve(self):
        if self._conditions is None:
            raise ValueError('Conditions are not set')

        self._solution = self._solve_ode()
        
    def _solve_ode(self):
        def rhs(t: float, x: np.ndarray):
            n = self.conditions.x_0.shape[0]
            rhs_ = np.asarray(x[n:])
            left_spring_a = -(self.conditions.k[0] + self.conditions.w[0]**2) * x[0] + self.conditions.k[0] * x[1]
            right_spring_a = -(self.conditions.k[-1] + self.conditions.w[-1]**2) * x[n-1] + self.conditions.k[-1] * x[n-2]
            spring_a = []
            for i in range(1, n-1):
                spring_a.append(
                    -(self.conditions.k[i-1] + self.conditions.k[i] + self.conditions.w[i]**2) * x[i] + \
                        self.conditions.k[i-1] * x[i-1] + self.conditions.k[i] * x[i+1]
                    )
            rhs_ = np.hstack((rhs_, left_spring_a, spring_a, right_spring_a))
            return rhs_

        t_last, x_last = self.conditions.t_0, np.hstack((self.conditions.x_0, self.conditions.v_0))
        match self.ode_method.lower():
            case 'rk':
                solver = RKSolver(rhs, t_last, x_last, self.step)
            case 'adams':
                solver = AdamsSolver(rhs, t_last, x_last, self.step)

        while t_last < self.conditions.t_end:
            t_last, x_last = solver.step()

        return solver.solution()

def main():
    n = 10
    cond = Conditions(
        x_0=np.hstack((1, np.zeros(n-1))),
        v_0=np.zeros(n),
        t_0=0,
        t_end=10,
        w=np.ones(n),
        k=np.ones(n-1)
        )
    solver = BoundSpringsSolver(0.01, conditions=cond)
    solver.solve()


if __name__ == '__main__':
    main()

