from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from crossing_ground_solver import (
    CrossingGroundSolver,
    Conditions as CGConditions
    )
from solvers.root_finding import SecantSolver
from solvers.interpolation import (
    LocalPolynomialInterpolator, InterpolatorBase
)


@dataclass
class Conditions:
    x_init: tuple[float, float]
    'Initial coordinates'
    x_cross: float
    'Crossing coordinates: x_2 = mu(x_cross) = mu(x_1)'
    vel_init: float
    'Initial velocity'
    m: float
    k: float
    mu: Callable[[float], float]
    mu_d: Callable[[float], float] | None = None,
    mu_d2: Callable[[float], float] | None = None
    g: float = 9.8


class FindAngleSolver:
    def __init__(self,
                 step: float = 1e-2,
                 conditions: Conditions | None = None,
                 local_estimator: Literal['Lagrange', 'Hermite'] = 'Lagrange',
                 **cg_solver_kwargs):
        self.step = step
        self._conditions = conditions
        self.local_estimator = local_estimator
        self._cg_kwargs = cg_solver_kwargs

        self._cg_evaluations: dict[float, tuple[float, float, float]] = {}
        'Dict for storing CG evaluations (angle: [t, x1, x2])'
        self._crossing_estimator: InterpolatorBase | None = None

        self.solution_est1 = []
        'Step by step progress of first estimation'
        self.solution_est2 = []
        'Step by step progress of second estimation'

    @property
    def conditions(self):
        return self._conditions

    @conditions.setter
    def conditions(self, new: Conditions):
        if self._conditions != new:
            self._conditions = new

    @property
    def solution(self):
        return self.solution_est2[-1]   \
            if self.solution_est2       \
            else self.solution_est1[-1]

    def solve(self):
        if not self._crossing_estimator:
            self._build_estimator()

        self._solve_est1()
        self._solve_est2()

        return True

    def _solve_est1(self):
        def f(t: float):
            return self._crossing_estimator(t) - self.conditions.x_cross

        nodes = sorted(self._cg_evaluations.keys())
        t1, t2 = -np.pi/2, np.pi/2
        for i in range(1, len(nodes)):
            if self.conditions.x_cross < self._cg_evaluations[nodes[i]][1]:
                t1, t2 = nodes[i-1], nodes[i]
                break

        solver = SecantSolver(f, t1, t2)
        self.solution_est1 = [t1, t2]

        while abs(t2 - t1) > self.step ** 2:
            t1, t2 = solver.step()
            self.solution_est1.append(t2)

    def _solve_est2(self):
        def f(t: float):
            cg_solver = self._cg_from_angle(t)
            cg_solver.solve()
            return cg_solver.crossing[1] - self.conditions.x_cross

        t1, t2 = self.solution_est1[-2], self.solution_est1[-1]
        solver = SecantSolver(f, t1, t2)
        self.solution_est2 = [t1, t2]

        while abs(t2 - t1) > self.step ** 4:
            t1, t2 = solver.step()
            self.solution_est2.append(t2)

    def _build_estimator(self):
        angles = np.linspace(-np.pi/2, np.pi/2, 16, endpoint=True)
        crossings = np.empty((angles.shape[0], 3))
        for i, angle in enumerate(angles):
            cgsolver = self._cg_from_angle(angle)
            cgsolver.solve()
            crossings[i] = np.asarray(cgsolver.crossing)
            self._cg_evaluations[angle] = cgsolver.crossing
        self._crossing_estimator = LocalPolynomialInterpolator.from_points(
            angles, crossings[:, 1], self.local_estimator
        )

    def _cg_from_angle(self, angle: float):
        return CrossingGroundSolver(
                self.step,
                conditions=CGConditions(
                    x_0=[self.conditions.x_init[0], self.conditions.x_init[1],
                         self.conditions.vel_init * np.cos(angle),
                         self.conditions.vel_init * np.sin(angle)],
                    m=self.conditions.m,
                    k=self.conditions.k,
                    mu=self.conditions.mu,
                    mu_d=self.conditions.mu_d,
                    mu_d2=self.conditions.mu_d2,
                    g=self.conditions.g
                ),
                **self._cg_kwargs)


def main():
    cond = Conditions(
        x_init=[0, 10],
        x_cross=4.2,
        vel_init=5,
        m=4,
        k=3,
        mu=lambda t: np.sin(t)
        )
    solver = FindAngleSolver()
    solver.conditions = cond
    solver.solve()


if __name__ == '__main__':
    main()
