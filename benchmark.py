import os
from timeit import timeit, Timer

import numpy as np
import matplotlib.pyplot as plt

from crossing_ground_solver import CrossingGroundSolver, Conditions
# from solvers.ivp import RKSolver, AdamsSolver
# from solvers.root_finding import SecantSolver, NewtonSolver
# from solvers.interpolation import HermiteInterpolation

os.environ['TCL_LIBRARY'] = \
    r'C:\Users\jubilant\AppData\Local\Programs\Python\Python313/tcl/tcl8.6'


def secant_vs_newton():
    cond = Conditions(
        x_0=[0, 10, 5, 2],
        m=4,
        k=3,
        mu=lambda t: np.sin(t),
        mu_d=lambda t: np.cos(t)
        )

    solver = CrossingGroundSolver(0.05, root_method='Secant')
    solver.conditions = cond
    solver.solve()
    ts, xs = solver.root_solution
    plt.plot(np.linalg.norm(xs - xs[:, -1, np.newaxis], axis=0),
             label='secant')

    solver = CrossingGroundSolver(0.05, root_method='Newton')
    solver.conditions = cond
    solver.solve()
    ts, xs = solver.root_solution
    plt.plot(np.linalg.norm(xs - xs[:, -1, np.newaxis], axis=0),
             label='newton')
    plt.yscale('log')
    plt.legend()

    plt.show()


def main(loops=1000):
    cond = Conditions(
        x_0=[0, 10, 5, 2],
        m=4,
        k=3,
        mu=lambda t: np.sin(t),
        mu_d=lambda t: np.cos(t)
        )

    solver = CrossingGroundSolver(
        0.05, ode_method='RK', root_method='Secant', conditions=cond)
    print(f"Benchmarking {solver}")
    time = timeit(solver.solve, number=loops)
    print(f"\t {loops} loops, {time / loops * 1e3} ms average")

    solver = CrossingGroundSolver(
        0.05, ode_method='Adams', root_method='Secant', conditions=cond)
    print(f"Benchmarking {solver}")
    time = timeit(solver.solve, number=loops)
    print(f"\t {loops} loops, {time / loops * 1e3} ms average")

    solver = CrossingGroundSolver(
        0.05, ode_method='RK', root_method='Newton', conditions=cond)
    print(f"Benchmarking {solver}")
    time = timeit(solver.solve, number=loops)
    print(f"\t {loops} loops, {time / loops * 1e3} ms average")

    solver = CrossingGroundSolver(
        0.05, ode_method='Adams', root_method='Newton', conditions=cond)
    print(f"Benchmarking {solver}")
    time = timeit(solver.solve, number=loops)
    print(f"\t {loops} loops, {time / loops * 1e3} ms average")


if __name__ == '__main__':
    main(1000)
    # secant_vs_newton()
