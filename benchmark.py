from timeit import timeit

import numpy as np

from crossing_ground_solver import CrossingGroundSolver, Conditions
# from solvers.ivp import RKSolver, AdamsSolver
# from solvers.root_finding import SecantSolver, NewtonSolver
# from solvers.interpolation import HermiteInterpolation


def main(loops=1000, step=0.05):
    cond = Conditions(
        x_0=[0, 10, 5, 2],
        m=4,
        k=3,
        mu=lambda t: np.sin(t),
        mu_d=lambda t: np.cos(t)
        )

    solver = CrossingGroundSolver(
        step, ode_method='RK', root_method='Secant', conditions=cond)
    print(f"Benchmarking {solver}")
    time = timeit(solver.solve, number=loops)
    print(f"\t {loops} loops, {time / loops * 1e3} ms average")

    solver = CrossingGroundSolver(
        step, ode_method='Adams', root_method='Secant', conditions=cond)
    print(f"Benchmarking {solver}")
    time = timeit(solver.solve, number=loops)
    print(f"\t {loops} loops, {time / loops * 1e3} ms average")

    solver = CrossingGroundSolver(
        step, ode_method='RK', root_method='Newton', conditions=cond)
    print(f"Benchmarking {solver}")
    time = timeit(solver.solve, number=loops)
    print(f"\t {loops} loops, {time / loops * 1e3} ms average")

    solver = CrossingGroundSolver(
        step, ode_method='Adams', root_method='Newton', conditions=cond)
    print(f"Benchmarking {solver}")
    time = timeit(solver.solve, number=loops)
    print(f"\t {loops} loops, {time / loops * 1e3} ms average")


if __name__ == '__main__':
    main(1000, step=0.05)
