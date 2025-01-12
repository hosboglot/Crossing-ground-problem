from timeit import timeit

import numpy as np

from crossing_ground_solver import CrossingGroundSolver, Conditions as CGConditions
from angle_from_crossing import FindAngleSolver, Conditions as FAConditions
# from solvers.ivp import RKSolver, AdamsSolver
# from solvers.root_finding import SecantSolver, NewtonSolver
# from solvers.interpolation import HermiteInterpolation


def crossing_ground(loops=1000, step=0.05):
    cond = CGConditions(
        x_0=[0, 10, 5, 2],
        m=4,
        k=3,
        mu=lambda t: np.sin(t),
        mu_d=lambda t: np.cos(t),
        mu_d2=lambda t: -np.sin(t)
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

    solver = CrossingGroundSolver(
        step, ode_method='RK', root_method='Halley', conditions=cond)
    print(f"Benchmarking {solver}")
    time = timeit(solver.solve, number=loops)
    print(f"\t {loops} loops, {time / loops * 1e3} ms average")

    solver = CrossingGroundSolver(
        step, ode_method='Adams', root_method='Halley', conditions=cond)
    print(f"Benchmarking {solver}")
    time = timeit(solver.solve, number=loops)
    print(f"\t {loops} loops, {time / loops * 1e3} ms average")


def angle_from_crossing(loops=1000, step=0.05):
    cond = FAConditions(
        x_init=[0, 10],
        x_cross=4.2,
        vel_init=5,
        m=4,
        k=3,
        mu=lambda t: np.sin(t)
        )

    solver = FindAngleSolver(
        step, local_estimator='Lagrange', conditions=cond)
    print(f"Benchmarking {solver}")
    time = timeit(solver.solve, number=loops)
    print(f"\t {loops} loops, {time / loops * 1e3} ms average")
    
    solver = FindAngleSolver(
        step, local_estimator='Hermite', conditions=cond)
    print(f"Benchmarking {solver}")
    time = timeit(solver.solve, number=loops)
    print(f"\t {loops} loops, {time / loops * 1e3} ms average")
    
    solver = FindAngleSolver(
        step, local_estimator=None, conditions=cond)
    print(f"Benchmarking {solver}")
    time = timeit(solver.solve, number=loops)
    print(f"\t {loops} loops, {time / loops * 1e3} ms average")


def main(loops=1000, step=0.05):
    # crossing_ground(loops, step)
    angle_from_crossing(loops, step)


if __name__ == '__main__':
    main(1000, step=0.05)
