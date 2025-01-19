from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import scipy.sparse as ssparse

from solvers.slae import CG


@dataclass
class Conditions:
    """
        Simple 1D ODE defined as                            \\
        -(a(x)u'(x))' + b(x)u(x) = f(x), x \\in (x_0, x_n)  \\
        u(x_0) = u_0                                        \\
        u(x_n) = u_n
    """
    a: Callable[[float], float]
    b: Callable[[float], float]
    f: Callable[[float], float]
    x_0: float
    x_n: float
    u_0: float
    u_n: float


class SimpleOdeFEM:
    def __init__(self, conditions: Conditions, n: int = 10,
                 preconditioner: Literal['identity', 'jacobi'] = 'jacobi'):
        self.conditions = conditions
        self.preconditioner = preconditioner
        self.n = n

    def solve(self):
        if self.conditions is None:
            raise ValueError("No conditions given")

        self._build_mesh()
        self._build_system()

        solution, success = self._solve_slae()
        if not success:
            raise RuntimeError("Method did not converge")
        self.solution = np.hstack((
            self.conditions.u_0,
            solution,
            self.conditions.u_n
        ))

    def _build_mesh(self):
        self.mesh, self.step = np.linspace(
            self.conditions.x_0,
            self.conditions.x_n,
            self.n + 1,
            endpoint=True,
            retstep=True
        )

    def _build_system(self):
        A = ssparse.dok_array((self.n+1, self.n+1), dtype=np.float64)
        F = np.zeros(self.n+1)

        for elem_n in range(self.n):
            # diagonal cell
            diag_elem = self._interp_in(elem_n)
            A[elem_n, elem_n] += diag_elem
            A[elem_n + 1, elem_n + 1] += diag_elem
            
            # side (non-diagonal) cell
            side_elem = self._interp_between(elem_n, elem_n + 1)
            A[elem_n, elem_n + 1] = A[elem_n + 1, elem_n] = side_elem

            f_elem = self._interp_in_F(elem_n)
            F[elem_n] += f_elem
            F[elem_n + 1] += f_elem

        F = F - self.conditions.u_0 * A[:, 0].toarray() - self.conditions.u_n * A[:, -1].toarray()
        A = A[1:-1, 1:-1]
        self.F: np.ndarray = F[1:-1].reshape(-1, 1)
        
        self.A: ssparse.csr_array = A.tocsr()

    def _solve_slae(self):
        self.slae_solver = CG(self.preconditioner)
        solution, exit_code = self.slae_solver.solve(self.A, self.F)
        return solution.flatten(), not exit_code

    # helpers
    def _interp_in(self, elem_n: int):
        res = 0
        x0 = self.mesh[elem_n]
        x1 = self.mesh[elem_n + 1]
        h = x1 - x0
        x_mid = (x1 + x0) / 2
        res += self.conditions.a(x_mid) / h
        res += 0.25 * h * self.conditions.b(x_mid)
        return res

    def _interp_between(self, node_n: int, node_n2: int):
        res = 0
        x0 = self.mesh[node_n]
        x1 = self.mesh[node_n2]
        h = x1 - x0
        x_mid = (x1 + x0) / 2
        res += -self.conditions.a(x_mid) / h
        res += 0.25 * h * self.conditions.b(x_mid)
        return res

    def _interp_in_F(self, elem_n: int):
        res = 0
        x0 = self.mesh[elem_n]
        x1 = self.mesh[elem_n + 1]
        h = x1 - x0
        x_mid = (x1 + x0) / 2
        res += 0.5 * h * self.conditions.f(x_mid)
        return res

def main():
    exact_func = lambda x: np.sin(x)
    cond = Conditions(
        a=lambda x: 1,
        b=lambda x: 2,
        f=lambda x: 3 * np.sin(x),
        x_0=0,
        x_n=1,
        u_0=exact_func(0),
        u_n=exact_func(1)
    )
    solver = SimpleOdeFEM(cond, 1000, 'jacobi')
    solver.solve()
    
    print(solver.slae_solver.iterations)
    print(solver.solution)


if __name__ == "__main__":
    main()