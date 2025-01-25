from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import scipy.sparse as ssparse

from solvers.slae import CG, preconditioners


@dataclass
class Conditions:
    """
        Simple 1D ODE defined as                            \\
        -(a(x)u'(x))' + b(x)u(x) = f(x), x \\in (x_0, x_n)  \\

        Left and right are boundary conditions                          \\
        If a float, then considered Dirichlet condition                 \\
        If a tuple of two floats, then considered Neumann               \\
        u(x_0) = left or left[0]*u(x_0) - a(x_0)*u'(x_0) = left[1]      \\
        u(x_n) = right or right[0]*u(x_n) + a(x_n)*u'(x_n) = right[1]   \\
    """
    a: Callable[[float], float]
    b: Callable[[float], float]
    f: Callable[[float], float]
    x_0: float
    x_n: float
    left: float | tuple[float, float]
    right: float | tuple[float, float]


class SimpleOdeFEM:
    def __init__(self, conditions: Conditions, n: int = 10,
                 slae_solver: CG | None = None):
        self.conditions = conditions
        self.slae_solver = slae_solver or CG()
        self.n = n

    def solve(self):
        if self.conditions is None:
            raise ValueError("No conditions given")

        self._build_mesh()
        self._build_system()

        solution, success = self._solve_slae()
        if not success:
            raise RuntimeError("Method did not converge")

        self.solution = solution
        if isinstance(self.conditions.left, float):
            self.solution = np.hstack((
                self.conditions.left,
                self.solution
            ))
        if isinstance(self.conditions.right, float):
            self.solution = np.hstack((
                self.solution,
                self.conditions.right
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

        A, F = self._build_boundaries(A, F)
        
        self.A: ssparse.csr_array = A.tocsr()
        self.F: np.ndarray = F.reshape(-1, 1)

    def _build_boundaries(self, A: ssparse.dok_array, F: np.ndarray):
        # left boundary Neumann conditions
        if isinstance(self.conditions.left, (tuple, list)):
            A[0, 0] += self.conditions.left[0]
            F[0] += self.conditions.left[1]
        # Dirichlet conditions
        else:
            F = F - self.conditions.left * A[:, 0].toarray()
            A = A[1:, 1:]
            F = F[1:]

        # right boundary Neumann conditions
        if isinstance(self.conditions.right, (tuple, list)):
            A[-1, -1] += self.conditions.right[0]
            F[-1] += self.conditions.right[1]
        # Dirichlet conditions
        else:
            F = F - self.conditions.right * A[:, -1].toarray()
            A = A[:-1, :-1]
            F = F[:-1]
            
        return A, F

    def _solve_slae(self):
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
    exact_func = lambda x: np.sin(x) * np.cos(2*x)

    solver = SimpleOdeFEM(
        conditions=Conditions(
            a=lambda x: 1,
            b=lambda x: 0,
            f=lambda x: 4 * np.sin(3*x) + np.sin(x) * np.cos(2*x),
            x_0=0,
            x_n=1,
            left=[1, 0],
            right=exact_func(1)
        ),
        n=100,
        slae_solver=CG(
            preconditioners.SOR(1, 1),
            max_it=200
        )
    )
    solver.solve()
    
    print(solver.slae_solver.iterations)
    print(solver.solution)


if __name__ == "__main__":
    main()