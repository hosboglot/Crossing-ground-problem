from dataclasses import dataclass
from typing import Callable, Literal
from copy import copy

import numpy as np
import scipy.sparse as ssparse
import scipy.sparse.linalg

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


class FEM:
    def __init__(self, conditions: Conditions, n: int = 10):
        self.conditions = conditions
        self.n = n

    def build(self):
        self.build_mesh()
        self.build_system()

        return self.A, self.F

    def build_mesh(self):
        self.mesh, self.step = np.linspace(
            self.conditions.x_0,
            self.conditions.x_n,
            self.n + 1,
            endpoint=True,
            retstep=True
        )

    def build_system(self):
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

        return self.A, self.F

    def _build_boundaries(self, A: ssparse.dok_array, F: np.ndarray):
        # left boundary Neumann conditions
        if isinstance(self.conditions.left, (tuple, list)):
            A[0, 0] += self.conditions.left[0]
            F[0] += self.conditions.left[1]
        # Dirichlet conditions
        else:
            F = F - self.conditions.left * A[:, 0].toarray()
            A[0, :] = 0
            A[:, 0] = 0
            A[0, 0] = 1
            F[0] = self.conditions.left

        # right boundary Neumann conditions
        if isinstance(self.conditions.right, (tuple, list)):
            A[-1, -1] += self.conditions.right[0]
            F[-1] += self.conditions.right[1]
        # Dirichlet conditions
        else:
            F = F - self.conditions.right * A[:, -1].toarray()
            A[-1, :] = 0
            A[:, -1] = 0
            A[-1, -1] = 1
            F[-1] = self.conditions.right

        return A, F

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

class FEMSolver(FEM):
    def __init__(self, conditions: Conditions, n: int = 10,
                 slae_solver: CG | None = None):
        super().__init__(conditions, n)
        self.slae_solver = slae_solver or CG()

    def solve(self):
        self.A, self.F = self.build()

        self.solution, success = self._solve_slae()
        if not success:
            print('No converge')
            # raise RuntimeError("Method did not converge")

    def _solve_slae(self):
        # solution = scipy.sparse.linalg.spsolve(self.A, self.F)
        solution, exit_code = self.slae_solver.solve(self.A, self.F)
        return solution.flatten(), not exit_code


class MultigridFEMSolver(FEMSolver):
    def __init__(self, conditions: Conditions, n: int = 10,
                 slae_solver: CG | None = None,
                 n_layers: int = 2, cycle: Literal['V', 'F', 'W'] = 'V',
                 presmoother: preconditioners.PreconditionerBase | None = None,
                 postsmoother: preconditioners.PreconditionerBase | None = None):
        super().__init__(conditions, n, slae_solver)
        self.n_layers = n_layers
        self.cycle = cycle
        self.presmoother  = presmoother  or preconditioners.Identity()
        self.postsmoother = postsmoother or preconditioners.Identity()

    def solve(self):
        # # n should divide by 2^n_layers - 1
        # n_mod = self.n % (2**(self.n_layers - 1))
        # if n_mod != 0:
        #     self.n += (2**(self.n_layers - 1)) - n_mod
        #     print('multigrid: number of elements corrected to', self.n)

        match self.cycle.lower():
            case 'v':
                self.slae_solver.preconditioner = MultigridFEMSolver.VCycle(
                    self, self.n_layers, self.presmoother, self.postsmoother
                )
            # case 'f':
            #     self._solve_f_cycle()
            # case 'w':
            #     self._solve_w_cycle()
            case _:
                raise ValueError("Unknown cycle type")
        super().solve()

    class StandaloneSolver:
        def __init__(self, preconditioner: preconditioners.PreconditionerBase | None = None,
                     atol: float = 0, rtol: float = 1e-5, max_it: int | None = None, verbose=False):
            self.preconditioner: preconditioners.PreconditionerBase = preconditioner or preconditioners.Identity()
            self.atol = atol
            self.rtol = rtol
            self.max_it = max_it
            self.verbose = verbose

        def solve(self, A: ssparse.sparray, b: np.ndarray, x_init: np.ndarray | None = None) -> tuple[np.ndarray, int]:
            A: ssparse.csr_array = A.tocsr()
            out_shape = x_init.shape if x_init else b.shape
            b = b.reshape(-1, 1)
            tol = max(self.rtol * np.linalg.norm(b), self.atol)
            
            u = x_init or np.zeros_like(b)
            self.preconditioner.init(A)

            it = 0
            converged = False
            while not self.max_it or it < self.max_it:
                r = b - A.dot(u)

                norm = np.linalg.norm(r)
                if self.verbose: print(norm)
                if norm <= tol:
                    converged = True
                    break

                u += self.preconditioner.solve(r)

                it += 1
            self.iterations = it
                
            return u.reshape(out_shape), 0 if converged else it

    class VCycle(preconditioners.PreconditionerBase):
        def __init__(self, fem: FEM, n_layers: int,
                     presmoother: preconditioners.PreconditionerBase,
                     postsmoother: preconditioners.PreconditionerBase):
            self.n_layers = n_layers
            self._layers = [fem]
            self._presmoothers  = [copy(presmoother)  for _ in range(n_layers)]
            self._postsmoothers = [copy(postsmoother) for _ in range(n_layers)]

        def init(self, A: ssparse.csc_array):
            # build meshes
            for _ in range(1, self.n_layers):
                self._layers.append(FEM(
                    conditions=self._layers[-1].conditions,
                    n=int((self._layers[-1].n + 1) // 2)
                ))
                self._layers[-1].build_mesh()

            # init restrictors
            self.R = []
            for layer in range(self.n_layers - 1):
                n1, n2 = self._layers[layer].n + 1, self._layers[layer + 1].n + 1
                R = ssparse.lil_array((n2, n1))
                R.rows[0] = [0, 1]
                R.data[0] = [3/4, 1/4]
                for i in range(1, n2 - 1):
                    idx = 2 * i
                    R.rows[i] = [idx - 1, idx, idx + 1]
                    R.data[i] = [1/4,     2/4, 1/4    ]
                R.rows[-1] = [n1 - 2, n1 - 1]
                R.data[-1] = [1/4, 3/4]
                self.R.append(R.tocsr())

            # init prolongators
            self.P = []
            for layer in range(self.n_layers - 1):
                self.P.append((2 * self.R[layer].T).tocsr())
                self.P[-1].data[0] = self.P[-1].data[-1] = 1
                if self.P[-1].shape[0] % 2 == 0:
                    self.P[-1].data[-4:] = [1, 0, 0, 1]

            # init smoothers
            for i in range(self.n_layers):
                self._presmoothers[i].init(self._layers[i].build_system()[0])
                self._postsmoothers[i].init(self._layers[i].build_system()[0])

        def solve(self, b: np.ndarray, x0: np.ndarray | None = None):
            if x0 is None: 
                y = np.zeros_like(b)
            else:
                y = x0.copy()

            # graphical watch
            # (b - self._layers[layer].A.dot(y)).flatten().tolist()
            # (self._layers[layer].A.dot(e)).flatten().tolist()
            # (self._prolongate(rhs, layer)).flatten().tolist()

            return self._solve_layer(y, b, 0)

        def _solve_layer(self, y: np.ndarray, b: np.ndarray, layer: int):
            # pre-smoothing
            y = self._presmoothers[layer].solve(b, x0=y)

            # calculate residual
            r = (b - self._layers[layer].A.dot(y))

            # restriction
            rhs = self._restrict(r, layer + 1)

            # solving
            if layer == len(self._layers) - 2:
                eps = scipy.sparse.linalg.spsolve(self._layers[-1].A, rhs).reshape(-1, 1)
            else:
                eps = self._solve_layer(np.zeros_like(rhs), rhs, layer + 1)

            # prolongation and correction
            e = self._prolongate(eps, layer)
            y += e

            # post-smoothing
            y = self._postsmoothers[layer].solve(b, x0=y)
            
            return y

        def _restrict(self, v: np.ndarray, on_layer: int):
            return self.R[on_layer - 1].dot(v)

        def _prolongate(self, v: np.ndarray, on_layer: int):
            return self.P[on_layer].dot(v)


def test_multigrid(n1: float = 100, n2: float = 200):
    solver = MultigridFEMSolver(
        conditions=Conditions(
            a=lambda x: np.exp(2*x),
            b=lambda x: 0,
            f=lambda x: 4 * np.sin(3*x) + np.sin(x) * np.cos(2*x),
            x_0=0,
            x_n=1,
            left=[1, -2],
            right=[1, -3]
        ),
        slae_solver=MultigridFEMSolver.StandaloneSolver(max_it=5000, verbose=False),
        # slae_solver=CG(max_it=5000, verbose=False),
        n_layers=2,
        presmoother=preconditioners.SSOR(4, 1),
        postsmoother=preconditioners.SSOR(4, 1),
    )

    solver.n = n1
    solver.solve()
    y1 = solver.solution
    solver.n = n1 * 2
    solver.solve()
    y2 = solver.solution[::2]

    print(np.linalg.norm(y1 - y2, np.inf) * n1**2)
    print(solver.slae_solver.iterations)

    solver.n_layers += 1
    solver.n = n2
    solver.solve()
    y1 = solver.solution
    solver.n = n2 * 2
    solver.solve()
    y2 = solver.solution[::2]

    print(np.linalg.norm(y1 - y2, np.inf) * n2**2)
    print(solver.slae_solver.iterations)

def test(n1: int = 100, n2: int = 200):
    solver = FEMSolver(
        conditions=Conditions(
            a=lambda x: 1,
            b=lambda x: 0,
            f=lambda x: 4 * np.sin(3*x) + np.sin(x) * np.cos(2*x),
            x_0=0,
            x_n=1,
            left=[1, -2],
            right=[1, -3]
        ),
        slae_solver=CG(
            # preconditioners.Identity(),
            preconditioners.Jacobi(4, 1),
            max_it=2000
        )
    )

    solver.n = n1
    solver.solve()
    y1 = solver.solution
    solver.n = n1 * 2
    solver.solve()
    y2 = solver.solution[::2]

    print(np.linalg.norm(y1 - y2, np.inf) * n1**2)
    print(np.argmax(abs(y1 - y2)))

    solver.n = n2
    solver.solve()
    y1 = solver.solution
    solver.n = n2 * 2
    solver.solve()
    y2 = solver.solution[::2]

    print(np.linalg.norm(y1 - y2, np.inf) * n2**2)
    print(np.argmax(abs(y1 - y2)))

def time_multigrid(n: int = 1000, loops: int = 10):
    cond = Conditions(
        a=lambda x: np.exp(2*x),
        b=lambda x: 0,
        f=lambda x: 4 * np.sin(3*x) + np.sin(x) * np.cos(2*x),
        x_0=0,
        x_n=1,
        left=[1, -2],
        right=[1, -3]
    )
    default = FEMSolver(
        conditions=cond, n=n,
        slae_solver=CG(
            preconditioners.Jacobi(1, 1),
            max_it=10 * n, verbose=False
        )
    )
    multigrid = MultigridFEMSolver(
        conditions=cond, n=n,
        slae_solver=MultigridFEMSolver.StandaloneSolver(max_it=10 * n, verbose=False),
        # slae_solver=CG(max_it=10 * n, verbose=False),
        n_layers=2,
        presmoother=preconditioners.Jacobi(2, 1),
        postsmoother=preconditioners.SSOR(1, 1),
    )

    from timeit import timeit
    from tqdm import tqdm

    with tqdm(range(loops), 'default') as progress:
        def progress_func():
            default.solve()
            progress.update()
            progress.set_description(f'{default.slae_solver.iterations} its')
        time = timeit(progress_func, number=loops)
    print(f"\t {loops} loops, {time / loops * 1e3} ms average")

    with tqdm(range(loops), 'multigrid') as progress:
        def progress_func():
            multigrid.solve()
            progress.update()
            progress.set_description(f'{multigrid.slae_solver.iterations} its')
        time = timeit(progress_func, number=loops)
    print(f"\t {loops} loops, {time / loops * 1e3} ms average")

def main_multigrid():
    solver = MultigridFEMSolver(
        conditions=Conditions(
            a=lambda x: np.exp(2*x),
            b=lambda x: 0,
            f=lambda x: 4 * np.sin(3*x) + np.sin(x) * np.cos(2*x),
            x_0=0,
            x_n=1,
            left=[1, -2],
            right=[1, -3]
        ),
        n=1000,
        slae_solver=MultigridFEMSolver.StandaloneSolver(max_it=5000, verbose=False),
        # slae_solver=CG(max_it=5000, verbose=False),
        n_layers=2,
        presmoother=preconditioners.SSOR(4, 1),
        postsmoother=preconditioners.SSOR(4, 1),
    )
    solver.solve()

    print(solver.slae_solver.iterations)

def main():
    exact_func = lambda x: np.sin(x) * np.cos(2*x)

    solver = FEMSolver(
        conditions=Conditions(
            a=lambda x: 1,
            b=lambda x: 0,
            f=lambda x: 4 * np.sin(3*x) + np.sin(x) * np.cos(2*x),
            x_0=0,
            x_n=1,
            left=[1, -1],
            right=exact_func(1)
        ),
        n=100,
        slae_solver=CG(
            # preconditioners.Identity(),
            preconditioners.Jacobi(4, 1),
            max_it=2000
        )
    )
    solver.solve()

    print(solver.slae_solver.iterations)


if __name__ == "__main__":
    # main()
    main_multigrid()
    # test()
    # test_multigrid()
    # time_multigrid()