from typing import Literal, Callable, Union

import numpy as np
import scipy.linalg
import scipy.sparse as ssparse
import scipy.sparse.linalg


class CG:
    def __init__(self, preconditioner: Union['preconditioners.PreconditionerBase', None] = None,
                 atol: float = 0, rtol: float = 1e-5, max_it: int | None = None, verbose=False):
        """
        Use Conjugate Gradient iteration to solve ``Ax = b``.

        Parameters
        ----------
        rtol, atol : float, optional
            Parameters for the convergence test. For convergence,
            ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
            The default is ``atol=0.`` and ``rtol=1e-5``.
        maxiter : int, optional
            Maximum number of iterations.  Iteration will stop after maxiter
            steps even if the specified tolerance has not been achieved.
        """
        self.preconditioner: preconditioners.PreconditionerBase = preconditioner or preconditioners.Identity()
        self.atol = atol
        self.rtol = rtol
        self.max_it = max_it
        self.verbose = verbose

    def solve(self, A: ssparse.sparray, b: np.ndarray, x_init: np.ndarray | None = None) -> tuple[np.ndarray, int]:
        A: ssparse.csr_array = A.tocsr()
        out_shape = x_init.shape if x_init else b.shape
        b = b.reshape(-1, 1)

        self.preconditioner.init(A)

        tol = max(self.rtol * np.linalg.norm(b), self.atol)
        
        u = x_init or np.zeros_like(b)
        r: np.ndarray = b - A.dot(u)
        r_B: np.ndarray = self.preconditioner.solve(r)
        p: np.ndarray = r_B.copy()
        rho: float = r.T.dot(r_B)
        a: np.ndarray
        alpha: float
        beta: float

        it = 0
        converged = False
        while not self.max_it or it < self.max_it:
            a = A.dot(p)
            alpha = rho / p.T.dot(a)
            u += alpha * p
            r += -alpha * a
            
            norm = np.linalg.norm(r)
            if self.verbose: print(norm)
            if norm <= tol:
                converged = True
                break

            r_B = self.preconditioner.solve(r)
            rho_ = rho
            rho = r.T.dot(r_B)
            beta = rho / rho_
            p = beta * p + r_B

            it += 1

        self.iterations = it
        return u.reshape(out_shape), 0 if converged else it

class preconditioners:
    class PreconditionerBase:
        def __init__(self):
            raise NotImplementedError()
        def init(self, A: ssparse.sparray):
            raise NotImplementedError()
        def solve(self, b: np.ndarray, x0: np.ndarray | None = None):
            raise NotImplementedError()

    class Identity(PreconditionerBase):
        def __init__(self):
            pass
        def init(self, A: ssparse.sparray):
            pass
        def solve(self, b: np.ndarray, x0: np.ndarray | None = None):
            return b

    class Jacobi(PreconditionerBase):
        def __init__(self, n_iterations: int = 1, w: float = 1):
            self.n_iterations = n_iterations
            self.w = w
        def init(self, A: ssparse.sparray):
            diag = A.diagonal()
            self.A = A
            self.diag = ssparse.eye_array(len(diag)) * self.w / diag
            # ssparse.dia_array((self.w / diag, [0]), shape=(len(diag), len(diag)))
        def solve(self, b: np.ndarray, x0: np.ndarray | None = None):
            if self.n_iterations == 0:
                return b
            if x0 is None: 
                y = np.zeros_like(b)
            else:
                y = x0.copy()

            for _ in range(self.n_iterations):
                y += self.diag.dot(b - self.A.dot(y))
            return y

    class SOR(PreconditionerBase):
        def __init__(self, n_iterations: int = 1, w: float = 1):
            self.n_iterations = n_iterations
            self.w = w
        def init(self, A: ssparse.sparray):
            lower = ssparse.tril(A, k=-1)
            diag = ssparse.eye_array(A.shape[0]) * A.diagonal()
            self.A = A
            self.B: ssparse.csc_array = (lower + diag / self.w).tocsc()
        def solve(self, b: np.ndarray, x0: np.ndarray | None = None):
            if self.n_iterations == 0:
                return b
            if x0 is None: 
                y = np.zeros_like(b)
            else:
                y = x0.copy()

            for _ in range(self.n_iterations):
                y += scipy.sparse.linalg.spsolve_triangular(self.B, b - self.A.dot(y))
            return y

    class SSOR(PreconditionerBase):
        def __init__(self, n_iterations: int = 1, w: float = 1):
            self.n_iterations = n_iterations
            self.w = w
        def init(self, A: ssparse.sparray):
            lower = ssparse.tril(A, k=-1)
            diag = ssparse.eye_array(A.shape[0]) * A.diagonal()
            self.A = A
            self.B1 = (lower + diag / self.w).tocsc()
            self.B2 = (lower.T + diag / self.w).tocsc()
        def solve(self, b: np.ndarray, x0: np.ndarray | None = None):
            if self.n_iterations == 0:
                return b
            if x0 is None: 
                y = np.zeros_like(b)
            else:
                y = x0.copy()

            for _ in range(self.n_iterations):
                y += scipy.sparse.linalg.spsolve_triangular(self.B1, b - self.A.dot(y))
                y += scipy.sparse.linalg.spsolve_triangular(self.B2, b - self.A.dot(y), lower=False)
            return y


if __name__ == "__main__":
    solver = CG(
        preconditioner=preconditioners.SOR(1, 1),
        max_it=2000
    )

    size = 1000
    np.random.seed(7)
    main = (np.random.rand(size)) + 1
    side = np.random.rand(size-1)
    A = ssparse.diags([side, main, side], offsets=[-1, 0, 1])
    b = (np.random.rand(size) - 0.5) * 10
    solution, it = solver.solve(A, b)

    print(solver.iterations, it)
    # print(scipy.sparse.linalg.cg(A, b, M=solver.preconditioner.B)[1])
