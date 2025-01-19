from typing import Literal, Callable

import numpy as np
import scipy.sparse as ssparse
import scipy.sparse.linalg


class CG:
    def __init__(self, preconditioner: Literal['identity', 'jacobi', 'ilu'] | Callable | None = None):
        self.preconditioner = preconditioner

    def solve(self, A: ssparse.sparray, b: np.ndarray, x_init: np.ndarray | None = None,
              atol: float = 0, rtol: float = 1e-5, max_it: int | None = None) -> tuple[np.ndarray, int]:
        """
        Use Conjugate Gradient iteration to solve ``Ax = b``.

        Parameters
        ----------
        A : sparse array | ndarray
            The real or complex N-by-N matrix of the linear system.
            `A` must represent a hermitian, positive definite matrix.
        b : ndarray
            Right hand side of the linear system. Has shape (N,) or (N,1).
        x_init : ndarray
            Starting guess for the solution.
        rtol, atol : float, optional
            Parameters for the convergence test. For convergence,
            ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
            The default is ``atol=0.`` and ``rtol=1e-5``.
        maxiter : int, optional
            Maximum number of iterations.  Iteration will stop after maxiter
            steps even if the specified tolerance has not been achieved.

        Returns
        -------
        x : ndarray
            The converged solution.
        info : integer
            Provides convergence information:
                0  : successful exit
                >0 : convergence to tolerance not achieved, number of iterations
        """
        A: ssparse.csr_array = A.tocsr()
        out_shape = x_init.shape if x_init else b.shape
        b = b.reshape(-1, 1)

        B = self._resolve_preconditioner(A)

        tol = max(rtol * np.linalg.norm(b), atol)
        
        u = x_init or np.zeros_like(b)
        r: np.ndarray = b - A.dot(u)
        r_B: np.ndarray = B(r)
        p: np.ndarray = r_B.copy()
        rho: float = r.T.dot(r_B)
        a: np.ndarray
        alpha: float
        beta: float

        it = 0
        while not max_it or it < max_it:
            a = A.dot(p)
            alpha = rho / p.T.dot(a)
            u += alpha * p
            r += -alpha * a
            if np.linalg.norm(r) <= tol:
                converged = True
                break

            r_B = B(r)
            rho_ = rho
            rho = r.T.dot(r_B)
            beta = rho / rho_
            p = beta * p + r_B

            it += 1

        self.iterations = it
        return u.reshape(out_shape), 0 if converged else it

    def _resolve_preconditioner(self, A: ssparse.csr_array) -> Callable[[np.ndarray], np.ndarray]:
        match self.preconditioner:
            case 'jacobi':
                diag = A.diagonal()
                B = ssparse.dia_array((1 / diag, [0]), shape=(len(diag), len(diag)))
                self.preconditioner = lambda v: B.dot(v)
            case 'ilu':
                B = scipy.sparse.linalg.spilu(A.tocsc(copy=False), drop_tol=0.2, fill_factor=1.5)
                self.preconditioner = lambda v: B.solve(v)
            case 'identity' | None:
                B = ssparse.eye_array(A.shape[0])
                self.preconditioner = lambda v: B.dot(v)
            case _:
                raise ValueError('Unknown preconditioner')

        return self.preconditioner
