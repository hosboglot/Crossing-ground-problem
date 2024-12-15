import numpy as np
import matplotlib.pyplot as plt
import os

from ivp import solveRK, solveAdams


os.environ['TCL_LIBRARY'] = \
    r'C:\Users\jubilant\AppData\Local\Programs\Python\Python313/tcl/tcl8.6'


def test_runge_kutta():
    '''
    System                  \\
    x''(t) + w_0^2 x(t) = 0 \\
    ->                      \\
    x'(t) = x_1(t)          \\
    x_1'(t) = -w_0^2 x(t)
    '''
    w_0 = 0.1
    x_0 = [1, 3]

    def precise_solution(t):
        return x_0[1] * np.sin(w_0 * t) / w_0 + \
                x_0[0] * np.cos(w_0 * t)

    def rhs(t, x):
        return np.asarray((x[1], -w_0**2 * x[0]))

    def stop_condition(t, x):
        return t >= 1 or np.allclose(t, 1)

    t1, x1 = solveRK(rhs, 0, x_0, stop_condition, step=1e-1)
    t2, x2 = solveRK(rhs, 0, x_0, stop_condition, step=5e-2)

    c1 = np.linalg.norm(x1[:, -1] - x2[:, -1]) / 1e-4

    t1, x1 = solveRK(rhs, 0, x_0, stop_condition, step=1e-2)
    t2, x2 = solveRK(rhs, 0, x_0, stop_condition, step=5e-3)

    c2 = np.linalg.norm(x2[:, -1] - x1[:, -1]) / 1e-8
    # plt.plot(x1[0], x1[1], x2[0], x2[1])

    c0 = np.linalg.norm(precise_solution(t2) - x2[0])

    print(c0, c1, c2)
    # plt.show()


def test_adams():
    '''
    System                  \\
    x''(t) + w_0^2 x(t) = 0 \\
    ->                      \\
    x'(t) = x_1(t)          \\
    x_1'(t) = -w_0^2 x(t)
    '''
    w_0 = 0.1
    x_0 = [1, 3]

    def precise_solution(t):
        return x_0[1] * np.sin(w_0 * t) / w_0 + \
                x_0[0] * np.cos(w_0 * t)

    def rhs(t, x):
        return np.asarray((x[1], -w_0**2 * x[0]))

    def stop_condition(t, x):
        return t >= 1 or np.allclose(t, 1)

    t1, x1 = solveAdams(rhs, 0, x_0, stop_condition, step=1e-1)
    t2, x2 = solveAdams(rhs, 0, x_0, stop_condition, step=5e-2)

    c1 = np.linalg.norm(x1[:, -1] - x2[:, -1]) / 1e-4

    t1, x1 = solveAdams(rhs, 0, x_0, stop_condition, step=1e-2)
    t2, x2 = solveAdams(rhs, 0, x_0, stop_condition, step=5e-3)

    c2 = np.linalg.norm(x2[:, -1] - x1[:, -1]) / 1e-8
    # plt.plot(x1[0], x1[1], x2[0], x2[1])

    c0 = np.linalg.norm(precise_solution(t2) - x2[0])

    print(c0, c1, c2)
    # plt.show()


if __name__ == '__main__':
    print('Testing Runge-Kutta')
    test_runge_kutta()
    print('Testing Adams')
    test_adams()
