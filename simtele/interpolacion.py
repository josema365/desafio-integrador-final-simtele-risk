"""Interpolación: Lagrange, Newton (diferencias divididas) y spline cúbico natural."""
import numpy as np
from .lineal import lu_solve


def lagrange_eval(xn, yn, x):
    xn = np.asarray(xn, float); yn = np.asarray(yn, float); x = np.asarray(x, float)
    total = np.zeros_like(x, dtype=float)
    for j in range(len(xn)):
        t = np.full_like(x, yn[j], dtype=float)
        for i in range(len(xn)):
            if i != j:
                t = t * (x - xn[i]) / (xn[j] - xn[i])
        total += t
    return total


def newton_coef(xn, yn):
    """Coeficientes c_k = f[x_0, …, x_k] de la tabla de diferencias divididas."""
    xn = np.asarray(xn, float)
    c = np.array(yn, float)
    n = len(xn)
    for k in range(1, n):
        c[k:] = (c[k:] - c[k - 1:-1]) / (xn[k:] - xn[:n - k])
    return c


def newton_eval(xn, c, x):
    x = np.asarray(x, float)
    p = np.full_like(x, c[-1], dtype=float)
    for k in range(len(c) - 2, -1, -1):
        p = p * (x - xn[k]) + c[k]
    return p


def spline_natural(xn, yn):
    """Spline cúbico natural (S''=0 en los extremos). Devuelve una función evaluable S(x).
    El sistema tridiagonal simétrico definido positivo para las segundas derivadas se resuelve con LU propio."""
    xn = np.asarray(xn, float); yn = np.asarray(yn, float)
    n = len(xn) - 1
    if n < 2:
        raise ValueError("El spline necesita al menos 3 nodos.")
    h = np.diff(xn)
    M = np.zeros(n + 1)
    if n >= 2:
        A = np.zeros((n - 1, n - 1)); rhs = np.zeros(n - 1)
        for i in range(1, n):
            A[i - 1, i - 1] = 2 * (h[i - 1] + h[i])
            if i > 1: A[i - 1, i - 2] = h[i - 1]
            if i < n - 1: A[i - 1, i] = h[i]
            rhs[i - 1] = 6 * ((yn[i + 1] - yn[i]) / h[i] - (yn[i] - yn[i - 1]) / h[i - 1])
        M[1:n] = lu_solve(A, rhs)[0]

    def S(x):
        x = np.asarray(x, float)
        i = np.clip(np.searchsorted(xn, x, side="right") - 1, 0, n - 1)
        dx0 = x - xn[i]; dx1 = xn[i + 1] - x; hi = h[i]
        return (M[i] * dx1 ** 3 / (6 * hi) + M[i + 1] * dx0 ** 3 / (6 * hi)
                + (yn[i] / hi - M[i] * hi / 6) * dx1 + (yn[i + 1] / hi - M[i + 1] * hi / 6) * dx0)
    return S


def lebesgue(xn, npts=4001):
    """Constante de Lebesgue Λ_n = max_x Σ_j |l_j(x)|: amplificación máxima de perturbaciones en los datos."""
    xn = np.asarray(xn, float)
    t = np.linspace(xn.min(), xn.max(), npts)
    L = np.zeros_like(t)
    for j in range(len(xn)):
        lj = np.ones_like(t)
        for i in range(len(xn)):
            if i != j:
                lj = lj * (t - xn[i]) / (xn[j] - xn[i])
        L += np.abs(lj)
    return float(L.max())


def nodos_chebyshev(a, b, n):
    k = np.arange(n)
    return (a + b) / 2 - (b - a) / 2 * np.cos(np.pi * (2 * k + 1) / (2 * n))
