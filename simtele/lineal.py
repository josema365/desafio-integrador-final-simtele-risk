"""Sistemas lineales A x = b: Jacobi, Gauss-Seidel, LU (Doolittle) y gradiente conjugado.
Teoría de errores incluida: nunca se divide por cero (erel = None si ‖x*‖ ≈ 0)."""
import numpy as np

TAU = 1e-12


def _rec(k, x, A, b, xs):
    nb = np.linalg.norm(b)
    r = np.linalg.norm(b - A @ x)
    nxs = np.linalg.norm(xs)
    eabs = float(np.linalg.norm(x - xs))
    return dict(k=k, x=x.copy(),
                res=float(r / nb) if nb > TAU else float(r),     # residual relativo (absoluto si b ≈ 0)
                eabs=eabs,
                erel=float(eabs / nxs) if nxs > TAU else None)   # None: no definido (evita dividir por cero)


def es_spd(A, tol_sim=1e-10):
    """True si A es simétrica y definida positiva (intento de Cholesky)."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or not np.allclose(A, A.T, atol=tol_sim):
        return False
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def jacobi(A, b, xs, tol=1e-10, maxit=2000, x0=None):
    n = len(b); x = np.zeros(n) if x0 is None else x0.astype(float).copy()
    D = np.diag(A)
    if np.any(np.abs(D) < 1e-15):
        raise ValueError("Jacobi requiere diagonal sin ceros.")
    H = [_rec(0, x, A, b, xs)]
    for k in range(1, maxit + 1):
        x = (b - (A @ x - D * x)) / D
        H.append(_rec(k, x, A, b, xs))
        if H[-1]['res'] < tol: return H, True
        if not np.isfinite(H[-1]['res']) or H[-1]['res'] > 1e12: return H, False   # divergencia
    return H, False


def gauss_seidel(A, b, xs, tol=1e-10, maxit=2000, x0=None):
    n = len(b); x = np.zeros(n) if x0 is None else x0.astype(float).copy()
    if np.any(np.abs(np.diag(A)) < 1e-15):
        raise ValueError("Gauss-Seidel requiere diagonal sin ceros.")
    H = [_rec(0, x, A, b, xs)]
    for k in range(1, maxit + 1):
        for i in range(n):
            s = A[i, :i] @ x[:i] + A[i, i + 1:] @ x[i + 1:]
            x[i] = (b[i] - s) / A[i, i]
        H.append(_rec(k, x, A, b, xs))
        if H[-1]['res'] < tol: return H, True
        if not np.isfinite(H[-1]['res']) or H[-1]['res'] > 1e12: return H, False
    return H, False


def cg(A, b, xs, tol=1e-10, maxit=2000, x0=None):
    """Gradiente conjugado estándar: exige A simétrica definida positiva (se valida)."""
    if not es_spd(A):
        raise ValueError("El gradiente conjugado estándar requiere una matriz simétrica definida positiva (SPD).")
    n = len(b); x = np.zeros(n) if x0 is None else x0.astype(float).copy()
    r = b - A @ x; p = r.copy(); rs = r @ r
    H = [_rec(0, x, A, b, xs)]
    for k in range(1, maxit + 1):
        Ap = A @ p; alpha = rs / (p @ Ap)
        x = x + alpha * p; r = r - alpha * Ap; rs_new = r @ r
        H.append(_rec(k, x, A, b, xs))
        if H[-1]['res'] < tol: return H, True
        p = r + (rs_new / rs) * p; rs = rs_new
    return H, False


def lu_doolittle(A):
    """A = L U con diag(L)=1, sin pivoteo (válido si los menores principales son ≠ 0, p. ej. SPD)."""
    n = len(A); L = np.eye(n); U = A.astype(float).copy()
    for k in range(n - 1):
        if abs(U[k, k]) < 1e-15:
            raise ValueError("Pivote nulo: la factorización LU sin pivoteo no existe para esta matriz.")
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
            U[i, k] = 0.0
    if abs(U[n - 1, n - 1]) < 1e-15:
        raise ValueError("Matriz singular (pivote nulo).")
    return L, U


def lu_solve(A, b):
    L, U = lu_doolittle(A); n = len(b)
    y = np.zeros(n)
    for i in range(n): y[i] = b[i] - L[i, :i] @ y[:i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1): x[i] = (y[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x, L, U, y


def iter_matrix_radii(A):
    """(ρ(T_J), ρ(T_GS)): los iterativos convergen para todo x0 si y solo si ambos son < 1."""
    D = np.diag(np.diag(A)); L = np.tril(A, -1); U = np.triu(A, 1)
    TJ = -np.linalg.solve(D, L + U); TG = -np.linalg.solve(D + L, U)
    return (max(abs(np.linalg.eigvals(TJ))), max(abs(np.linalg.eigvals(TG))))


def tridiag(n, d, o=-1.0):
    return np.diag([d] * n) + np.diag([o] * (n - 1), 1) + np.diag([o] * (n - 1), -1)


def hilbert(n):
    return np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
