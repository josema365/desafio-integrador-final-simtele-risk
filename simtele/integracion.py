"""Integración numérica compuesta sobre muestras equiespaciadas y(x_i), paso h:
trapecio compuesto, Simpson 1/3 compuesto (n par) y Romberg (n = 2^m intervalos)."""
import numpy as np


def trapecio_muestras(y, h):
    y = np.asarray(y, float)
    return float(h * (y[0] / 2 + y[1:-1].sum() + y[-1] / 2))


def simpson_muestras(y, h):
    y = np.asarray(y, float)
    n = len(y) - 1
    if n < 2 or n % 2:
        raise ValueError("Simpson 1/3 compuesto requiere un número par (≥ 2) de subintervalos.")
    return float(h / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum()))


def romberg_muestras(y, h):
    """Tabla de Romberg R[k][j] (k = nivel, 2^k intervalos). Requiere len(y)-1 = 2^m."""
    y = np.asarray(y, float)
    n = len(y) - 1
    m = int(round(np.log2(n))) if n >= 1 else -1
    if n < 1 or 2 ** m != n:
        raise ValueError("Romberg requiere 2^m subintervalos.")
    R = []
    for k in range(m + 1):
        paso_k = n // (2 ** k)
        sub = y[::paso_k]
        fila = [trapecio_muestras(sub, h * paso_k)]
        for j in range(1, k + 1):
            fila.append(fila[j - 1] + (fila[j - 1] - R[k - 1][j - 1]) / (4 ** j - 1))
        R.append(fila)
    return R


def _muestras(f, a, b, n):
    x = np.linspace(a, b, n + 1)
    return np.asarray(f(x), float), (b - a) / n


def trapecio(f, a, b, n):
    y, h = _muestras(f, a, b, n); return trapecio_muestras(y, h)


def simpson(f, a, b, n):
    y, h = _muestras(f, a, b, n); return simpson_muestras(y, h)


def romberg(f, a, b, m):
    y, h = _muestras(f, a, b, 2 ** m); return romberg_muestras(y, h)
