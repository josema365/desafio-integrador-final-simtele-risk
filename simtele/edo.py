"""EDO: Runge-Kutta de 2.º orden (variante declarada: HEUN / trapezoidal mejorado) y de 4.º orden clásico.
f(t, y) devuelve un arreglo del mismo tamaño que y. Paso fijo h = (T - t0)/n."""
import numpy as np


def rk2_heun(f, t0, y0, T, n):
    h = (T - t0) / n
    y = np.array(y0, float)
    ts = [t0]; ys = [y.copy()]
    for i in range(n):
        t = t0 + i * h
        k1 = np.asarray(f(t, y), float)
        k2 = np.asarray(f(t + h, y + h * k1), float)
        y = y + h * (k1 + k2) / 2.0
        ts.append(t0 + (i + 1) * h); ys.append(y.copy())
    return np.array(ts), np.array(ys)


def rk4(f, t0, y0, T, n):
    h = (T - t0) / n
    y = np.array(y0, float)
    ts = [t0]; ys = [y.copy()]
    for i in range(n):
        t = t0 + i * h
        k1 = np.asarray(f(t, y), float)
        k2 = np.asarray(f(t + h / 2, y + h * k1 / 2), float)
        k3 = np.asarray(f(t + h / 2, y + h * k2 / 2), float)
        k4 = np.asarray(f(t + h, y + h * k3), float)
        y = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        ts.append(t0 + (i + 1) * h); ys.append(y.copy())
    return np.array(ts), np.array(ys)
