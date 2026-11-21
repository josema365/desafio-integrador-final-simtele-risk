"""Raíces de f(x)=0: bisección, Newton-Raphson y secante. Cada método devuelve un diccionario:
{hist: [{k, x, fx, paso}], convergio, x, nota}. Criterio de parada: |f(x_k)| < tol o |x_k - x_{k-1}| < tol."""


def _derivada_numerica(f, x):
    h = 1e-6 * (1.0 + abs(x))
    return (f(x + h) - f(x - h)) / (2 * h)


def biseccion(f, a, b, tol=1e-10, max_iter=200):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        return dict(hist=[], convergio=False, x=None,
                    nota="No hay cambio de signo en [a, b]: la bisección no es aplicable.")
    hist = []
    if fa == 0 or fb == 0:
        x = a if fa == 0 else b
        return dict(hist=[dict(k=1, x=x, fx=0.0, paso=0.0)], convergio=True, x=x, nota="Raíz en un extremo.")
    for k in range(1, max_iter + 1):
        c = (a + b) / 2.0
        fc = f(c)
        hist.append(dict(k=k, x=c, fx=fc, paso=(b - a) / 2.0))
        if abs(fc) < tol or (b - a) / 2.0 < tol:
            return dict(hist=hist, convergio=True, x=c, nota="")
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return dict(hist=hist, convergio=False, x=hist[-1]["x"], nota="Se alcanzó max_iter.")


def newton(f, df, x0, tol=1e-10, max_iter=100):
    df = df or (lambda x: _derivada_numerica(f, x))
    x = x0
    hist = [dict(k=0, x=x, fx=f(x), paso=None)]
    for k in range(1, max_iter + 1):
        d = df(x)
        if abs(d) < 1e-14:
            return dict(hist=hist, convergio=False, x=x, nota="Derivada ≈ 0: Newton no puede continuar.")
        xn = x - f(x) / d
        hist.append(dict(k=k, x=xn, fx=f(xn), paso=abs(xn - x)))
        if abs(hist[-1]["fx"]) < tol or abs(xn - x) < tol:
            return dict(hist=hist, convergio=True, x=xn, nota="")
        x = xn
    return dict(hist=hist, convergio=False, x=x, nota="Se alcanzó max_iter.")


def secante(f, x0, x1, tol=1e-10, max_iter=100):
    hist = [dict(k=0, x=x0, fx=f(x0), paso=None), dict(k=1, x=x1, fx=f(x1), paso=abs(x1 - x0))]
    for k in range(2, max_iter + 2):
        f0, f1 = hist[-2]["fx"], hist[-1]["fx"]
        if abs(f1 - f0) < 1e-300:
            return dict(hist=hist, convergio=False, x=hist[-1]["x"], nota="f(x_k) = f(x_{k-1}): secante no puede continuar.")
        xn = hist[-1]["x"] - f1 * (hist[-1]["x"] - hist[-2]["x"]) / (f1 - f0)
        hist.append(dict(k=k, x=xn, fx=f(xn), paso=abs(xn - hist[-1]["x"])))
        if abs(hist[-1]["fx"]) < tol or hist[-1]["paso"] < tol:
            return dict(hist=hist, convergio=True, x=xn, nota="")
    return dict(hist=hist, convergio=False, x=hist[-1]["x"], nota="Se alcanzó max_iter.")
