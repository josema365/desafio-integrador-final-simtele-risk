"""Teoría de errores transversal (todos los módulos). Nunca se divide por cero:
si la referencia es ~0, el error relativo/porcentual es None y solo se informa el absoluto."""
import math
import numpy as np

TAU = 1e-12


def error_absoluto(x, ref):
    return float(np.linalg.norm(np.asarray(ref, float) - np.asarray(x, float)))


def error_relativo(x, ref, tau=TAU):
    n = float(np.linalg.norm(np.asarray(ref, float)))
    return None if n <= tau else error_absoluto(x, ref) / n


def error_porcentual(x, ref, tau=TAU):
    e = error_relativo(x, ref, tau)
    return None if e is None else 100.0 * e


def residual_relativo(A, b, x, tau=TAU):
    r = float(np.linalg.norm(np.asarray(b, float) - np.asarray(A, float) @ np.asarray(x, float)))
    nb = float(np.linalg.norm(b))
    return r / nb if nb > tau else r


def numero_condicion(A):
    return float(np.linalg.cond(np.asarray(A, float), 2))


def orden_observado(e_grueso, e_fino, razon=2.0, piso=1e-11):
    """p ≈ log_razon(e_grueso / e_fino). None si algún error está por debajo del ruido de redondeo."""
    if e_grueso is None or e_fino is None or e_grueso <= piso or e_fino <= piso:
        return None
    return math.log(e_grueso / e_fino) / math.log(razon)


def orden_convergencia_iterativa(errores, piso=1e-14):
    """Orden observado p ≈ ln(e_{k+1}/e_k) / ln(e_k/e_{k-1}) con los últimos errores significativos."""
    e = [x for x in errores if x is not None and x > piso]
    if len(e) < 3:
        return None
    a, b, c = e[-3], e[-2], e[-1]
    if a == b or b == c:
        return None
    try:
        return math.log(c / b) / math.log(b / a)
    except (ValueError, ZeroDivisionError):
        return None


def sensibilidad_relativa(f_base, f_pert, p_base, p_pert):
    """S = |Δf/f| / |Δp/p|. None si f_base≈0 o p_base≈0 o alguna evaluación no existe."""
    if None in (f_base, f_pert) or abs(f_base) <= TAU or abs(p_base) <= TAU or p_pert == p_base:
        return None
    return abs((f_pert - f_base) / f_base) / abs((p_pert - p_base) / p_base)


def amplificacion_perturbacion(A, b, x_ref, eps=1e-6, semilla=42, resolver=np.linalg.solve):
    A = np.asarray(A, float); b = np.asarray(b, float)
    d = np.random.default_rng(semilla).standard_normal(len(b))
    d = d / np.linalg.norm(d) * eps * np.linalg.norm(b)
    e = error_relativo(resolver(A, b + d), x_ref)
    return e, (None if e is None else e / eps)
