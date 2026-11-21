"""Utilidades comunes de los módulos de caso."""
import math
import numpy as np

ETIQUETAS = {"NORMAL": "NORMAL", "ESTRES": "ESTRÉS", "MAL_CONDICIONADO": "MAL CONDICIONADO"}
TIPOS = ["NORMAL", "ESTRES", "MAL_CONDICIONADO"]


def limpio(o):
    """Convierte a tipos JSON puros (sin numpy; NaN/inf -> None)."""
    if isinstance(o, dict):
        return {str(k): limpio(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [limpio(v) for v in o]
    if isinstance(o, np.ndarray):
        return limpio(o.tolist())
    if isinstance(o, (np.floating, float)):
        o = float(o)
        return o if math.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def it(numero, aprox, residual=None, eabs=None, erel=None):
    return dict(numero=int(numero), aprox=limpio(aprox), residual=limpio(residual),
                error_abs=limpio(eabs), error_rel=limpio(erel))


def ejec(metodo, convergio, iteraciones, valor=None, metricas=None, referencia=None, texto="",
         tolerancia=None, paso=None, max_iter=None, tiempo_ms=0.0):
    return dict(metodo=metodo, convergio=bool(convergio), iteraciones=iteraciones,
                valor=limpio(valor or {}), metricas=limpio(metricas or {}), referencia=limpio(referencia or {}),
                texto=texto, tolerancia=tolerancia, paso=paso, max_iter=max_iter, tiempo_ms=float(tiempo_ms))


def submuestreo(x, y, max_pts=400):
    x = np.asarray(x); y = np.asarray(y)
    if len(x) <= max_pts:
        return dict(x=limpio(x), y=limpio(y))
    idx = np.unique(np.linspace(0, len(x) - 1, max_pts).astype(int))
    return dict(x=limpio(x[idx]), y=limpio(y[idx]))


def serie_error(xlabel, xs, ys, log_x=False):
    return dict(xlabel=xlabel, x=limpio(xs), y=limpio(ys), log_x=log_x)


def parsear(spec, form, defaults):
    """Combina los valores del formulario con los por defecto; devuelve (params, errores_de_formato)."""
    params = dict(defaults); errores = []
    for s in spec:
        k = s["key"]
        crudo = form.get(k)
        if crudo is None or str(crudo).strip() == "":
            continue
        try:
            v = float(str(crudo).replace(",", "."))
            if not math.isfinite(v):
                raise ValueError
            params[k] = int(v) if s.get("entero") else v
            if s.get("entero") and v != int(v):
                errores.append(f"{s['label']}: debe ser un entero.")
        except ValueError:
            errores.append(f"{s['label']}: valor numérico inválido.")
    return params, errores


def rangos(spec, params):
    errores = []
    for s in spec:
        v = params.get(s["key"])
        if v is None:
            continue
        lo, hi = s.get("min"), s.get("max")
        if lo is not None and v < lo:
            errores.append(f"{s['label']} debe ser ≥ {lo:g} (valor: {v:g}).")
        if hi is not None and v > hi:
            errores.append(f"{s['label']} debe ser ≤ {hi:g} (valor: {v:g}).")
    return errores


def fmt(v, d=4):
    if v is None:
        return "—"
    if isinstance(v, float) and (abs(v) < 1e-3 or abs(v) >= 1e5) and v != 0:
        return f"{v:.2e}"
    return f"{v:.{d}g}"
