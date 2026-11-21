"""Módulo 1 · Detención de emergencia de cabinas.
Modelo: x' = v, v' = -(a0 + b·v). Con b=0 es el caso analítico (desaceleración constante).
Escenario MAL_CONDICIONADO: v(t) = v0 (1 - t/T)^2 (raíz doble en t = T)."""
import math
import time

import numpy as np

from ..edo import rk2_heun, rk4
from ..errores import orden_convergencia_iterativa, orden_observado, sensibilidad_relativa
from ..integracion import romberg_muestras, simpson_muestras, trapecio_muestras
from ..raices import biseccion, newton, secante
from .comun import ejec, fmt, it, parsear, rangos, serie_error, submuestreo

NOMBRE = "M1. Detención de emergencia"
TITULO = "Detención de emergencia de cabinas"
PREGUNTA = "¿Cómo varían tiempo y distancia de detención frente a cambios de velocidad, reacción y frenado?"
MODELO = ("x′ = v,  v′ = −a(v,t) con a = a₀ + b·v.  Analítico (b = 0): t_det = v₀/a₀;  d_frenado = v₀²/(2a₀).  "
          "Distancia total = v₀·t_r + d_frenado.")
METODOS = "RK2 (Heun), RK4, trapecio, Simpson 1/3, Romberg, bisección, Newton, secante"


def defaults(tipo):
    if tipo == "NORMAL":
        return dict(modelo="lineal", v0=5.0, a0=0.8, b=0.0, tr=1.0, d_disp=40.0, n0=4, tol=1e-10)
    if tipo == "ESTRES":
        return dict(modelo="lineal", v0=6.0, a0=0.5, b=0.02, tr=2.0, d_disp=40.0, n0=2, tol=1e-10)
    return dict(modelo="raiz_doble", v0=5.0, T=6.25, tol=1e-10)


def spec(tipo):
    if tipo == "MAL_CONDICIONADO":
        return [dict(key="v0", label="Velocidad inicial v₀", unidad="m/s", min=0.1, max=20),
                dict(key="T", label="Tiempo de detención T", unidad="s", min=1, max=60),
                dict(key="tol", label="Tolerancia", unidad="", min=1e-14, max=1e-3)]
    return [dict(key="v0", label="Velocidad inicial v₀", unidad="m/s", min=0.1, max=20),
            dict(key="a0", label="Desaceleración base a₀", unidad="m/s²", min=0.05, max=10),
            dict(key="b", label="Coeficiente variable b (a = a₀ + b·v)", unidad="1/s", min=0, max=1),
            dict(key="tr", label="Tiempo de reacción t_r", unidad="s", min=0, max=10),
            dict(key="d_disp", label="Distancia disponible (supuesta)", unidad="m", min=1, max=500),
            dict(key="n0", label="Pasos base n₀ (se duplican 4 veces)", unidad="", min=2, max=256, entero=True),
            dict(key="tol", label="Tolerancia de raíces", unidad="", min=1e-14, max=1e-3)]


def validar(tipo, params):
    return rangos(spec(tipo), params)


# ---------- referencias analíticas ----------
def _v_ex(p):
    v0, a0, b = p["v0"], p["a0"], p["b"]
    if b > 0:
        return lambda t: (v0 + a0 / b) * np.exp(-b * np.asarray(t, float)) - a0 / b
    return lambda t: v0 - a0 * np.asarray(t, float)


def _t_det(p):
    v0, a0, b = p["v0"], p["a0"], p["b"]
    return math.log(1 + b * v0 / a0) / b if b > 0 else v0 / a0


def _d_fren(p):
    v0, a0, b = p["v0"], p["a0"], p["b"]
    if b > 0:
        return v0 / b - (a0 / b ** 2) * math.log(1 + b * v0 / a0)
    return v0 ** 2 / (2 * a0)


def _totales(p):
    td, df = _t_det(p), _d_fren(p)
    return td, df, p["v0"] * p["tr"] + df


def _orden_lista(errs, razon=2.0):
    return [orden_observado(errs[i - 1], errs[i], razon) for i in range(1, len(errs))]


def _fila_raiz(res, ref):
    filas, errs = [], []
    for h in res["hist"]:
        e = abs(h["x"] - ref)
        errs.append(e)
        filas.append(it(h["k"], {"t": h["x"], "paso": h["paso"]}, residual=abs(h["fx"]), eabs=e,
                        erel=e / abs(ref) if abs(ref) > 1e-12 else None))
    return filas, errs


def _ejec_raiz(nombre, res, ref, tol, extra=None):
    t0 = time.perf_counter()
    filas, errs = _fila_raiz(res, ref)
    ok = res["convergio"]
    err = errs[-1] if errs else None
    met = dict(iteraciones=len(res["hist"]), error_abs=err, residual=abs(res["hist"][-1]["fx"]) if res["hist"] else None,
               orden_observado=orden_convergencia_iterativa(errs), nota=res["nota"])
    val = dict(raiz=res["x"], huella=[res["x"]] if res["x"] is not None else [],
               serie_error=serie_error("iteración k (raíz de v(t)=0)", [f["numero"] for f in filas], errs))
    if extra:
        val.update(extra)
    return ejec(nombre, ok, filas, val, met, texto=res["nota"] or ("Convergió." if ok else "No convergió."),
                tolerancia=tol, max_iter=200, tiempo_ms=(time.perf_counter() - t0) * 1000)


def _ejec_integracion(v_fun, a, b, ref, ns=(2, 4, 8, 16, 32, 64)):
    out = []
    xs = {}
    tabla = {}
    for n in ns:
        t = np.linspace(a, b, n + 1); y = np.asarray(v_fun(t), float); h = (b - a) / n
        tabla[n] = dict(trap=trapecio_muestras(y, h), simp=simpson_muestras(y, h))
    n_max = ns[-1]; m = int(round(math.log2(n_max)))
    t = np.linspace(a, b, n_max + 1); R = romberg_muestras(np.asarray(v_fun(t), float), (b - a) / n_max)
    for nombre, clave in (("Trapecio compuesto", "trap"), ("Simpson 1/3 compuesto", "simp"), ("Romberg", None)):
        t0 = time.perf_counter(); filas = []; errs = []; vals = []
        for l, n in enumerate(ns):
            val = tabla[n][clave] if clave else R[int(round(math.log2(n)))][int(round(math.log2(n)))]
            e = abs(val - ref); errs.append(e); vals.append(val)
            filas.append(it(l + 1, {"n": n, "h": (b - a) / n, "I": val}, eabs=e, erel=e / abs(ref) if abs(ref) > 1e-12 else None))
        ords = _orden_lista(errs)
        met = dict(error_abs_final=errs[-1], orden_observado=ords, referencia=ref)
        v = dict(integral=vals[-1], huella=[vals[-1]],
                 serie_error=serie_error("n (pasos / subintervalos)", list(ns), errs, log_x=True))
        if clave is None:
            v["tabla_romberg"] = R
        out.append(ejec(nombre, True, filas, v, met, texto=f"Integral ≈ {vals[-1]:.10g} (n={n_max}).",
                        paso=(b - a) / n_max, tiempo_ms=(time.perf_counter() - t0) * 1000))
    return out


# ---------- ejecución ----------
def ejecutar(tipo, params):
    return _ejecutar_raiz_doble(params) if tipo == "MAL_CONDICIONADO" else _ejecutar_lineal(tipo, params)


def _ejecutar_lineal(tipo, p):
    v0, a0, b, tr = p["v0"], p["a0"], p["b"], p["tr"]
    td, dfren, dtot = _totales(p)
    vex = _v_ex(p)
    ts = np.linspace(0, td, 200)
    ref = dict(t_det=td, d_frenado=dfren, d_reaccion=v0 * tr, d_total=dtot, d_disponible=p["d_disp"],
               margen=p["d_disp"] - dtot, descripcion="Solución analítica (a = a₀ + b·v)",
               curva=submuestreo(ts, vex(ts)), curva_nombre="v(t) exacta [m/s]",
               eje_curvas=dict(xlabel="tiempo t [s]", ylabel="velocidad v [m/s]"))
    ejs = []
    f = lambda t, y: np.array([y[1], -(a0 + b * y[1])])
    for nombre, rk in (("RK2 (Heun)", rk2_heun), ("RK4", rk4)):
        t0 = time.perf_counter(); filas = []; errs = []; ns = []
        for l in range(5):
            n = p["n0"] * 2 ** l
            tt, yy = rk(f, 0.0, [0.0, v0], td, n)
            e = abs(yy[-1, 0] - dfren); errs.append(e); ns.append(n)
            filas.append(it(l + 1, {"n": n, "h": td / n, "x_T": yy[-1, 0], "v_T": yy[-1, 1]},
                            residual=abs(yy[-1, 1]), eabs=e, erel=e / dfren))
            if l == 0:
                curva = submuestreo(tt, yy[:, 1])
        ords = _orden_lista(errs)
        val = dict(d_frenado=yy[-1, 0], v_final=yy[-1, 1], curva=curva, huella=[yy[-1, 0], yy[-1, 1]],
                   serie_error=serie_error("n (pasos / subintervalos)", ns, errs, log_x=True))
        met = dict(error_abs_final=errs[-1], error_abs_grueso=errs[0], orden_observado=ords,
                   error_v_final=abs(yy[-1, 1]))
        ejs.append(ejec(nombre, True, filas, val, met, tolerancia=None, paso=td / ns[-1],
                        texto=f"Error en la distancia con n={ns[-1]}: {errs[-1]:.2e} m.",
                        tiempo_ms=(time.perf_counter() - t0) * 1000))
    # sensibilidad (analítica) a v0, a0, tr
    sens = []
    for par in ("v0", "a0", "tr"):
        for d in (-0.1, 0.1):
            q = dict(p); q[par] = p[par] * (1 + d)
            tq, dq, totq = _totales(q)
            sens.append(dict(parametro=par, delta=d, t_det=tq, d_total=totq,
                             S_d_total=sensibilidad_relativa(dtot, totq, p[par], q[par])))
    ejs[0]["valor"]["sensibilidad"] = sens
    ejs[0]["referencia"] = ref
    # raíces de v(t)=0 (formulación continua)
    tol = p["tol"]; df = lambda t: -(a0 + b * float(vex(t)))
    f1 = lambda t: float(vex(t))
    ejs.append(_ejec_raiz("Bisección", biseccion(f1, 0.0, 3 * v0 / a0, tol), td, tol))   # cota física independiente de la respuesta
    ejs.append(_ejec_raiz("Newton-Raphson", newton(f1, df, 0.0, tol), td, tol))
    ejs.append(_ejec_raiz("Secante", secante(f1, 0.0, 1.0, tol), td, tol))
    ejs += _ejec_integracion(lambda t: vex(t), 0.0, td, dfren)
    return dict(ejecuciones=ejs)


def _ejecutar_raiz_doble(p):
    v0, T, tol = p["v0"], p["T"], p["tol"]
    f1 = lambda t: v0 * (1 - t / T) ** 2
    df = lambda t: -2 * v0 / T * (1 - t / T)
    ts = np.linspace(0, T, 200)
    ref = dict(t_det=T, d_frenado=v0 * T / 3, descripcion="v(t)=v₀(1−t/T)²: raíz doble en t=T; distancia v₀T/3",
               curva=submuestreo(ts, f1(ts)), curva_nombre="v(t) exacta [m/s]",
               eje_curvas=dict(xlabel="tiempo t [s]", ylabel="velocidad v [m/s]"))
    sens = [dict(epsilon=e, desplazamiento_raiz=T * math.sqrt(e / v0)) for e in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4)]
    ejs = [_ejec_raiz("Bisección", biseccion(f1, 0.0, 2 * T, tol), T, tol),
           _ejec_raiz("Newton-Raphson", newton(f1, df, 0.0, tol), T, tol, extra=dict(sensibilidad=sens)),
           _ejec_raiz("Secante", secante(f1, 0.0, 1.0, tol), T, tol)]
    ejs[0]["referencia"] = ref
    ejs += _ejec_integracion(f1, 0.0, T, v0 * T / 3)
    return dict(ejecuciones=ejs)


# ---------- interpretación ----------
def titular(tipo, params, ejs):
    if tipo == "MAL_CONDICIONADO":
        return f"Raíz doble en t = {params['T']:g} s; distancia {params['v0'] * params['T'] / 3:.3f} m"
    td, df, dt = _totales(params)
    return f"t_det = {td:.3f} s · d_total = {dt:.2f} m (disp. {params['d_disp']:g} m)"


def interpretar(tipo, p, ejs):
    num, prob = [], []
    g = lambda m, k: (ejs.get(m, {}).get("metricas", {}) or {}).get(k)
    if tipo == "MAL_CONDICIONADO":
        bis = ejs.get("Bisección")
        if bis and not bis["convergio"]:
            num.append("Bisección NO es aplicable: v(t) ≥ 0 en todo el intervalo y no hay cambio de signo (la raíz es doble).")
        nw, sc = ejs.get("Newton-Raphson"), ejs.get("Secante")
        if nw:
            num.append(f"Newton converge solo linealmente (orden observado ≈ {fmt(g('Newton-Raphson', 'orden_observado'), 3)}, "
                       f"{g('Newton-Raphson', 'iteraciones')} iteraciones) en lugar de cuadráticamente, y termina con |v| = "
                       f"{fmt(g('Newton-Raphson', 'residual'))} pero error en t de {fmt(g('Newton-Raphson', 'error_abs'))} s: el residual engaña.")
        if sc:
            num.append(f"Secante: {g('Secante', 'iteraciones')} iteraciones, error final {fmt(g('Secante', 'error_abs'))} s.")
        et, es = g("Trapecio compuesto", "error_abs_final"), g("Simpson 1/3 compuesto", "error_abs_final")
        num.append(f"Integración de v(t) (cuadrática): trapecio tiene error {fmt(et)} (orden 2) mientras Simpson y Romberg son exactos "
                   f"(≈ {fmt(es)}), porque Simpson integra exactamente polinomios de grado ≤ 3.")
        prob.append(f"La detención ocurre en t = {p['T']:g} s con una distancia de {p['v0'] * p['T'] / 3:.3f} m, pero el instante exacto es hipersensible: "
                    f"una incertidumbre ε en la velocidad desplaza la raíz en T·√(ε/v₀); con ε = 10⁻⁶ m/s el instante se mueve ≈ "
                    f"{p['T'] * math.sqrt(1e-6 / p['v0']):.1e} s.")
        prob.append("Esto es sensibilidad matemática de la raíz doble (subexperimento numérico), no un riesgo físico adicional ni una falla del frenado.")
        prob.append("No puede inferirse el comportamiento de una cabina real: el perfil de desaceleración es una hipótesis sintética.")
        return dict(numerica=num, problema=prob)
    td, df, dt = _totales(p)
    e2, e4 = g("RK2 (Heun)", "error_abs_final"), g("RK4", "error_abs_final")
    o2 = [o for o in (g("RK2 (Heun)", "orden_observado") or []) if o is not None]
    o4 = [o for o in (g("RK4", "orden_observado") or []) if o is not None]
    if (e2 or 0) < 1e-11 and (e4 or 0) < 1e-11:
        num.append(f"RK2 y RK4 resuelven la EDO sin error apreciable (≈ {fmt(max(e2 or 0, e4 or 0))} m): con a constante la solución es "
                   "un polinomio de grado 2 y ambos métodos son exactos salvo redondeo.")
    else:
        num.append(f"Con n={ejs['RK2 (Heun)']['iteraciones'][-1]['aprox']['n']} pasos el error en la distancia es {fmt(e2)} m para RK2 y {fmt(e4)} m para RK4; "
                   f"órdenes observados ≈ {fmt(sum(o2) / len(o2), 3) if o2 else '—'} (RK2) y ≈ {fmt(sum(o4) / len(o4), 3) if o4 else '—'} (RK4), coherentes con 2 y 4.")
    its = {m: g(m, "iteraciones") for m in ("Bisección", "Newton-Raphson", "Secante")}
    num.append(f"Raíz de v(t)=0 (tol {p['tol']:g}): bisección {its['Bisección']} it., Newton {its['Newton-Raphson']} it., secante {its['Secante']} it. "
               f"(error final de t: {fmt(g('Bisección', 'error_abs'))}, {fmt(g('Newton-Raphson', 'error_abs'))}, {fmt(g('Secante', 'error_abs'))} s).")
    num.append(f"Integración de v(t) con n=64: error trapecio {fmt(g('Trapecio compuesto', 'error_abs_final'))}, Simpson "
               f"{fmt(g('Simpson 1/3 compuesto', 'error_abs_final'))}, Romberg {fmt(g('Romberg', 'error_abs_final'))} m.")
    marg = p["d_disp"] - dt
    cota = max(e for e in (e2, e4, g("Trapecio compuesto", "error_abs_final")) if e is not None)
    prob.append(f"Tiempo de detención {td:.3f} s; distancia de frenado {df:.3f} m + reacción {p['v0'] * p['tr']:.3f} m = {dt:.3f} m. "
                + (f"Supera la distancia disponible supuesta ({p['d_disp']:g} m) en {-marg:.3f} m: riesgo MODELADO alto."
                   if marg < 0 else f"Queda un margen de {marg:.3f} m frente a los {p['d_disp']:g} m supuestos."))
    prob.append(f"El error numérico máximo observado ({fmt(cota)} m) es "
                + ("despreciable" if cota < abs(marg) / 100 else "comparable") + f" frente al margen ({abs(marg):.3f} m): la decisión no depende del método.")
    sens = (ejs.get("RK2 (Heun)", {}).get("valor", {}) or {}).get("sensibilidad") or []
    sv = [s for s in sens if s.get("S_d_total") is not None]
    if sv:
        top = max(sv, key=lambda s: s["S_d_total"])
        nombres = {"v0": "la velocidad v₀", "a0": "la desaceleración a₀", "tr": "el tiempo de reacción t_r"}
        prob.append(f"La variable que más influye en la distancia total es {nombres[top['parametro']]} (sensibilidad relativa ≈ {top['S_d_total']:.2f}).")
    prob.append("No puede inferirse el comportamiento de cabinas reales (parámetros sintéticos) ni que el riesgo modelado equivalga a un fallo numérico: son cosas distintas.")
    return dict(numerica=num, problema=prob)
