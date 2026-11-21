"""Módulo 4 · Reconstrucción de señales de sensores.
Datos sintéticos y(t) = sin t en [0, 2π] (+ ruido gaussiano). Se ocultan nodos interiores y se reconstruyen con
Lagrange, Newton (diferencias divididas) y spline cúbico natural; el error real se mide contra sin t (sin ruido)."""
import math
import time

import numpy as np

from ..errores import orden_convergencia_iterativa
from ..interpolacion import (lagrange_eval, lebesgue, newton_coef, newton_eval, nodos_chebyshev, spline_natural)
from ..raices import biseccion, newton, secante
from .comun import ejec, fmt, it, rangos, serie_error, submuestreo

NOMBRE = "M4. Reconstrucción de sensores"
TITULO = "Reconstrucción de señales de sensores"
PREGUNTA = "¿Cómo reconstruir registros faltantes y medir la distorsión que producen datos con ruido?"
MODELO = ("Señal sintética y(t) = sin t en [0, 2π] con ruido gaussiano σ. Se ocultan nodos interiores (registros faltantes) "
          "y se reconstruyen con los restantes; error real = |p(t) − sin t|.")
METODOS = "Lagrange, Newton (diferencias divididas), spline cúbico natural · bisección, Newton, secante (cruce de umbral)"
A, B = 0.0, 2 * math.pi


def defaults(tipo):
    if tipo == "NORMAL":
        return dict(n=9, sigma=0.0, semilla=42, umbral=0.5, tol=1e-10)
    if tipo == "ESTRES":
        return dict(n=5, sigma=0.05, semilla=42, umbral=0.5, tol=1e-10)
    return dict(n=21, sigma=1e-3, semilla=42, umbral=0.5, tol=1e-10)


def spec(tipo):
    return [dict(key="n", label="Número de nodos n (equiespaciados)", unidad="", min=5, max=41, entero=True),
            dict(key="sigma", label="Desviación del ruido σ", unidad="", min=0, max=0.5),
            dict(key="semilla", label="Semilla del ruido", unidad="", min=0, max=10 ** 6, entero=True),
            dict(key="umbral", label="Umbral de la señal (cruce)", unidad="", min=0.05, max=0.95),
            dict(key="tol", label="Tolerancia de raíces", unidad="", min=1e-14, max=1e-3)]


def validar(tipo, p):
    return rangos(spec(tipo), p)


def _ocultos(n):
    """Índices interiores ocultados (≤ 3): aproximadamente n/4, n/2 y 3n/4."""
    idx = sorted({max(1, min(n - 2, n // 4)), n // 2, max(1, min(n - 2, (3 * n) // 4))})
    return [i for i in idx if 0 < i < n - 1][: 3] if n >= 7 else [n // 2]


def _datos(n, sigma, semilla, nodos=None):
    t = np.linspace(A, B, n) if nodos is None else np.asarray(nodos, float)
    rng = np.random.default_rng(semilla)
    ruido = rng.standard_normal(len(t))
    return t, np.sin(t) + sigma * ruido


def _metodos(tk, yk):
    c = newton_coef(tk, yk); S = spline_natural(tk, yk)
    return {"Lagrange": lambda x: lagrange_eval(tk, yk, x),
            "Newton (diferencias divididas)": lambda x: newton_eval(tk, c, x),
            "Spline cúbico natural": S}


def _errores(f, th, tf):
    eh = np.abs(f(th) - np.sin(th)); ef = np.abs(f(tf) - np.sin(tf))
    return float(eh.max()), float(ef.max()), float(math.sqrt(np.mean(ef ** 2)))


def ejecutar(tipo, p):
    n, sigma, sem, u, tol = int(p["n"]), p["sigma"], int(p["semilla"]), p["umbral"], p["tol"]
    t, y = _datos(n, sigma, sem)
    ocultos = _ocultos(n)
    keep = [i for i in range(n) if i not in ocultos]
    tk, yk = t[keep], y[keep]
    th = t[ocultos]
    tf = np.linspace(tk[0], tk[-1], 2001)
    dens = np.linspace(A, B, 400)
    ref = dict(descripcion="y(t) = sin t (sin ruido); nodos ocultos: " + ", ".join(f"t={x:.3f}" for x in th),
               curva=submuestreo(dens, np.sin(dens)), curva_nombre="sin t exacto",
               puntos={"datos usados": dict(x=tk.tolist(), y=yk.tolist()),
                       "datos ocultos (referencia)": dict(x=th.tolist(), y=np.sin(th).tolist())},
               eje_curvas=dict(xlabel="tiempo t", ylabel="señal y(t)"),
               nodos_ocultos=[int(i) for i in ocultos], lebesgue_equi=lebesgue(tk), lebesgue_cheb=lebesgue(nodos_chebyshev(A, B, len(tk))))
    fns = _metodos(tk, yk)
    ejs = []
    ns_tabla = [5, 9, 17, 21, 33]
    for nombre, f in fns.items():
        t0 = time.perf_counter()
        eh, ef, rms = _errores(f, th, tf)
        filas = []; errs = []
        for l, nn in enumerate(ns_tabla):
            tt, yy = _datos(nn, sigma, sem); oc = _ocultos(nn); kp = [i for i in range(nn) if i not in oc]
            fn = _metodos(tt[kp], yy[kp])[nombre]
            tfine = np.linspace(tt[kp][0], tt[kp][-1], 2001)
            e_h, e_f, _ = _errores(fn, tt[oc], tfine)
            errs.append(e_f)
            filas.append(it(l + 1, {"n": nn, "h": (B - A) / (nn - 1), "error_oculto": e_h, "error_max": e_f}, eabs=e_f))
        val = dict(curva=submuestreo(dens, f(dens)), huella=[eh, ef, rms],
                   error_nodos_ocultos=[float(abs(f(np.array([x]))[0] - math.sin(x))) for x in th],
                   serie_error=serie_error("n (número de nodos)", ns_tabla, errs, log_x=False))
        met = dict(error_oculto_max=eh, error_max=ef, error_rms=rms, n=n, sigma=sigma)
        ejs.append(ejec(nombre, True, filas, val, met, texto=f"Error máx. en nodos ocultos {eh:.2e}; en toda la curva {ef:.2e}.",
                        tiempo_ms=(time.perf_counter() - t0) * 1000))
    ejs[0]["referencia"] = ref
    xs = np.linspace(tk[0], tk[-1], 2001)
    dif = float(np.max(np.abs(fns["Lagrange"](xs) - fns["Newton (diferencias divididas)"](xs))))
    ejs[1]["metricas"]["diferencia_con_lagrange"] = dif
    # comparación con nodos de Chebyshev (mismo ruido), solo cuando el espaciamiento equiespaciado es desfavorable
    if tipo == "MAL_CONDICIONADO":
        t0 = time.perf_counter()
        tc = nodos_chebyshev(A, B, len(tk)); _, yc = _datos(len(tk), sigma, sem, tc)
        fc = lambda x: lagrange_eval(tc, yc, x)
        tfc = np.linspace(tc[0], tc[-1], 2001)
        ehc, efc, rmsc = _errores(fc, tc[len(tc) // 2:len(tc) // 2 + 1], tfc)
        ej = ejec("Lagrange (nodos Chebyshev)", True, [it(1, {"n": len(tk), "error_max": efc}, eabs=efc)],
                  dict(curva=submuestreo(dens, fc(dens)), huella=[efc, rmsc]),
                  dict(error_max=efc, error_rms=rmsc, n=len(tk), sigma=sigma, lebesgue=lebesgue(tc)),
                  texto=f"Error máx. {efc:.2e} con nodos de Chebyshev (Λ={lebesgue(tc):.2f}).",
                  tiempo_ms=(time.perf_counter() - t0) * 1000)
        ejs.append(ej)
    # cruce de umbral sobre la curva interpolada (spline)
    S = fns["Spline cúbico natural"]
    fS = lambda x: float(S(np.array([x]))[0] - u)
    ref_cruce = math.asin(u)
    a0, b0 = 0.0, math.pi / 2
    for nombre, res in (("Bisección (cruce de umbral)", biseccion(fS, a0, b0, tol)),
                        ("Newton (cruce de umbral)", newton(fS, None, 0.8, tol)),
                        ("Secante (cruce de umbral)", secante(fS, 0.3, 1.0, tol))):
        t0 = time.perf_counter(); filas, errs = [], []
        for h_ in res["hist"]:
            e = abs(h_["x"] - ref_cruce); errs.append(e)
            filas.append(it(h_["k"], {"t": h_["x"]}, residual=abs(h_["fx"]), eabs=e, erel=e / ref_cruce))
        ejs.append(ejec(nombre, res["convergio"], filas,
                        dict(raiz=res["x"], huella=[res["x"]] if res["x"] is not None else [],
                             serie_error=serie_error("iteración k (cruce de umbral)", [f["numero"] for f in filas], errs)),
                        dict(iteraciones=len(res["hist"]), error_abs=errs[-1] if errs else None,
                             residual=abs(res["hist"][-1]["fx"]) if res["hist"] else None,
                             orden_observado=orden_convergencia_iterativa(errs), referencia=ref_cruce),
                        texto=res["nota"] or f"Cruce en t ≈ {res['x']:.8f} (exacto {ref_cruce:.8f}).", tolerancia=tol,
                        tiempo_ms=(time.perf_counter() - t0) * 1000))
    return dict(ejecuciones=ejs)


def titular(tipo, p, ejs):
    return f"n = {int(p['n'])} nodos · σ = {p['sigma']:g}"


def interpretar(tipo, p, ejs):
    num, prob = [], []
    g = lambda m, k: (ejs.get(m, {}).get("metricas", {}) or {}).get(k)
    L, N, S = "Lagrange", "Newton (diferencias divididas)", "Spline cúbico natural"
    ref = next((e["referencia"] for e in ejs.values() if e.get("referencia", {}).get("lebesgue_equi") is not None), {})
    num.append(f"Error máximo en los nodos ocultos: Lagrange {fmt(g(L, 'error_oculto_max'))}, Newton {fmt(g(N, 'error_oculto_max'))}, spline {fmt(g(S, 'error_oculto_max'))}; "
               f"sobre toda la curva: {fmt(g(L, 'error_max'))}, {fmt(g(N, 'error_max'))}, {fmt(g(S, 'error_max'))}.")
    num.append(f"Lagrange y Newton construyen el mismo polinomio: difieren solo en {fmt(g(N, 'diferencia_con_lagrange'))} (redondeo); Newton es más barato de evaluar (Horner) y permite agregar nodos.")
    lam = ref.get("lebesgue_equi"); lamc = ref.get("lebesgue_cheb")
    if lam:
        num.append(f"Constante de Lebesgue de los nodos usados: Λ ≈ {fmt(lam, 3)} (equiespaciados) frente a ≈ {fmt(lamc, 3)} con nodos de Chebyshev: un error δ en los datos puede amplificarse hasta Λ·δ en el polinomio.")
    if p["sigma"] == 0:
        num.append("Sin ruido, el polinomio de grado alto es muy preciso en el interior; con más nodos equiespaciados el error disminuye hasta que domina el redondeo.")
    elif lam and lam > 100:
        c = ejs.get("Lagrange (nodos Chebyshev)")
        num.append(f"Con ruido σ = {p['sigma']:g} el polinomio equiespaciado amplifica la distorsión (error máx. {fmt(g(L, 'error_max'))} ≈ {fmt(g(L, 'error_max') / p['sigma'] if p['sigma'] else None, 3)}·σ), "
                   f"mientras el spline la mantiene en el orden del ruido ({fmt(g(S, 'error_max'))})"
                   + (f" y Lagrange con nodos de Chebyshev da {fmt(g('Lagrange (nodos Chebyshev)', 'error_max'))}." if c else "."))
    else:
        num.append(f"Con ruido σ = {p['sigma']:g} y pocos nodos, el spline (error {fmt(g(S, 'error_max'))}) suaviza mejor que el polinomio global (error {fmt(g(L, 'error_max'))}).")
    num.append(f"Cruce del umbral {p['umbral']:g}: bisección {g('Bisección (cruce de umbral)', 'iteraciones')} it., Newton {g('Newton (cruce de umbral)', 'iteraciones')} it., secante {g('Secante (cruce de umbral)', 'iteraciones')} it.; "
               f"error del cruce {fmt(g('Bisección (cruce de umbral)', 'error_abs'))} s (incluye el error de reconstrucción del spline).")
    prob.append(f"Con {int(p['n'])} nodos (separación {(B - A) / (int(p['n']) - 1):.3f}) y σ = {p['sigma']:g}, la reconstrucción de registros faltantes tiene un error máximo de {fmt(min(g(S, 'error_max'), g(L, 'error_max')))} "
                f"en la mejor curva y de {fmt(max(g(S, 'error_max'), g(L, 'error_max')))} en la peor, frente a una señal de amplitud 1.")
    if lam and lam > 100:
        prob.append("La confiabilidad de la reconstrucción depende del método y del espaciamiento: con nodos equiespaciados densos y ruido, el polinomio global NO es confiable (amplifica el ruido sobre todo en los extremos) aunque interpole exactamente los datos disponibles.")
    else:
        prob.append("Un espaciamiento mayor aumenta el error de reconstrucción en los registros ocultos; el ruido de los sensores se transmite a la curva reconstruida y limita la precisión alcanzable.")
    prob.append("Interpolar pasa por los datos ruidosos: un ajuste suavizado sería más apropiado si el ruido es grande. El cruce de umbral heredado de la curva reconstruida tiene la incertidumbre de la reconstrucción.")
    prob.append("No puede inferirse el comportamiento de sensores reales: la señal sin t y el ruido gaussiano son sintéticos; tampoco se extrapola fuera del intervalo de nodos.")
    return dict(numerica=num, problema=prob)
