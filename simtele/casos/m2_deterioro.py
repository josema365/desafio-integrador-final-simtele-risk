"""Módulo 2 · Deterioro mecánico y mantenimiento.
Modelo hipotético D(t) = D0 + α t + β t² (indicador adimensional, t en meses).
(a) Raíces: instante t* en que D(t*) = D_umbral (bisección, Newton, secante).
(b) Calibración: sistema A x = b SPD (Jacobi, Gauss-Seidel, LU, gradiente conjugado)."""
import math
import time

import numpy as np

from ..errores import numero_condicion, orden_convergencia_iterativa, sensibilidad_relativa
from ..lineal import (cg, gauss_seidel, jacobi, iter_matrix_radii, lu_solve, tridiag, hilbert)
from ..raices import biseccion, newton, secante
from .comun import ejec, fmt, it, rangos, serie_error, submuestreo

NOMBRE = "M2. Deterioro y mantenimiento"
TITULO = "Deterioro mecánico y mantenimiento"
PREGUNTA = "¿Cuándo alcanzaría un indicador hipotético de deterioro un umbral de inspección?"
MODELO = ("D(t) = D₀ + αt + βt²  [D adimensional, t en meses].  Resolver D(t) − D_umbral = 0 y elegir la raíz físicamente "
          "admisible (la menor positiva). Calibración de coeficientes: A x = b con A simétrica definida positiva.")
METODOS = "Bisección, Newton-Raphson, secante · Jacobi, Gauss-Seidel, LU, gradiente conjugado"


def defaults(tipo):
    base = dict(D0=0.10, tol=1e-10)
    if tipo == "NORMAL":
        return dict(base, alpha=0.020, beta=0.0015, Du=0.60, calib=dict(tipo="tridiag", n=4, diag=4.0, xref="secuencia"))
    if tipo == "ESTRES":   # mantenimiento retrasado: el deterioro avanza más rápido y la calibración es mayor
        return dict(base, alpha=0.030, beta=0.0025, Du=0.60, calib=dict(tipo="tridiag", n=20, diag=2.05, xref="unos"))
    return dict(base, alpha=0.04, beta=-0.002, Du=0.30, calib=dict(tipo="hilbert", n=4, xref="unos"))


def spec(tipo):
    return [dict(key="D0", label="Deterioro inicial D₀", unidad="adim.", min=0, max=1),
            dict(key="alpha", label="Tasa lineal α", unidad="1/mes", min=0, max=1),
            dict(key="beta", label="Aceleración β", unidad="1/mes²", min=-0.1, max=0.1),
            dict(key="Du", label="Umbral de inspección D_umbral", unidad="adim.", min=0.01, max=5),
            dict(key="tol", label="Tolerancia", unidad="", min=1e-14, max=1e-3)]


def validar(tipo, p):
    e = rangos(spec(tipo), p)
    if p["Du"] <= p["D0"]:
        e.append("El umbral debe ser mayor que el deterioro inicial (D_umbral > D₀).")
    if not e and raiz_analitica(p) is None:
        e.append("Con estos parámetros el indicador nunca alcanza el umbral (la ecuación no tiene raíz real positiva).")
    return e


# ---------- referencia analítica ----------
def _disc(a, b, c):
    return b * b - 4 * a * c


def raices_analiticas(D0, alpha, beta, Du):
    """Raíces reales positivas de β t² + α t + (D0 - Du) = 0, ordenadas (lista vacía si no hay)."""
    c = D0 - Du
    if abs(beta) < 1e-15:
        return [-c / alpha] if alpha > 0 and -c / alpha > 0 else []
    disc = _disc(beta, alpha, c)
    if disc < -1e-14:
        return []
    disc = max(disc, 0.0)
    rs = sorted({(-alpha + s * math.sqrt(disc)) / (2 * beta) for s in (1, -1)})
    return [r for r in rs if r > 0]


def raiz_analitica(p):
    r = raices_analiticas(p["D0"], p["alpha"], p["beta"], p["Du"])
    return r[0] if r else None


def _D(p):
    return lambda t: p["D0"] + p["alpha"] * np.asarray(t, float) + p["beta"] * np.asarray(t, float) ** 2


def _doble(p):
    return abs(_disc(p["beta"], p["alpha"], p["D0"] - p["Du"])) < 1e-12


def _matriz(calib):
    n = calib["n"]
    A = tridiag(n, calib["diag"]) if calib["tipo"] == "tridiag" else hilbert(n)
    x = np.arange(1.0, n + 1) if calib.get("xref") == "secuencia" else np.ones(n)
    return A, A @ x, x


def _fila_raiz(res, ref):
    filas, errs = [], []
    for h in res["hist"]:
        e = abs(h["x"] - ref); errs.append(e)
        filas.append(it(h["k"], {"t": h["x"], "paso": h["paso"]}, residual=abs(h["fx"]), eabs=e, erel=e / ref if ref > 1e-12 else None))
    return filas, errs


def _ejec_raiz(nombre, res, ref, tol):
    t0 = time.perf_counter()
    filas, errs = _fila_raiz(res, ref)
    met = dict(iteraciones=len(res["hist"]), error_abs=errs[-1] if errs else None,
               residual=abs(res["hist"][-1]["fx"]) if res["hist"] else None,
               orden_observado=orden_convergencia_iterativa(errs), nota=res["nota"])
    val = dict(raiz=res["x"], huella=[res["x"]] if res["x"] is not None else [],
               serie_error=serie_error("iteración k (raíz de D(t)=D_umbral)", [f["numero"] for f in filas], errs))
    return ejec(nombre, res["convergio"], filas, val, met, texto=res["nota"] or "Convergió.", tolerancia=tol, max_iter=200,
                tiempo_ms=(time.perf_counter() - t0) * 1000)


def _ejec_lineal(nombre, f, A, b, xs, tol, max_iter=50000):
    t0 = time.perf_counter()
    if nombre == "LU":
        x, *_ = lu_solve(A, b)
        r = float(np.linalg.norm(b - A @ x) / np.linalg.norm(b)); e = float(np.linalg.norm(x - xs))
        H = [dict(k=1, x=x, res=r, eabs=e, erel=e / np.linalg.norm(xs))]; ok = True
    else:
        H, ok = f(A, b, xs, tol=tol, maxit=max_iter); x = H[-1]["x"]
    filas = [it(h["k"], h["x"], residual=h["res"], eabs=h["eabs"], erel=h["erel"]) for h in H[:200]]
    ult = H[-1]
    met = dict(iteraciones=ult["k"], residual=ult["res"], error_abs=ult["eabs"], error_rel=ult["erel"],
               kappa=numero_condicion(A))
    val = dict(x=x, huella=list(x), serie_error=serie_error("iteración k (sistema lineal)", [h["k"] for h in H[1:200]],
                                                           [h["eabs"] for h in H[1:200]]) if len(H) > 2 else None)
    texto = f"Convergió en {ult['k']} iteración(es)." if ok else f"No convergió (k={ult['k']}, residual {ult['res']:.2e})."
    return ejec(nombre, ok, filas, val, met, texto=texto, tolerancia=tol, max_iter=max_iter, tiempo_ms=(time.perf_counter() - t0) * 1000)


def ejecutar(tipo, p):
    tstar = raiz_analitica(p)
    tol = p["tol"]
    D = _D(p)
    tv = -p["alpha"] / (2 * p["beta"]) if p["beta"] < 0 else None
    # intervalo de bisección independiente de la respuesta: cota con D(t) >= D0 + α t (β >= 0) o hasta el vértice (β < 0)
    if p["beta"] >= 0:
        tmax = (p["Du"] - p["D0"]) / p["alpha"] if p["alpha"] > 0 else math.sqrt((p["Du"] - p["D0"]) / p["beta"])
    elif _doble(p):
        tmax = 2 * tv
    else:
        tmax = tv
    t_fin = 1.6 * tstar if p["beta"] >= 0 else 2 * tv
    tt = np.linspace(0, t_fin, 200)
    ref = dict(t_umbral=tstar, descripcion="Raíz positiva admisible de β t² + α t + D₀ − D_u = 0",
               curva=submuestreo(tt, D(tt)), curva_nombre="D(t) exacta",
               curvas_extra={"umbral D_u": dict(x=[0.0, float(tt[-1])], y=[p["Du"], p["Du"]])},
               eje_curvas=dict(xlabel="tiempo t [meses]", ylabel="deterioro D [adim.]"))
    f = lambda t: float(p["D0"] + p["alpha"] * t + p["beta"] * t * t - p["Du"])
    df = lambda t: float(p["alpha"] + 2 * p["beta"] * t)
    ejs = [_ejec_raiz("Bisección", biseccion(f, 0.0, tmax, tol), tstar, tol),
           _ejec_raiz("Newton-Raphson", newton(f, df, 0.0, tol), tstar, tol),
           _ejec_raiz("Secante", secante(f, 0.0, 1.0, tol), tstar, tol)]
    # sensibilidad de t* a los parámetros (±1 % y ±10 %)
    sens = []
    for par in ("D0", "alpha", "beta", "Du"):
        for d in (-0.1, -0.01, 0.01, 0.1):
            q = dict(p); q[par] = p[par] * (1 + d)
            r = raiz_analitica(q) if (q["Du"] > q["D0"]) else None
            sens.append(dict(parametro=par, delta=d, t_umbral=r, S=sensibilidad_relativa(tstar, r, p[par], q[par])))
    ejs[0]["valor"]["sensibilidad"] = sens
    ejs[0]["referencia"] = ref
    A, b, xs = _matriz(p["calib"])
    rj, rg = iter_matrix_radii(A)
    for nombre, fn in (("Jacobi", jacobi), ("Gauss-Seidel", gauss_seidel), ("Gradiente Conjugado", cg), ("LU", None)):
        e = _ejec_lineal(nombre, fn, A, b, xs, tol)
        e["valor"]["radios"] = dict(rho_J=float(rj), rho_GS=float(rg), kappa=numero_condicion(A), n=len(b))
        ejs.append(e)
    return dict(ejecuciones=ejs)


def titular(tipo, p, ejs):
    r = raiz_analitica(p)
    return f"t* = {r:.3f} meses" + (" (raíz doble)" if _doble(p) else "")


def interpretar(tipo, p, ejs):
    num, prob = [], []
    g = lambda m, k: (ejs.get(m, {}).get("metricas", {}) or {}).get(k)
    tstar = raiz_analitica(p)
    bis = ejs.get("Bisección")
    if bis and not bis["convergio"]:
        num.append("Bisección NO es aplicable: no hay cambio de signo de D(t) − D_u (el umbral coincide con el máximo de D: raíz doble).")
    elif bis:
        num.append(f"Bisección: {g('Bisección', 'iteraciones')} iteraciones (garantizada, convergencia lineal).")
    num.append(f"Newton: {g('Newton-Raphson', 'iteraciones')} it. (error {fmt(g('Newton-Raphson', 'error_abs'))} meses, orden observado ≈ {fmt(g('Newton-Raphson', 'orden_observado'), 3)}); "
               f"secante: {g('Secante', 'iteraciones')} it. (error {fmt(g('Secante', 'error_abs'))}, orden ≈ {fmt(g('Secante', 'orden_observado'), 3)}).")
    if _doble(p):
        num.append(f"Con raíz doble Newton pierde el orden cuadrático (orden ≈ 1) y se detiene con |f| = {fmt(g('Newton-Raphson', 'residual'))} pero error {fmt(g('Newton-Raphson', 'error_abs'))} meses: el residual engaña.")
    jac, gs, cgm, lu = (ejs.get(m) for m in ("Jacobi", "Gauss-Seidel", "Gradiente Conjugado", "LU"))
    rad = (lu or {}).get("valor", {}).get("radios", {})
    kap = rad.get("kappa") or g("LU", "kappa")
    if jac and gs and cgm:
        if not jac["convergio"]:
            num.append(f"Calibración lineal (κ={fmt(kap)}): Jacobi diverge (ρ(T_J)={fmt(rad.get('rho_J'), 3)} > 1) aunque A es SPD — fallo de convergencia del algoritmo, distinto del mal condicionamiento.")
        else:
            num.append(f"Calibración lineal (κ={fmt(kap)}): Jacobi {jac['metricas']['iteraciones']} it., Gauss-Seidel {gs['metricas']['iteraciones']} it., gradiente conjugado {cgm['metricas']['iteraciones']} it.; LU directo.")
        if kap and kap > 1e4:
            num.append(f"El residual de Gauss-Seidel ({fmt(gs['metricas']['residual'])}) subestima su error relativo ({fmt(gs['metricas']['error_rel'])}) en ≈ κ veces; ningún método elimina este mal condicionamiento.")
    if tstar is not None:
        prob.append(f"Con D₀={p['D0']:g}, α={p['alpha']:g}, β={p['beta']:g} el indicador alcanza el umbral {p['Du']:g} en t* ≈ {tstar:.3f} meses (referencia analítica).")
    sens = [s for s in (bis or ejs.get("Newton-Raphson") or {}).get("valor", {}).get("sensibilidad", []) if s.get("S") is not None]
    if _doble(p):
        sin = [s for s in (bis or ejs.get("Bisección") or {}).get("valor", {}).get("sensibilidad", []) if s.get("t_umbral") is None]
        prob.append("La raíz es hipersensible: si el umbral sube apenas un 1 % por encima del máximo del indicador, el umbral ya no se alcanza nunca; si baja un 1 %, aparecen dos instantes distintos. "
                    "Esto es sensibilidad matemática del problema, no un deterioro mecánico más grave.")
    elif sens:
        top = max(sens, key=lambda s: s["S"])
        nom = {"D0": "el deterioro inicial D₀", "alpha": "la tasa α", "beta": "la aceleración β", "Du": "el umbral de inspección"}
        prob.append(f"El parámetro que más desplaza t* es {nom[top['parametro']]} (sensibilidad relativa ≈ {top['S']:.2f}).")
    if tipo == "ESTRES":
        prob.append("Mantenimiento hipotéticamente retrasado: el deterioro más rápido adelanta el instante de inspección respecto del escenario NORMAL (t* menor). La mayor dimensión de la calibración (n=20) sube el costo de cálculo, no el riesgo de inestabilidad.")
    if kap and kap > 1e4:
        prob.append("La calibración tipo Hilbert amplifica ≈ κ veces el ruido de las mediciones en los coeficientes: no conviene fijar una fecha puntual de inspección con esa estimación; habría que dar un intervalo o rediseñar las mediciones.")
    prob.append("No puede inferirse el estado de equipos reales (modelo y datos sintéticos), ni que un κ alto implique daño físico, ni la validez del modelo cuadrático fuera del horizonte calibrado.")
    return dict(numerica=num, problema=prob)
