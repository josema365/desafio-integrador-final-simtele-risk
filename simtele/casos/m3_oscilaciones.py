"""Módulo 3 · Oscilaciones por viento.  x' = v,  v' = (F(t) - c v - k x)/m,  F(t) = F0·sin(Ω t).
Referencia analítica exacta (ζ < 1): parte estacionaria + transitorio subamortiguado."""
import math
import time

import numpy as np

from ..edo import rk2_heun, rk4
from ..errores import orden_observado, sensibilidad_relativa
from ..integracion import romberg_muestras, simpson_muestras, trapecio_muestras
from .comun import ejec, fmt, it, rangos, serie_error, submuestreo

NOMBRE = "M3. Oscilaciones por viento"
TITULO = "Oscilaciones por viento"
PREGUNTA = "¿Cómo se modifica la oscilación ante cambios de fuerza externa, amortiguamiento y parámetros del modelo?"
MODELO = ("x′ = v,  v′ = (F(t) − c·v − k·x)/m,  F(t) = F₀ sin(Ωt).  Libre subamortiguado (F₀=0): "
          "x = e^(−ζω₀t)[C₁cos ω_d t + C₂ sin ω_d t],  ω₀ = √(k/m),  c = 2ζ√(km).")
METODOS = "RK2 (Heun), RK4, trapecio, Simpson 1/3, Romberg (sobre x(t)²)"


def defaults(tipo):
    base = dict(m=1000.0, k=40000.0, x0=0.10, v0=0.0)
    if tipo == "NORMAL":
        return dict(base, zeta=0.05, F0=0.0, r=0.9, T=20.0, n0=256)
    if tipo == "ESTRES":
        return dict(base, x0=0.0, zeta=0.02, F0=2000.0, r=0.9, T=60.0, n0=512)
    return dict(base, x0=0.0, zeta=0.01, F0=2000.0, r=0.99, T=60.0, n0=1024)


def spec(tipo):
    return [dict(key="m", label="Masa m", unidad="kg", min=1, max=1e6),
            dict(key="k", label="Rigidez k", unidad="N/m", min=1, max=1e8),
            dict(key="zeta", label="Razón de amortiguamiento ζ (0<ζ<1)", unidad="", min=0.001, max=0.99),
            dict(key="F0", label="Amplitud de la fuerza F₀ (0 = vibración libre)", unidad="N", min=0, max=1e6),
            dict(key="r", label="Razón de frecuencias Ω/ω₀", unidad="", min=0.1, max=3),
            dict(key="x0", label="Posición inicial x(0)", unidad="m", min=-5, max=5),
            dict(key="v0", label="Velocidad inicial v(0)", unidad="m/s", min=-20, max=20),
            dict(key="T", label="Tiempo de simulación T", unidad="s", min=1, max=200),
            dict(key="n0", label="Pasos base n₀ (potencia de 2; se duplican 4 veces)", unidad="", min=64, max=2048, entero=True)]


def validar(tipo, p):
    e = rangos(spec(tipo), p)
    n0 = int(p["n0"])
    if n0 & (n0 - 1):
        e.append("n₀ debe ser una potencia de 2 (64, 128, 256, …) para poder aplicar Romberg.")
    return e


# ---------- referencia analítica ----------
def _par(p):
    w0 = math.sqrt(p["k"] / p["m"]); c = 2 * p["zeta"] * math.sqrt(p["k"] * p["m"])
    return w0, c, w0 * math.sqrt(1 - p["zeta"] ** 2), p["r"] * w0


def x_exacta(p, t):
    t = np.asarray(t, float)
    w0, c, wd, Om = _par(p)
    z = p["zeta"]
    xp = np.zeros_like(t); xp0 = vp0 = 0.0
    if p["F0"] != 0:
        H = p["F0"] / (p["k"] - p["m"] * Om ** 2 + 1j * c * Om)
        xp = H.imag * np.cos(Om * t) + H.real * np.sin(Om * t)
        xp0, vp0 = H.imag, Om * H.real
    C1 = p["x0"] - xp0
    C2 = (p["v0"] - vp0 + z * w0 * C1) / wd
    return xp + np.exp(-z * w0 * t) * (C1 * np.cos(wd * t) + C2 * np.sin(wd * t))


def _amplitud(p, N=2 ** 15):
    t = np.linspace(0, p["T"], N + 1)
    return float(np.max(np.abs(x_exacta(p, t))))


def _integral_ref(p):
    t = np.linspace(0, p["T"], 2 ** 18 + 1)
    return simpson_muestras(x_exacta(p, t) ** 2, p["T"] / 2 ** 18)


def _factor_estacionario(p):
    r, z = p["r"], p["zeta"]
    return 1 / math.sqrt((1 - r * r) ** 2 + (2 * z * r) ** 2)


def _rhs(p):
    w0, c, wd, Om = _par(p)
    return lambda t, y: np.array([y[1], (p["F0"] * math.sin(Om * t) - c * y[1] - p["k"] * y[0]) / p["m"]])


def _integral_rapida(p, N=2 ** 13):
    t = np.linspace(0, p["T"], N + 1)
    return simpson_muestras(x_exacta(p, t) ** 2, p["T"] / N)


def _sensibilidad(p):
    """Sensibilidad relativa de la amplitud máxima (S) y de la magnitud acumulada ∫x² (S_I) a los parámetros."""
    A0, I0 = _amplitud(p, 2 ** 13), _integral_rapida(p); out = []
    pars = ["zeta", "k"] + (["r"] if p["F0"] != 0 else [])
    for par in pars:
        for d in (-0.1, -0.01, 0.01, 0.1):
            q = dict(p); q[par] = p[par] * (1 + d)
            if par == "zeta" and not (0 < q[par] < 1):
                continue
            Aq, Iq = _amplitud(q, 2 ** 13), _integral_rapida(q)
            out.append(dict(parametro=par, delta=d, amplitud=Aq, integral=Iq,
                            S=sensibilidad_relativa(A0, Aq, p[par], q[par]),
                            S_I=sensibilidad_relativa(I0, Iq, p[par], q[par])))
    return out


# ---------- ejecución ----------
def ejecutar(tipo, p):
    w0, c, wd, Om = _par(p)
    T, n0 = p["T"], int(p["n0"])
    A_ex = _amplitud(p); I_ref = _integral_ref(p)
    tt = np.linspace(0, T, 600)
    ref = dict(amplitud_max=A_ex, integral_x2=I_ref, omega0=w0, omega_d=wd, c=c,
               factor_estacionario=_factor_estacionario(p) if p["F0"] != 0 else None,
               descripcion="Solución analítica (estacionaria + transitorio); integral de x² por Simpson con 2¹⁸ subintervalos",
               curva=submuestreo(tt, x_exacta(p, tt)), curva_nombre="x(t) exacta [m]",
               eje_curvas=dict(xlabel="tiempo t [s]", ylabel="posición x [m]"))
    f = _rhs(p)
    ejs, finos = [], {}
    for nombre, rk in (("RK2 (Heun)", rk2_heun), ("RK4", rk4)):
        t0 = time.perf_counter(); filas = []; errs = []; ns = []; curvas = []; amp = []
        for l in range(5):
            n = n0 * 2 ** l
            ts, ys = rk(f, 0.0, [p["x0"], p["v0"]], T, n)
            ex = x_exacta(p, ts)
            e = float(np.max(np.abs(ys[:, 0] - ex))); errs.append(e); ns.append(n)
            amp.append(float(np.max(np.abs(ys[:, 0]))))
            filas.append(it(l + 1, {"n": n, "h": T / n, "omega_h": w0 * T / n, "x_T": ys[-1, 0], "A_max": amp[-1]},
                            eabs=e, erel=e / A_ex if A_ex > 1e-12 else None))
            curvas.append(submuestreo(ts, ys[:, 0], 500))
            if l == 4:
                finos[nombre] = ys[:, 0]
        ords = [orden_observado(errs[i - 1], errs[i]) for i in range(1, 5)]
        ejs.append(dict(nombre=nombre, filas=filas, errs=errs, ns=ns, curvas=curvas, ords=ords, amp=amp, t0=t0,
                        xT=float(ys[-1, 0]), vT=float(ys[-1, 1])))
    # nivel de la curva a mostrar: el más grueso con error de RK2 menor que la mitad de la amplitud real
    nivel = next((l for l in range(5) if ejs[0]["errs"][l] < 0.5 * A_ex), 4)
    res = []
    for e in ejs:
        val = dict(x_T=e["xT"], curva=e["curvas"][nivel], nivel_curva=dict(n=e["ns"][nivel], h=T / e["ns"][nivel]),
                   huella=[e["xT"], e["vT"], e["errs"][-1]],
                   serie_error=serie_error("n (pasos / subintervalos)", e["ns"], e["errs"], log_x=True))
        met = dict(error_abs_final=e["errs"][-1], error_abs_grueso=e["errs"][0], orden_observado=e["ords"],
                   amplitud_max=e["amp"][-1], amplitud_max_grueso=e["amp"][0], amplitud_exacta=A_ex,
                   error_amplitud=abs(e["amp"][-1] - A_ex), omega_h_grueso=w0 * T / e["ns"][0])
        res.append(ejec(e["nombre"], True, e["filas"], val, met, paso=T / e["ns"][-1],
                        texto=f"Error global máx. con n={e['ns'][-1]}: {e['errs'][-1]:.2e} m.",
                        tiempo_ms=(time.perf_counter() - e["t0"]) * 1000))
    res[0]["referencia"] = ref
    res[0]["valor"]["sensibilidad"] = _sensibilidad(p)
    # integración de x(t)²: cuadratura sobre muestras de la solución analítica (aísla el error del método de integración);
    # el valor obtenido con las muestras de RK4 se informa aparte en las métricas.
    Nf = n0 * 16; h = T / Nf
    tg = np.linspace(0, T, Nf + 1); y2 = x_exacta(p, tg) ** 2
    I_rk4 = simpson_muestras(finos["RK4"] ** 2, h)
    R = romberg_muestras(y2, h)
    ns_int = [Nf // 2 ** s for s in (4, 3, 2, 1, 0)]
    for nombre, clave in (("Trapecio compuesto", "trap"), ("Simpson 1/3 compuesto", "simp"), ("Romberg", None)):
        t0 = time.perf_counter(); filas = []; errs = []
        for l, n in enumerate(ns_int):
            s_ = Nf // n; yy = y2[::s_]; hh = h * s_
            val = (trapecio_muestras(yy, hh) if clave == "trap" else simpson_muestras(yy, hh) if clave == "simp"
                   else R[int(round(math.log2(n)))][int(round(math.log2(n)))])
            e = abs(val - I_ref); errs.append(e)
            filas.append(it(l + 1, {"n": n, "h": hh, "I": val}, eabs=e, erel=e / I_ref if I_ref > 1e-12 else None))
        ords = [orden_observado(errs[i - 1], errs[i]) for i in range(1, len(errs))]
        v = dict(integral=val, huella=[val], serie_error=serie_error("n (pasos / subintervalos)", ns_int, errs, log_x=True))
        met = dict(error_abs_final=errs[-1], orden_observado=ords, referencia=I_ref)
        if clave is None:
            v["tabla_romberg"] = R
            met.update(integral_con_muestras_RK4=I_rk4, error_con_muestras_RK4=abs(I_rk4 - I_ref))
        res.append(ejec(nombre, True, filas, v, met, texto=f"∫x² dt ≈ {val:.8g} (referencia {I_ref:.8g}).", paso=h,
                        tiempo_ms=(time.perf_counter() - t0) * 1000))
    return dict(ejecuciones=res)


def titular(tipo, p, ejs):
    w0 = math.sqrt(p["k"] / p["m"])
    return f"ω₀ = {w0:.3f} rad/s · ζ = {p['zeta']:g}" + (f" · Ω/ω₀ = {p['r']:g}" if p["F0"] else " · vibración libre")


def interpretar(tipo, p, ejs):
    num, prob = [], []
    g = lambda m, k: (ejs.get(m, {}).get("metricas", {}) or {}).get(k)
    e2, e4 = g("RK2 (Heun)", "error_abs_final"), g("RK4", "error_abs_final")
    c2, c4 = g("RK2 (Heun)", "error_abs_grueso"), g("RK4", "error_abs_grueso")
    o2 = [o for o in (g("RK2 (Heun)", "orden_observado") or []) if o is not None]
    o4 = [o for o in (g("RK4", "orden_observado") or []) if o is not None]
    A = g("RK2 (Heun)", "amplitud_exacta")
    n0, nf = int(p["n0"]), int(p["n0"]) * 16
    num.append(f"Error global máximo vs. la solución analítica: con n={n0} pasos RK2 = {fmt(c2)} m y RK4 = {fmt(c4)} m; con n={nf} pasos RK2 = {fmt(e2)} m y RK4 = {fmt(e4)} m "
               f"(RK4 es ≈ {fmt((e2 / e4) if e4 else None, 3)} veces más preciso).")
    if o2 and o4:
        num.append(f"Órdenes observados ≈ {fmt(o2[-1], 3)} (RK2) y ≈ {fmt(o4[-1], 3)} (RK4) al duplicar n, coherentes con 2 y 4.")
    if A and c2 and c2 > A:
        num.append(f"Con el paso más grueso (ω₀h = {fmt(g('RK2 (Heun)', 'omega_h_grueso'), 3)}) RK2 amplifica artificialmente la amplitud: error {fmt(c2)} m frente a una amplitud real de {fmt(A)} m. "
                   "Es un efecto de ESTABILIDAD del método (factor √(1+(ωh)⁴/4) por paso), no mal condicionamiento del problema: al refinar el paso desaparece.")
    it_ = {m: g(m, "error_abs_final") for m in ("Trapecio compuesto", "Simpson 1/3 compuesto", "Romberg")}
    num.append(f"Integral de x(t)² (n={nf}): error trapecio {fmt(it_['Trapecio compuesto'])}, Simpson {fmt(it_['Simpson 1/3 compuesto'])}, Romberg {fmt(it_['Romberg'])} "
               f"(cuadratura sobre la solución analítica); con las muestras de RK4 la integral se desvía {fmt(g('Romberg', 'error_con_muestras_RK4'))}, es decir, el error de la EDO {'domina al' if (g('Romberg', 'error_con_muestras_RK4') or 0) > max(v or 0 for v in it_.values()) else 'es comparable o menor que el'} de la cuadratura.")
    prob.append(f"Amplitud máxima de la oscilación: {fmt(A)} m. "
                + (f"Con Ω/ω₀ = {p['r']:g} y ζ = {p['zeta']:g} el factor de amplificación estacionario es ≈ {fmt(_factor_estacionario(p), 3)} (cuando Ω ≈ ω₀ y ζ es pequeño, el viento periódico se amplifica mucho)."
                   if p["F0"] else "Es vibración libre: la amplitud decae con la constante 1/(ζω₀) = %.2f s." % (1 / (p["zeta"] * math.sqrt(p["k"] / p["m"])))))
    sv = (ejs.get("RK2 (Heun)", {}).get("valor", {}) or {}).get("sensibilidad", [])
    clave, que = ("S", "la amplitud máxima") if p["F0"] else ("S_I", "la magnitud acumulada ∫x² dt")
    sens = [s for s in sv if s.get(clave) is not None]
    if sens:
        top = max(sens, key=lambda s: s[clave])
        nom = {"zeta": "el amortiguamiento ζ", "k": "la rigidez k", "r": "la razón de frecuencias Ω/ω₀"}
        prob.append(f"La variable que más modifica {que} es {nom[top['parametro']]} (sensibilidad relativa ≈ {top[clave]:.1f}).")
        if tipo == "MAL_CONDICIONADO":
            prob.append("Cerca de la resonancia con poco amortiguamiento, variaciones pequeñas de Ω o ζ cambian mucho la amplitud: sensibilidad demostrable del modelo, distinta de la falta de convergencia del algoritmo.")
    prob.append("Magnitud acumulada ∫x² dt = %s m²·s (energía de desplazamiento acumulada, no un esfuerzo estructural medido)." % fmt(g("Romberg", "referencia")))
    prob.append("No puede inferirse el esfuerzo real sobre cables o cabinas: el modelo es de un solo grado de libertad, lineal y con parámetros sintéticos; ignora no linealidades, turbulencia y geometría.")
    return dict(numerica=num, problema=prob)
