"""Pruebas de los cuatro módulos: referencias analíticas, escenarios 3×4 e interpretación doble."""
import json
import math

import pytest

from simtele.casos import CASOS, ORDEN, TIPOS
from simtele.casos import m1_detencion as M1, m2_deterioro as M2, m3_oscilaciones as M3, m4_sensores as M4


def _ejs(r):
    return {e["metodo"]: e for e in r["ejecuciones"]}


def test_los_12_escenarios_se_ejecutan_con_parametros_validos_y_son_serializables():
    for k in ORDEN:
        for t in TIPOS:
            m = CASOS[k]; p = m.defaults(t)
            assert m.validar(t, p) == [], (k, t)
            r = m.ejecutar(t, p)
            assert r["ejecuciones"] and json.dumps(r)
            for e in r["ejecuciones"]:
                assert {"metodo", "convergio", "iteraciones", "valor", "metricas", "referencia"} <= set(e)


def test_interpretacion_tiene_dos_conclusiones_diferenciadas_en_los_12_escenarios():
    for k in ORDEN:
        for t in TIPOS:
            m = CASOS[k]; p = m.defaults(t); i = m.interpretar(t, p, _ejs(m.ejecutar(t, p)))
            assert len(i["numerica"]) >= 3 and len(i["problema"]) >= 3, (k, t)
            assert any("No puede inferirse" in x for x in i["problema"]), (k, t)


def test_m1_normal_coincide_con_la_solucion_analitica():
    p = M1.defaults("NORMAL"); td, df, dt = M1._totales(p)
    assert td == pytest.approx(6.25) and df == pytest.approx(15.625) and dt == pytest.approx(20.625)
    e = _ejs(M1.ejecutar("NORMAL", p))
    assert e["Newton-Raphson"]["valor"]["raiz"] == pytest.approx(6.25, abs=1e-9)
    assert e["RK4"]["metricas"]["error_abs_final"] < 1e-12       # exacto: a constante => polinomio de grado 2


def test_m1_estres_ordenes_2_y_4_y_riesgo_modelado_separado_del_error_numerico():
    p = M1.defaults("ESTRES"); r = M1.ejecutar("ESTRES", p); e = _ejs(r)
    o2 = [o for o in e["RK2 (Heun)"]["metricas"]["orden_observado"] if o]; o4 = [o for o in e["RK4"]["metricas"]["orden_observado"] if o]
    assert o2[-1] == pytest.approx(2, abs=0.1) and o4[-1] == pytest.approx(4, abs=0.1)
    assert M1._totales(p)[2] > p["d_disp"]                      # riesgo modelado: excede la distancia disponible
    assert e["RK4"]["metricas"]["error_abs_final"] < 1e-6       # y el error numérico es despreciable


def test_m1_raiz_doble_biseccion_no_aplicable_y_newton_lineal():
    e = _ejs(M1.ejecutar("MAL_CONDICIONADO", M1.defaults("MAL_CONDICIONADO")))
    assert not e["Bisección"]["convergio"]
    assert e["Newton-Raphson"]["metricas"]["orden_observado"] == pytest.approx(1.0, abs=0.05)
    assert e["Simpson 1/3 compuesto"]["metricas"]["error_abs_final"] < 1e-12   # Simpson exacto en grado 2
    assert e["Trapecio compuesto"]["metricas"]["error_abs_final"] > 1e-4


def test_m2_raiz_analitica_y_los_tres_metodos_coinciden():
    p = M2.defaults("NORMAL")
    assert M2.raiz_analitica(p) == pytest.approx(12.7698, abs=1e-3)
    e = _ejs(M2.ejecutar("NORMAL", p))
    for m in ("Bisección", "Newton-Raphson", "Secante"):
        assert e[m]["valor"]["raiz"] == pytest.approx(M2.raiz_analitica(p), abs=1e-6)
    assert e["Bisección"]["metricas"]["iteraciones"] > 20        # bisección: lineal; no acierta de casualidad


def test_m2_estres_adelanta_el_umbral_y_mal_cond_tiene_raiz_doble_y_jacobi_diverge():
    assert M2.raiz_analitica(M2.defaults("ESTRES")) < M2.raiz_analitica(M2.defaults("NORMAL"))
    p = M2.defaults("MAL_CONDICIONADO"); assert M2._doble(p) and M2.raiz_analitica(p) == pytest.approx(10.0)
    e = _ejs(M2.ejecutar("MAL_CONDICIONADO", p))
    assert not e["Bisección"]["convergio"] and not e["Jacobi"]["convergio"]
    assert e["Gradiente Conjugado"]["convergio"] and e["Gradiente Conjugado"]["metricas"]["kappa"] > 1e4


def test_m2_valida_umbral_inalcanzable():
    p = M2.defaults("MAL_CONDICIONADO"); p["Du"] = 0.31
    assert any("nunca alcanza" in x for x in M2.validar("MAL_CONDICIONADO", p))
    p2 = M2.defaults("NORMAL"); p2["Du"] = 0.05
    assert any("mayor" in x for x in M2.validar("NORMAL", p2))


def test_m3_solucion_analitica_cumple_condiciones_iniciales_y_rk_ordenes():
    p = M3.defaults("NORMAL")
    assert float(M3.x_exacta(p, 0.0)) == pytest.approx(p["x0"], abs=1e-12)
    pf = M3.defaults("ESTRES"); pf["x0"], pf["v0"] = 0.03, 0.2
    h = 1e-6
    assert float(M3.x_exacta(pf, 0.0)) == pytest.approx(0.03, abs=1e-12)
    assert (float(M3.x_exacta(pf, h)) - float(M3.x_exacta(pf, -h))) / (2 * h) == pytest.approx(0.2, abs=1e-6)
    e = _ejs(M3.ejecutar("NORMAL", p))
    assert e["RK2 (Heun)"]["metricas"]["orden_observado"][-1] == pytest.approx(2, abs=0.1)
    assert e["RK4"]["metricas"]["orden_observado"][-1] == pytest.approx(4, abs=0.1)
    assert e["RK4"]["metricas"]["error_abs_final"] < e["RK2 (Heun)"]["metricas"]["error_abs_final"] / 100


def test_m3_integracion_ordenes_y_resonancia():
    e = _ejs(M3.ejecutar("MAL_CONDICIONADO", M3.defaults("MAL_CONDICIONADO")))
    assert e["Trapecio compuesto"]["metricas"]["orden_observado"][-1] == pytest.approx(2, abs=0.1)
    assert e["Simpson 1/3 compuesto"]["metricas"]["orden_observado"][-1] == pytest.approx(4, abs=0.2)
    assert e["Romberg"]["metricas"]["error_abs_final"] < e["Simpson 1/3 compuesto"]["metricas"]["error_abs_final"]
    assert M3._factor_estacionario(M3.defaults("MAL_CONDICIONADO")) == pytest.approx(35.6, abs=0.1)


def test_m3_validacion_potencia_de_dos_y_amortiguamiento():
    p = M3.defaults("NORMAL"); p["n0"] = 100
    assert any("potencia de 2" in x for x in M3.validar("NORMAL", p))
    p["n0"] = 128; p["zeta"] = 1.5
    assert any("ζ" in x for x in M3.validar("NORMAL", p))


def test_m4_ruido_amplificado_por_lagrange_equiespaciado_pero_no_por_spline_ni_chebyshev():
    e = _ejs(M4.ejecutar("MAL_CONDICIONADO", M4.defaults("MAL_CONDICIONADO")))
    L, S, C = (e[m]["metricas"]["error_max"] for m in ("Lagrange", "Spline cúbico natural", "Lagrange (nodos Chebyshev)"))
    assert L > 20 * S and L > 20 * C
    assert e["Newton (diferencias divididas)"]["metricas"]["diferencia_con_lagrange"] < 1e-8
    assert e["Lagrange"]["referencia"]["lebesgue_equi"] > 100


def test_m4_normal_sin_ruido_reconstruye_bien_y_cruce_de_umbral():
    e = _ejs(M4.ejecutar("NORMAL", M4.defaults("NORMAL")))
    assert e["Lagrange"]["metricas"]["error_oculto_max"] < 0.02
    for m in ("Bisección (cruce de umbral)", "Newton (cruce de umbral)", "Secante (cruce de umbral)"):
        assert e[m]["valor"]["raiz"] == pytest.approx(math.asin(0.5), abs=0.01)
