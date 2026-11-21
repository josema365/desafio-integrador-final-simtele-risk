"""Pruebas reproducibles con solución analítica de cada familia de métodos."""
import math

import numpy as np
import pytest

from simtele.edo import rk2_heun, rk4
from simtele.errores import orden_observado, sensibilidad_relativa
from simtele.integracion import romberg, simpson, trapecio
from simtele.interpolacion import (lagrange_eval, lebesgue, newton_coef, newton_eval,
                                   nodos_chebyshev, spline_natural)
from simtele.raices import biseccion, newton, secante


# ---------- raíces ----------
def test_raices_sqrt2():
    f = lambda x: x * x - 2
    r1 = biseccion(f, 0, 2, 1e-12); r2 = newton(f, lambda x: 2 * x, 1.0, 1e-12); r3 = secante(f, 1.0, 2.0, 1e-12)
    for r in (r1, r2, r3):
        assert r["convergio"] and abs(r["x"] - math.sqrt(2)) < 1e-10
    assert len(r2["hist"]) < len(r1["hist"])            # Newton converge más rápido


def test_biseccion_sin_cambio_de_signo_no_es_aplicable():
    r = biseccion(lambda x: (x - 1) ** 2, 0, 2)
    assert not r["convergio"] and "no es aplicable" in r["nota"]


def test_newton_en_raiz_doble_converge_linealmente():
    r = newton(lambda x: (x - 1) ** 2, lambda x: 2 * (x - 1), 0.0, 1e-12, 200)
    e = [abs(h["x"] - 1) for h in r["hist"]]
    assert e[5] / e[4] == pytest.approx(0.5, abs=1e-6)  # el error se reduce a la mitad por iteración


def test_newton_derivada_cero_falla_con_mensaje():
    r = newton(lambda x: x * x + 1, lambda x: 0.0, 1.0)
    assert not r["convergio"]


# ---------- integración ----------
def test_integracion_polinomios_y_seno():
    assert trapecio(lambda x: x, 0, 2, 4) == pytest.approx(2.0, abs=1e-14)       # exacto para grado 1
    assert simpson(lambda x: x ** 3, 0, 2, 2) == pytest.approx(4.0, abs=1e-13)   # exacto para grado ≤ 3
    R = romberg(np.sin, 0, math.pi, 6)
    assert R[-1][-1] == pytest.approx(2.0, abs=1e-12)


def test_ordenes_de_integracion():
    e_t = [abs(trapecio(np.sin, 0, math.pi, n) - 2) for n in (8, 16)]
    e_s = [abs(simpson(np.sin, 0, math.pi, n) - 2) for n in (8, 16)]
    assert orden_observado(*e_t) == pytest.approx(2, abs=0.1)
    assert orden_observado(*e_s) == pytest.approx(4, abs=0.2)


def test_simpson_exige_n_par():
    with pytest.raises(ValueError):
        simpson(np.sin, 0, 1, 3)


# ---------- EDO ----------
def test_rk_ordenes_en_y_prima_igual_menos_y():
    f = lambda t, y: -y
    ex = math.exp(-1.0)
    for rk, p in ((rk2_heun, 2), (rk4, 4)):
        e = [abs(rk(f, 0, [1.0], 1.0, n)[1][-1, 0] - ex) for n in (10, 20)]
        assert orden_observado(*e) == pytest.approx(p, abs=0.25)


def test_rk_oscilador_libre_con_solucion_analitica():
    w = 2.0
    f = lambda t, y: np.array([y[1], -w * w * y[0]])
    ts, ys = rk4(f, 0, [1.0, 0.0], 3.0, 600)
    assert np.max(np.abs(ys[:, 0] - np.cos(w * ts))) < 1e-9


# ---------- interpolación ----------
def test_lagrange_newton_reproducen_polinomio_y_coinciden():
    xn = np.array([0., 1., 2., 3.]); yn = xn ** 3 - 2 * xn + 1
    x = np.linspace(0, 3, 13)
    l = lagrange_eval(xn, yn, x); n = newton_eval(xn, newton_coef(xn, yn), x)
    assert np.allclose(l, x ** 3 - 2 * x + 1, atol=1e-12) and np.allclose(l, n, atol=1e-12)


def test_spline_pasa_por_los_nodos_y_aproxima_el_seno():
    xn = np.linspace(0, 2 * math.pi, 13); S = spline_natural(xn, np.sin(xn))
    assert np.allclose(S(xn), np.sin(xn), atol=1e-12)
    x = np.linspace(0.5, 2 * math.pi - 0.5, 300)
    assert np.max(np.abs(S(x) - np.sin(x))) < 5e-3


def test_lebesgue_equiespaciados_vs_chebyshev():
    eq = lebesgue(np.linspace(0, 1, 21)); ch = lebesgue(nodos_chebyshev(0, 1, 21))
    assert eq > 1e3 and ch < 4


def test_sensibilidad_relativa():
    assert sensibilidad_relativa(2.0, 2.2, 1.0, 1.1) == pytest.approx(1.0)
    assert sensibilidad_relativa(0.0, 1.0, 1.0, 1.1) is None
