"""Prueba obligatoria con los 4 módulos: ejecutar -> guardar -> cerrar/reiniciar -> recuperar parámetros, iteraciones y
gráficos -> repetir el cálculo -> coincidencia dentro de la tolerancia. El historial no depende de memoria."""
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

from simtele import bd  # noqa: E402
from simtele.casos import CASOS, ORDEN, TIPOS  # noqa: E402
from simtele.graficos import graficos_de_simulacion  # noqa: E402


def _guardar_todo(path):
    c = bd.conectar(path); bd.inicializar(c); ids = {}
    for k in ORDEN:
        for t in TIPOS:
            p = CASOS[k].defaults(t)
            ids[(k, t)] = bd.guardar_simulacion(c, k, t, p, CASOS[k].ejecutar(t, p), usuario="prueba")
    c.close()                      # cierre explícito
    return ids


def test_ciclo_completo_12_escenarios(tmp_path):
    ruta = str(tmp_path / "t.db"); ids = _guardar_todo(ruta)
    c = bd.conectar(ruta)          # conexión nueva = "reinicio" de la aplicación
    try:
        for (k, t), sid in ids.items():
            s = bd.cargar_simulacion(c, sid)
            assert s["params"] == CASOS[k].defaults(t) or s["params"].keys() == CASOS[k].defaults(t).keys()
            assert s["ejecuciones"] and all(e["iteraciones"] or e["texto"] for e in s["ejecuciones"])
            assert len(graficos_de_simulacion(s["ejecuciones"])) >= 2            # al menos dos gráficos por módulo
            nuevo, comp = bd.relanzar(c, sid)
            assert comp and all(x["coincide"] for x in comp), (k, t, comp)
            assert max((x["diff"] or 0) for x in comp) <= 1e-12
    finally:
        c.close()


def test_historial_busqueda_y_ultimas(tmp_path):
    ruta = str(tmp_path / "t.db"); _guardar_todo(ruta)
    c = bd.conectar(ruta)
    try:
        assert len(bd.listar_historial(c)) == 12
        assert len(bd.listar_historial(c, "Newton")) == 9      # M1, M2 y M4 usan Newton; M3 no
        assert {r["tipo"] for r in bd.listar_historial(c, "MAL_COND")} == {"MAL_CONDICIONADO"}
        assert len(bd.ultimas_por_escenario(c)) == 12
    finally:
        c.close()


def test_parametros_editados_se_guardan_y_se_recuperan(tmp_path):
    ruta = str(tmp_path / "t.db"); c = bd.conectar(ruta); bd.inicializar(c)
    p = CASOS["M1"].defaults("NORMAL"); p["v0"] = 4.0
    sid = bd.guardar_simulacion(c, "M1", "NORMAL", p, CASOS["M1"].ejecutar("NORMAL", p)); c.close()
    c = bd.conectar(ruta)
    try:
        assert bd.cargar_simulacion(c, sid)["params"]["v0"] == 4.0
    finally:
        c.close()
