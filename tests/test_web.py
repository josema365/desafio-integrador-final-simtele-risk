import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)
from web.app import create_app  # noqa: E402


def _cli(tmp_path):
    return create_app(str(tmp_path / "t.db")).test_client()


def test_inicio_muestra_los_cuatro_modulos(tmp_path):
    html = _cli(tmp_path).get("/").get_data(as_text=True)
    for nombre in ("Detención", "Deterioro", "Oscilaciones", "Sensores"):
        assert nombre in html


def test_cada_modulo_y_escenario_muestra_parametros(tmp_path):
    c = _cli(tmp_path)
    for n in (1, 2, 3, 4):
        for t in ("NORMAL", "ESTRES", "MAL_CONDICIONADO"):
            r = c.get(f"/modulo/{n}?tipo={t}")
            assert r.status_code == 200 and "<input" in r.get_data(as_text=True)


def test_validacion_de_parametros_rechaza_valores_invalidos(tmp_path):
    c = _cli(tmp_path)
    r = c.post("/modulo/1/ejecutar", data={"tipo": "NORMAL", "v0": "-3", "a0": "abc"})
    h = r.get_data(as_text=True)
    assert r.status_code == 400 and "Corrige" in h and "v₀" in h


def test_ejecutar_guardar_reiniciar_recuperar_y_relanzar(tmp_path):
    db = str(tmp_path / "t.db")
    r = create_app(db).test_client().post("/modulo/4/ejecutar", data={"tipo": "MAL_CONDICIONADO", "usuario": "equipo-1"})
    assert r.status_code == 302
    c2 = create_app(db).test_client()            # reinicio de la aplicación sobre el mismo archivo
    assert "equipo-1" in c2.get("/historial?q=Sensores").get_data(as_text=True)
    sim = c2.get("/simulacion/1").get_data(as_text=True)
    assert sim.count("<svg") >= 2 and "Conclusión numérica" in sim and "Conclusión del problema" in sim
    rel = c2.post("/simulacion/1/relanzar").get_data(as_text=True)
    assert 'class="no"' not in rel and 'class="si"' in rel


def test_ejecutar_todo_llena_el_panel_de_comparacion(tmp_path):
    c = _cli(tmp_path)
    assert c.post("/ejecutar-todo").status_code == 302
    h = c.get("/comparacion").get_data(as_text=True)
    assert h.count("Abrir") == 0 and h.count("#") >= 12
