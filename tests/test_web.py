import os, sys
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)
from web.app import create_app  # noqa: E402


def test_inicio_muestra_cuatro_modulos():
    html = create_app().test_client().get("/").get_data(as_text=True)
    for nombre in ("Detención", "Deterioro", "Oscilaciones", "Sensores"):
        assert nombre in html


def test_cada_modulo_muestra_tres_escenarios():
    c = create_app().test_client()
    for n in (1, 2, 3, 4):
        html = c.get(f"/modulo/{n}").get_data(as_text=True)
        assert "NORMAL" in html and "ESTRÉS" in html and "MAL CONDICIONADO" in html
