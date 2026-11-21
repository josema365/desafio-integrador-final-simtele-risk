from . import m1_detencion as M1, m2_deterioro as M2, m3_oscilaciones as M3, m4_sensores as M4
from .comun import ETIQUETAS, TIPOS

CASOS = {"M1": M1, "M2": M2, "M3": M3, "M4": M4}
ORDEN = ["M1", "M2", "M3", "M4"]
NUMERO = {"M1": 1, "M2": 2, "M3": 3, "M4": 4}
POR_NOMBRE = {m.NOMBRE: k for k, m in CASOS.items()}
CORTO = {"M1": "Detención", "M2": "Deterioro", "M3": "Oscilaciones", "M4": "Sensores"}   # nombres de las tarjetas de Inicio
