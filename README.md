# SIMTELE-RISK

Desafío Integrador Final de **Métodos Numéricos**: aplicación web educativa con cuatro casos de estudio (detención de emergencia, deterioro y mantenimiento, oscilaciones por viento y reconstrucción de sensores), inspirados en un sistema de transporte por cable (tipo Mi Teleférico, La Paz).

> **Aviso:** proyecto estrictamente educativo. Escenarios hipotéticos que **no describen fallas verificadas** de ninguna empresa. Datos y parámetros **sintéticos o supuestos**. Sin conexión con equipos de control reales.

**Integrantes:** Jose Manuel Huanca Tito, Jhonatan Pongo Gavincha · **Mención:** Ingeniería de Sistemas / Desarrollo de Software e Innovación Tecnológica · **Docente:** Lic. Brigida Alexandra Carvajal Blanco 

## Estado (Semana 1)

- [x] Planteamiento de los cuatro casos y sus tres escenarios (NORMAL, ESTRÉS, MAL CONDICIONADO)
- [x] Sistemas lineales: Jacobi, Gauss-Seidel, LU y gradiente conjugado (`simtele/lineal.py`) con teoría de errores
- [x] Primeras pruebas de los tres escenarios en el Módulo 2 (calibración lineal)
- [x] Prototipo web con los cuatro módulos visibles (`web/`)
- [x] DER inicial (`db/diagrama_der.png`) y esquema SQL (`db/schema.sql`)
- [x] Prueba de persistencia: guardar → cerrar → recuperar → repetir → comparar (`tests/test_persistencia.py`)
- [ ] Interpolación, raíces, integración y EDO (Semana 2)

## Instalación y uso

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Desde la **raíz** del proyecto:

```bash
python web/app.py               # prototipo web en http://127.0.0.1:5000
pytest -v                       # pruebas
python tests/test_persistencia.py   # prueba de persistencia con tabla de resultados
```

## Escenarios de la Semana 1 (Módulo 2, calibración lineal)

| Escenario | Matriz A | κ(A) | Qué demuestra |
|---|---|---|---|
| NORMAL | Tridiagonal SPD 4×4 | ≈ 2.36 | Todos convergen |
| ESTRÉS | Tridiagonal SPD 20×20 (diag. 2.05) | ≈ 55.7 | Más exigente, bien condicionado |
| MAL CONDICIONADO | Hilbert 4×4 (SPD) | ≈ 1.55×10⁴ | Jacobi diverge; residual pequeño ≠ error pequeño |

En todos `b = A·x*` con `x*` conocido, para medir el error real.

## Estructura

```text
simtele-risk/
├── README.md · requirements.txt · pytest.ini · .gitignore
├── db/            schema.sql · diagrama_der.png
├── simtele/       __init__.py · lineal.py
├── web/           app.py · templates/ · static/
└── tests/         test_persistencia.py · test_web.py
```

## Modelo de datos

`caso_estudio` → `escenario` → `simulacion` → `ejecucion_metodo` → `iteracion_resultado` y `resultado_resumen`. Cada simulación guarda `semilla` y `hash_parametros` para reproducirla; `error_rel` es anulable para no dividir por cero.