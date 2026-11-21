# SIMTELE-RISK

**Plataforma educativa de simulación numérica, análisis de errores y persistencia experimental aplicada a escenarios hipotéticos de transporte por cable.**


---

## Tabla de contenidos

1. [Descripción general](#descripción-general)
2. [Casos de estudio y métodos numéricos](#casos-de-estudio-y-métodos-numéricos)
3. [Criterio metodológico de evaluación](#criterio-metodológico-de-evaluación)
4. [Estructura del proyecto](#estructura-del-proyecto)
5. [Requisitos e instalación](#requisitos-e-instalación)
6. [Ejecución de la aplicación web](#ejecución-de-la-aplicación-web)
7. [Ejecución de pruebas automatizadas](#ejecución-de-pruebas-automatizadas)
8. [Persistencia y reproducibilidad (SQLite)](#persistencia-y-reproducibilidad-sqlite)
9. [Autoría y créditos](#autoría-y-créditos)

---

## Descripción general

**SIMTELE-RISK** es una aplicación web desarrollada en Python (Flask) con un núcleo matemático modular (`simtele/`) y persistencia relacional en SQLite (`db/`). Permite ejecutar, comparar y auditar doce experimentos numéricos organizados en una matriz de **4 módulos de estudio $\times$ 3 escenarios** (`NORMAL`, `ESTRÉS` y `MAL CONDICIONADO`), contrastando cada método contra su solución analítica de referencia.

---

## Casos de estudio y métodos numéricos

| Módulo | Fenómeno modelado | Ecuación / Modelo base | Métodos numéricos implementados |
| :--- | :--- | :--- | :--- |
| **M1. Detención de emergencia** | Cinemática de reacción y frenado hasta reposo | $x' = v,\quad v' = -a(v,t)$ con $a = a_0 + bv$ | **EDO:** RK2 (Heun), RK4<br>**Raíces:** Bisección, Newton-Raphson, Secante<br>**Integración:** Trapecio, Simpson 1/3, Romberg |
| **M2. Deterioro y mantenimiento** | Evolución cuadrática de desgaste y calibración lineal | $D(t) = D_0 + \alpha t + \beta t^2 = D_u$<br>$A\mathbf{x} = \mathbf{b}$ ($A$ SPD) | **Raíces:** Bisección, Newton-Raphson, Secante<br>**Sistemas lineales:** Jacobi, Gauss-Seidel, LU (Doolittle), Gradiente Conjugado |
| **M3. Oscilaciones por viento** | Vibración libre y forzada de 1 grado de libertad | $mx'' + cx' + kx = F_0\sin(\Omega t)$<br>Energía acumulada: $\int_0^T x(t)^2\,dt$ | **EDO:** RK2 (Heun), RK4<br>**Integración:** Trapecio, Simpson 1/3, Romberg |
| **M4. Reconstrucción de sensores** | Reconstrucción de registros ocultos en telemetría con ruido | $y(t) = \sin(t) + \varepsilon,\quad \varepsilon \sim \mathcal{N}(0, \sigma^2)$ | **Interpolación:** Lagrange, Newton (diferencias divididas), Splines cúbicos naturales<br>**Raíces:** Cruce de umbral |

---

## Criterio metodológico de evaluación

Todas las simulaciones reportan métricas transversales de error ($E_{\text{abs}}$, $E_{\text{rel}}$, $E_{\text{pct}}$, residual relativo, número de condición $\kappa_2(A)$ y constante de Lebesgue $\Lambda_n$) bajo cuatro categorías estrictamente diferenciadas:

1. **Riesgo físico modelado:** el resultado supera un umbral operativo del modelo (ej. distancia de frenado superior a la longitud disponible), aun cuando el error numérico sea despreciable.
2. **Alta exigencia computacional:** problema bien condicionado que requiere mayor dimensión o paso más fino para converger.
3. **Mal condicionamiento:** hipersensibilidad intrínseca del modelo ante mínimas perturbaciones en los datos ($\kappa(A) \gg 1$, raíz doble o nodos equiespaciados densos con ruido).
4. **Falta de convergencia / inestabilidad del algoritmo:** falla propia del método numérico (ej. divergencia de Jacobi cuando $\rho(T_J) > 1$ o amplificación de RK2 con paso grueso).

---

## Estructura del proyecto

```text
simtele-risk-Desafio/
├── .gitignore
├── pytest.ini
├── README.md
├── requirements.txt
├── db/
│   ├── diagrama_der.png              # Diagrama Entidad-Relación (6 tablas)
│   └── schema.sql                    # Esquema SQLite portable a PostgreSQL
├── docs/
│   ├── informe_semana1/              # Avance técnico PDF y figuras (Semana 1)
│   └── informe_semana2/              # Informe de experimentos, fallos y figuras (Semana 2)
├── scripts/
│   └── generar_informe_semana2.py    # Script reproducible de tablas y figuras
├── simtele/
│   ├── __init__.py
│   ├── bd.py                         # Capa de persistencia en SQLite
│   ├── edo.py                        # Runge-Kutta 2 (Heun) y Runge-Kutta 4
│   ├── errores.py                    # Métricas transversales de error y sensibilidad
│   ├── graficos.py                   # Generación de curvas de solución y convergencia
│   ├── integracion.py                # Trapecio compuesto, Simpson 1/3 y Romberg
│   ├── interpolacion.py              # Lagrange, Newton y Splines cúbicos naturales
│   ├── lineal.py                     # Jacobi, Gauss-Seidel, LU y Gradiente Conjugado
│   ├── raices.py                     # Bisección, Newton-Raphson y Secante
│   └── casos/                        # Orquestación de los 4 módulos y 12 escenarios
│       ├── __init__.py
│       ├── comun.py
│       ├── m1_detencion.py
│       ├── m2_deterioro.py
│       ├── m3_oscilaciones.py
│       └── m4_sensores.py
├── tests/                            # Suite de pruebas unitarias e integración
│   ├── test_casos.py
│   ├── test_lineal.py
│   ├── test_metodos_numericos.py
│   ├── test_persistencia.py
│   ├── test_persistencia_casos.py
│   └── test_web.py
└── web/                              # Aplicación web Flask
    ├── __init__.py
    ├── app.py
    ├── static/
    │   └── estilo.css
    └── templates/
        ├── base.html
        ├── comparacion.html
        ├── historial.html
        ├── index.html
        ├── modulo.html
        ├── relanzada.html
        └── simulacion.html
```

---

## Requisitos e instalación

* **Python:** 3.10 o superior.
* **Dependencias:** `numpy`, `flask`, `matplotlib`, `pytest` (especificadas en `requirements.txt`).

### Pasos de instalación

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/josema365/desafio-integrador-final-simtele-risk.git
   cd desafio-integrador-final-simtele-risk
   ```

2. **Crear y activar un entorno virtual:**
   * **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
     *(Si PowerShell restringe la ejecución de scripts, habilítalos previamente con `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`).*

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Ejecución de la aplicación web

Desde la carpeta raíz del proyecto (con el entorno virtual activo), inicia el servidor local de Flask:

```bash
python -m web.app
```

Abre en tu navegador la dirección:

```text
http://127.0.0.1:5000
```

### Funcionalidades disponibles en la interfaz
* **Módulos individuales (`Detención`, `Deterioro`, `Oscilaciones`, `Sensores`):** selección de escenario (`NORMAL`, `ESTRÉS`, `MAL CONDICIONADO`), edición personalizada de parámetros, tablas de iteraciones/refinamiento, gráficos de solución y convergencia, y conclusiones diferenciadas.
* **Panel general de comparación (`/comparacion`):** matriz consolidada $3 \times 4$ con los resultados de los 12 escenarios y botón de ejecución masiva.
* **Historial (`/historial`):** consulta de simulaciones persistidas en SQLite, inspección detallada y **relanzamiento determinista** para verificar reproducibilidad.

---

## Ejecución de pruebas automatizadas

Para validar todos los métodos numéricos contra sus soluciones analíticas, los 12 escenarios, las rutas web y el ciclo completo de base de datos, ejecuta:

```bash
pytest
```

Si deseas ejecutar únicamente las pruebas de persistencia en SQLite:

```bash
pytest tests/test_persistencia.py tests/test_persistencia_casos.py
```

---

## Persistencia y reproducibilidad (SQLite)

El esquema relacional (`db/schema.sql`) consta de **6 tablas**: `caso_estudio`, `escenario`, `simulacion`, `ejecucion_metodo`, `iteracion_resultado` y `resultado_resumen`.

* **Inicialización automática:** al ejecutar cualquier simulación desde la web por primera vez, `simtele/bd.py` crea automáticamente el archivo SQLite local a partir de `db/schema.sql` (excluido del control de versiones mediante `.gitignore`).
* **Reproducibilidad:** cada simulación registra su `semilla` aleatoria y su `hash_parametros` junto con la restricción `UNIQUE(caso_id, tipo, parametros_json)`, permitiendo recuperar cualquier experimento histórico y relanzarlo desde la interfaz web con diferencia máxima $0$.

---

## Autoría y créditos

* **Estudiantes:**
  * Jose Manuel Huanca Tito
  * Jhonatan Pongo Gavincha
* **Mención:** Ingeniería de Sistemas / Desarrollo de Software e Innovación Tecnológica
* **Docente:** Lic. Brigida Alexandra Carvajal Blanco
* **Asignatura:** Métodos Numéricos — Desafío Integrador Final (Noviembre 2026)