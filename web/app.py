"""SIMTELE-RISK · prototipo web (Semana 1): cuatro módulos visibles con su modelo y sus tres escenarios.
Ejecutar desde la raíz:  python web/app.py  ->  http://127.0.0.1:5000
Sistema estrictamente educativo; datos sintéticos; sin conexión con equipos reales."""
from flask import Flask, abort, render_template

TIPOS = ["NORMAL", "ESTRES", "MAL_CONDICIONADO"]
ETIQUETA = {"NORMAL": "NORMAL", "ESTRES": "ESTRÉS", "MAL_CONDICIONADO": "MAL CONDICIONADO"}

MODULOS = {
    1: dict(nombre="Detención", titulo="Detención de emergencia de cabinas",
            pregunta="¿Cómo varían tiempo y distancia de detención frente a cambios de velocidad, reacción y frenado?",
            modelo="x′ = v,  v′ = −a(v,t).  Analítico (a constante): t_det = v₀/a;  d = v₀²/(2a).",
            metodos="RK2, RK4, trapecio, Simpson, Romberg, bisección, Newton, secante",
            escenarios={"NORMAL": "v₀=5 m/s, a=0.8 m/s², t_r=1 s → t_det=6.25 s, d=15.625 m",
                        "ESTRES": "v₀=6 m/s, a=0.5 m/s², t_r=2 s → t_det=12 s, d_total=48 m (excede 40 m supuestos)",
                        "MAL_CONDICIONADO": "v(t)=v₀(1−t/T)²: raíz doble en t=T (la bisección no ve cambio de signo)"}),
    2: dict(nombre="Deterioro", titulo="Deterioro mecánico y mantenimiento",
            pregunta="¿Cuándo alcanzaría un indicador hipotético de deterioro un umbral de inspección?",
            modelo="D(t) = D₀ + αt + βt²;  resolver D(t) − D_umbral = 0. Calibración: A x = b con A SPD.",
            metodos="Jacobi, Gauss-Seidel, LU, gradiente conjugado (calibración) · bisección, Newton, secante",
            escenarios={"NORMAL": "Tridiagonal SPD 4×4 (diag. 4, fuera −1), κ ≈ 2.36",
                        "ESTRES": "Tridiagonal SPD 20×20 (diag. 2.05, fuera −1), κ ≈ 55.7",
                        "MAL_CONDICIONADO": "Hilbert 4×4 (SPD), κ ≈ 1.55×10⁴"}),
    3: dict(nombre="Oscilaciones", titulo="Oscilaciones por viento",
            pregunta="¿Cómo se modifica la oscilación ante cambios de fuerza externa, amortiguamiento y parámetros?",
            modelo="x′ = v,  v′ = (F(t) − c v − k x)/m.  Libre subamortiguado: x = e^(−ζω₀t)[C₁cos ω_d t + C₂ sin ω_d t].",
            metodos="RK2, RK4, trapecio, Simpson, Romberg",
            escenarios={"NORMAL": "m=1000 kg, k=40000 N/m, ζ=0.05, F=0, x(0)=0.10 m",
                        "ESTRES": "ζ=0.02, F=2000 sin(Ωt) N, Ω=0.9ω₀, T=60 s, pasos mayores",
                        "MAL_CONDICIONADO": "ζ=0.01, Ω=0.99ω₀ (cerca de resonancia): amplitud muy sensible"}),
    4: dict(nombre="Sensores", titulo="Reconstrucción de señales de sensores",
            pregunta="¿Cómo reconstruir registros faltantes y medir la distorsión que producen datos con ruido?",
            modelo="Datos sintéticos y(t) = sin t en [0, 2π] (con ruido); se ocultan nodos para medir el error real.",
            metodos="Lagrange, Newton (diferencias divididas), splines cúbicos, cruce de umbral",
            escenarios={"NORMAL": "9 nodos equiespaciados, sin ruido, 3 nodos ocultos",
                        "ESTRES": "5 nodos y ruido σ=0.05 (semilla 42)",
                        "MAL_CONDICIONADO": "21 nodos equiespaciados: constante de Lebesgue ≈ 1×10⁴"}),
}


def create_app():
    app = Flask(__name__)

    @app.context_processor
    def comunes():
        return dict(MODULOS=MODULOS, ETIQUETA=ETIQUETA, TIPOS=TIPOS)

    @app.route("/")
    def inicio():
        return render_template("index.html")

    @app.route("/modulo/<int:n>")
    def modulo(n):
        if n not in MODULOS:
            abort(404)
        return render_template("modulo.html", n=n, m=MODULOS[n])

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
