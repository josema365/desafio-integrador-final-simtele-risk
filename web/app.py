"""SIMTELE-RISK · aplicación web (Semana 2).  Ejecutar desde la raíz:  python web/app.py  ->  http://127.0.0.1:5000
Sistema estrictamente educativo; datos sintéticos; sin conexión con equipos de control reales."""
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

from flask import Flask, abort, redirect, render_template, request, url_for  # noqa: E402

from simtele import bd  # noqa: E402
from simtele.casos import CASOS, CORTO, ETIQUETAS, NUMERO, ORDEN, TIPOS  # noqa: E402
from simtele.casos.comun import fmt, parsear  # noqa: E402
from simtele.graficos import graficos_de_simulacion  # noqa: E402

ICONOS = {"M1": "", "M2": "", "M3": "", "M4": ""}
DESC = {
    ("M1", "NORMAL"): "Frenado con desaceleración constante: caso analítico de la consigna (t_det = v₀/a, d = v₀²/2a).",
    ("M1", "ESTRES"): "Frenado más exigente con término variable a = a₀ + b·v, mayor reacción y pasos gruesos; la distancia total supera la disponible supuesta.",
    ("M1", "MAL_CONDICIONADO"): "v(t) = v₀(1 − t/T)²: raíz doble en t = T (la bisección no ve cambio de signo; Newton y secante pierden velocidad).",
    ("M2", "NORMAL"): "Deterioro cuadrático con raíz simple; calibración con matriz tridiagonal SPD 4×4 (κ ≈ 2.36).",
    ("M2", "ESTRES"): "Mantenimiento hipotéticamente retrasado (deterioro más rápido) y calibración de 20×20 (κ ≈ 55.7).",
    ("M2", "MAL_CONDICIONADO"): "Umbral igual al máximo del indicador (raíz doble) y calibración tipo Hilbert 4×4 (κ ≈ 1.55×10⁴).",
    ("M3", "NORMAL"): "Vibración libre subamortiguada con solución analítica.",
    ("M3", "ESTRES"): "Viento periódico F(t) = F₀ sin(Ωt), Ω = 0.9ω₀, ζ pequeño y pasos gruesos (RK2 se vuelve inestable).",
    ("M3", "MAL_CONDICIONADO"): "Cerca de resonancia (Ω = 0.99ω₀, ζ = 0.01): la amplitud es muy sensible a Ω y ζ.",
    ("M4", "NORMAL"): "9 nodos equiespaciados sin ruido; se ocultan 3 nodos interiores.",
    ("M4", "ESTRES"): "5 nodos y ruido σ = 0.05: mayor espaciamiento y distorsión.",
    ("M4", "MAL_CONDICIONADO"): "21 nodos equiespaciados con ruido σ = 10⁻³: el polinomio amplifica el ruido (constante de Lebesgue alta).",
}
CLAVES_ERROR = ("error_abs_final", "error_max", "error_abs", "error_oculto_max")


def create_app(db_path=None):
    app = Flask(__name__)
    app.config["DB_PATH"] = db_path or os.environ.get("SIMTELE_DB", os.path.join(RAIZ, "db", "simtele.db"))

    def conn():
        c = bd.conectar(app.config["DB_PATH"]); bd.inicializar(c); return c

    def corto(a):
        if isinstance(a, dict):
            return ", ".join(f"{k}={fmt(v, 5) if isinstance(v, (int, float)) and not isinstance(v, bool) else v}" for k, v in a.items())
        if isinstance(a, list):
            return "(" + ", ".join(fmt(v, 5) for v in a[:5]) + (", …)" if len(a) > 5 else ")")
        return fmt(a, 6) if isinstance(a, (int, float)) else str(a)

    app.jinja_env.filters["f"] = lambda v, d=4: fmt(v, d)
    app.jinja_env.filters["corto"] = corto
    app.jinja_env.filters["abrev"] = lambda s, n=26: s if len(str(s)) <= n else str(s)[:n] + "…"

    @app.context_processor
    def comunes():
        return dict(CASOS=CASOS, ORDEN=ORDEN, NUMERO=NUMERO, ETIQUETAS=ETIQUETAS, TIPOS=TIPOS, ICONOS=ICONOS, CORTO=CORTO)

    def clave(n):
        k = f"M{n}"
        if k not in CASOS:
            abort(404)
        return k

    @app.route("/")
    def inicio():
        return render_template("index.html")

    def _form(k, tipo, valores=None, errores=None):
        m = CASOS[k]
        vals = valores if valores is not None else m.defaults(tipo)
        return render_template("modulo.html", k=k, n=NUMERO[k], m=m, tipo=tipo, spec=m.spec(tipo), vals=vals,
                               errores=errores or [], desc=DESC[(k, tipo)])

    @app.route("/modulo/<int:n>")
    def modulo(n):
        k = clave(n)
        tipo = request.args.get("tipo", "NORMAL")
        if tipo not in TIPOS:
            abort(404)
        return _form(k, tipo)

    @app.route("/modulo/<int:n>/ejecutar", methods=["POST"])
    def ejecutar(n):
        k = clave(n); m = CASOS[k]
        tipo = request.form.get("tipo", "NORMAL")
        if tipo not in TIPOS:
            abort(400)
        params, errores = parsear(m.spec(tipo), request.form, m.defaults(tipo))
        errores = errores + m.validar(tipo, params)   # rangos y reglas del dominio también cuando hay errores de formato
        if errores:
            return _form(k, tipo, params, errores), 400
        res = m.ejecutar(tipo, params)
        c = conn()
        try:
            sim = bd.guardar_simulacion(c, k, tipo, params, res, usuario=request.form.get("usuario") or "web")
        finally:
            c.close()
        return redirect(url_for("simulacion", sim_id=sim))

    @app.route("/simulacion/<int:sim_id>")
    def simulacion(sim_id):
        c = conn()
        try:
            s = bd.cargar_simulacion(c, sim_id)
        finally:
            c.close()
        if not s:
            abort(404)
        m = CASOS[s["caso_key"]]
        ejs = {e["metodo"]: e for e in s["ejecuciones"]}
        for e in s["ejecuciones"]:
            e["principal"] = next((e["metricas"][q] for q in CLAVES_ERROR if e["metricas"].get(q) is not None), None)
            e["pasos"] = e["metricas"].get("iteraciones") or e["metricas"].get("n") or (e["iteraciones"][-1]["aprox"].get("n") if e["iteraciones"] and isinstance(e["iteraciones"][-1]["aprox"], dict) and "n" in e["iteraciones"][-1]["aprox"] else None)
        return render_template("simulacion.html", s=s, m=m, n=NUMERO[s["caso_key"]], interp=m.interpretar(s["tipo"], s["params"], ejs),
                               graficos=graficos_de_simulacion(s["ejecuciones"]))

    @app.route("/simulacion/<int:sim_id>/relanzar", methods=["POST"])
    def relanzar(sim_id):
        c = conn()
        try:
            if not bd.cargar_simulacion(c, sim_id):
                abort(404)
            nuevo, comp = bd.relanzar(c, sim_id)
        finally:
            c.close()
        return render_template("relanzada.html", original=sim_id, nuevo=nuevo, comp=comp)

    @app.route("/historial")
    def historial():
        q = request.args.get("q", "").strip()
        c = conn()
        try:
            filas = bd.listar_historial(c, q)
        finally:
            c.close()
        return render_template("historial.html", filas=filas, q=q)

    @app.route("/comparacion")
    def comparacion():
        c = conn()
        try:
            ult = bd.ultimas_por_escenario(c)
            celdas = {}
            for (k, t), sid in ult.items():
                s = bd.cargar_simulacion(c, sid)
                ejs = {e["metodo"]: e for e in s["ejecuciones"]}
                nconv = sum(1 for e in s["ejecuciones"] if not e["convergio"])
                celdas[(k, t)] = dict(id=sid, titular=CASOS[k].titular(t, s["params"], ejs), fallos=nconv, total=len(ejs))
        finally:
            c.close()
        return render_template("comparacion.html", celdas=celdas)

    @app.route("/ejecutar-todo", methods=["POST"])
    def ejecutar_todo():
        c = conn()
        try:
            for k in ORDEN:
                for t in TIPOS:
                    p = CASOS[k].defaults(t)
                    bd.guardar_simulacion(c, k, t, p, CASOS[k].ejecutar(t, p), usuario="ejecución global")
        finally:
            c.close()
        return redirect(url_for("comparacion"))

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
