"""Prueba obligatoria: ejecutar -> guardar -> cerrar -> reabrir -> recuperar -> repetir -> comparar.
Los parámetros (A, b, x*) se guardan EN la base y se recuperan desde ella (no dependen del código ni de memoria).
Ejecutar desde la raíz:  pytest -v   o   python tests/test_persistencia.py"""
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import time

import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from simtele.lineal import jacobi, gauss_seidel, cg, lu_solve, tridiag, hilbert  # noqa: E402

SCHEMA_PATH = os.path.join(BASE_DIR, "db", "schema.sql")
TOL = 1e-10
MAXIT = 50000
TOL_COINCIDENCIA = 1e-12
METODOS = ["Jacobi", "Gauss-Seidel", "Gradiente Conjugado", "LU"]


def construir_escenarios():
    A1 = tridiag(4, 4.0, -1.0);   x1 = np.array([1.0, 2.0, 3.0, 4.0])
    A2 = tridiag(20, 2.05, -1.0); x2 = np.ones(20)
    A3 = hilbert(4);              x3 = np.ones(4)
    return {"NORMAL": (A1, A1 @ x1, x1), "ESTRES": (A2, A2 @ x2, x2), "MAL_CONDICIONADO": (A3, A3 @ x3, x3)}


def ejecutar_desde_nucleo(metodo, A, b, xs, tol=TOL, maxit=MAXIT):
    t0 = time.perf_counter()
    if metodo == "LU":
        x, _, _, _ = lu_solve(A, b)
        r = np.linalg.norm(b - A @ x); e = float(np.linalg.norm(x - xs))
        H = [dict(k=1, x=x.copy(), res=float(r / np.linalg.norm(b)), eabs=e, erel=e / np.linalg.norm(xs))]
        return x, True, H, (time.perf_counter() - t0) * 1000
    f = {"Jacobi": jacobi, "Gauss-Seidel": gauss_seidel, "Gradiente Conjugado": cg}[metodo]
    H, conv = f(A, b, xs, tol=tol, maxit=maxit)
    return H[-1]["x"], conv, H[:200], (time.perf_counter() - t0) * 1000


def guardar(db_path):
    conn = sqlite3.connect(db_path)
    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        cur = conn.cursor()
        caso_id = cur.execute("INSERT INTO caso_estudio (nombre, modelo_version) VALUES (?, ?)",
                              ("M2. Deterioro y mantenimiento (calibración lineal)", "v1.0")).lastrowid
        for tipo, (A, b, xs) in construir_escenarios().items():
            params = json.dumps({"A": A.tolist(), "b": b.tolist(), "x_ref": xs.tolist(),
                                 "tol": TOL, "maxit": MAXIT}, sort_keys=True)
            esc_id = cur.execute("INSERT INTO escenario (caso_id, tipo, parametros_json, fuente_datos) VALUES (?,?,?,?)",
                                 (caso_id, tipo, params, "sintetico_semana1: b = A x_ref")).lastrowid
            sim_id = cur.execute("INSERT INTO simulacion (escenario_id, usuario_equipo, estado, semilla, hash_parametros) "
                                 "VALUES (?,?,?,?,?)", (esc_id, "equipo_verificacion", "COMPLETADA", 42,
                                                        hashlib.sha256(params.encode()).hexdigest()[:16])).lastrowid
            for metodo in METODOS:
                sol, conv, hist, t_ms = ejecutar_desde_nucleo(metodo, A, b, xs)
                ej = cur.execute("INSERT INTO ejecucion_metodo (simulacion_id, metodo, tolerancia, paso, max_iter, "
                                 "convergio, tiempo_ms) VALUES (?,?,?,?,?,?,?)",
                                 (sim_id, metodo, TOL, None, MAXIT, int(conv), t_ms)).lastrowid
                for reg in hist:
                    cur.execute("INSERT INTO iteracion_resultado (ejecucion_id, numero, aproximacion_json, residual, "
                                "error_abs, error_rel) VALUES (?,?,?,?,?,?)",
                                (ej, reg["k"], json.dumps(reg["x"].tolist()), reg["res"], reg["eabs"], reg["erel"]))
                cur.execute("INSERT INTO resultado_resumen (ejecucion_id, referencia, valor_json, metrica_error_json, "
                            "interpretacion) VALUES (?,?,?,?,?)",
                            (ej, json.dumps(xs.tolist()), json.dumps(sol.tolist()),
                             json.dumps({"error_abs": hist[-1]["eabs"], "error_rel": hist[-1]["erel"],
                                         "residual": hist[-1]["res"]}),
                             "Verificación Semana 1 usando simtele.lineal"))
        conn.commit()
    finally:
        conn.close()   # cierre explícito


def recuperar_y_comparar(db_path):
    conn = sqlite3.connect(db_path)
    try:
        filas = conn.execute(
            """SELECT e.tipo, e.parametros_json, s.hash_parametros, em.id, em.metodo, em.convergio, rr.valor_json
               FROM ejecucion_metodo em JOIN simulacion s ON em.simulacion_id = s.id
               JOIN escenario e ON s.escenario_id = e.id JOIN resultado_resumen rr ON rr.ejecucion_id = em.id
               ORDER BY em.id""").fetchall()
        out = []
        for tipo, pj, hash_db, ej, metodo, conv_db, valor_json in filas:
            assert hashlib.sha256(pj.encode()).hexdigest()[:16] == hash_db, f"Parámetros alterados en {tipo}"
            p = json.loads(pj)
            A, b, xs = np.array(p["A"]), np.array(p["b"]), np.array(p["x_ref"])
            its = conn.execute("SELECT numero, aproximacion_json FROM iteracion_resultado WHERE ejecucion_id=? "
                               "ORDER BY numero", (ej,)).fetchall()
            assert its, f"Sin iteraciones guardadas en {tipo}-{metodo}"
            x_rec, conv_rec, H_rec, _ = ejecutar_desde_nucleo(metodo, A, b, xs, p["tol"], p["maxit"])
            xg = np.array(json.loads(valor_json))
            diff = float(np.linalg.norm(xg - x_rec) / max(1.0, np.linalg.norm(xg)))
            k_ult, aprox = its[-1]
            x_k = {h["k"]: h["x"] for h in H_rec}[k_ult]
            diff_it = float(np.linalg.norm(np.array(json.loads(aprox)) - x_k) / max(1.0, np.linalg.norm(x_k)))
            assert bool(conv_db) == conv_rec, f"Cambió la convergencia en {tipo}-{metodo}"
            assert diff < TOL_COINCIDENCIA and diff_it < TOL_COINCIDENCIA, f"Diferencia en {tipo}-{metodo}: {diff}, {diff_it}"
            out.append((tipo, metodo, len(its), bool(conv_db), max(diff, diff_it)))
        return out
    finally:
        conn.close()


def test_ciclo_persistencia(tmp_path):
    db = str(tmp_path / "simtele_test.db")
    guardar(db)
    assert len(recuperar_y_comparar(db)) == 12      # 3 escenarios x 4 métodos


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        ruta = os.path.join(d, "simtele_test.db")
        guardar(ruta)
        res = recuperar_y_comparar(ruta)
    print(f"{'ESCENARIO':<18} | {'METODO':<20} | {'ITER':<5} | {'CONV':<5} | DIF")
    print("-" * 68)
    for t, m, n, c, d_ in res:
        print(f"{t:<18} | {m:<20} | {n:<5} | {str(c):<5} | {d_:.2e}")
    print(f"\nPrueba de persistencia verificada: {len(res)} ejecuciones coinciden (diff < {TOL_COINCIDENCIA:g}).")
