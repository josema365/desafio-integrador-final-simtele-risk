import hashlib
import json
import os
import sqlite3
import sys
import time
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from simtele.lineal import (
    jacobi,
    gauss_seidel,
    cg,
    lu_solve,
    tridiag,
    hilbert,
)

SCHEMA_PATH = os.path.join(BASE_DIR, "db", "schema.sql")
DB_PATH = os.path.join(BASE_DIR, "simtele_test.db")


def construir_escenarios():
    A_norm = tridiag(4, 4.0, -1.0)
    x_norm = np.array([1.0, 2.0, 3.0, 4.0])

    A_est = tridiag(20, 2.05, -1.0)
    x_est = np.ones(20)

    A_hilb = hilbert(4)
    x_hilb = np.ones(4)

    return {
        "NORMAL": (A_norm, A_norm @ x_norm, x_norm),
        "ESTRES": (A_est, A_est @ x_est, x_est),
        "MAL_CONDICIONADO": (A_hilb, A_hilb @ x_hilb, x_hilb),
    }


def ejecutar_desde_nucleo(metodo, A, b, xs, tol=1e-10, maxit=50000):
    t0 = time.perf_counter()

    if metodo == "Jacobi":
        H, conv = jacobi(A, b, xs, tol=tol, maxit=maxit)
        t_ms = (time.perf_counter() - t0) * 1000
        return H[-1]["x"], conv, H[:200], t_ms

    if metodo == "Gauss-Seidel":
        H, conv = gauss_seidel(A, b, xs, tol=tol, maxit=maxit)
        t_ms = (time.perf_counter() - t0) * 1000
        return H[-1]["x"], conv, H[:200], t_ms

    if metodo == "Gradiente Conjugado":
        H, conv = cg(A, b, xs, tol=tol, maxit=maxit)
        t_ms = (time.perf_counter() - t0) * 1000
        return H[-1]["x"], conv, H[:200], t_ms

    if metodo == "LU":
        x, _, _, _ = lu_solve(A, b)
        t_ms = (time.perf_counter() - t0) * 1000
        r = b - A @ x
        H = [
            dict(
                k=1,
                x=x.copy(),
                res=float(np.linalg.norm(r) / np.linalg.norm(b)),
                eabs=float(np.linalg.norm(x - xs)),
                erel=float(np.linalg.norm(x - xs) / np.linalg.norm(xs)),
            )
        ]
        return x, True, H, t_ms

    raise ValueError(f"Método no soportado: {metodo}")


def test_ciclo_persistencia():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        conn.executescript(f.read())

    cur = conn.cursor()
    cur.execute(
        "INSERT INTO caso_estudio (nombre, modelo_version) VALUES (?, ?)",
        ("M2. Deterioro y mantenimiento", "v1.0"),
    )
    caso_id = cur.lastrowid

    escenarios = construir_escenarios()
    metodos = ["Jacobi", "Gauss-Seidel", "Gradiente Conjugado", "LU"]

    for tipo, (A, b, xs) in escenarios.items():
        params = json.dumps({"n": len(b), "tol": 1e-10, "maxit": 50000})
        hash_p = hashlib.sha256(params.encode()).hexdigest()[:16]

        cur.execute(
            "INSERT INTO escenario (caso_id, tipo, parametros_json, fuente_datos) VALUES (?, ?, ?, ?)",
            (caso_id, tipo, params, "sintetico_semana1"),
        )
        esc_id = cur.lastrowid

        cur.execute(
            "INSERT INTO simulacion (escenario_id, usuario_equipo, estado, semilla, hash_parametros) VALUES (?, ?, ?, ?, ?)",
            (esc_id, "equipo_verificacion", "COMPLETADA", 42, hash_p),
        )
        sim_id = cur.lastrowid

        for metodo in metodos:
            sol, conv, hist, t_ms = ejecutar_desde_nucleo(metodo, A, b, xs)
            cur.execute(
                """INSERT INTO ejecucion_metodo
                   (simulacion_id, metodo, tolerancia, paso, max_iter, convergio, tiempo_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (sim_id, metodo, 1e-10, None, 50000, int(conv), t_ms),
            )
            ejec_id = cur.lastrowid

            for reg in hist:
                cur.execute(
                    """INSERT INTO iteracion_resultado
                       (ejecucion_id, numero, aproximacion_json, residual, error_abs, error_rel)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        ejec_id,
                        reg["k"],
                        json.dumps(reg["x"].tolist()),
                        reg["res"],
                        reg["eabs"],
                        reg["erel"],
                    ),
                )

            cur.execute(
                """INSERT INTO resultado_resumen
                   (ejecucion_id, referencia, valor_json, metrica_error_json, interpretacion)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    ejec_id,
                    json.dumps(xs.tolist()),
                    json.dumps(sol.tolist()),
                    json.dumps(
                        {
                            "error_abs": float(np.linalg.norm(xs - sol)),
                            "error_rel": float(np.linalg.norm(xs - sol) / np.linalg.norm(xs)),
                        }
                    ),
                    "Verificacion Semana 1 usando simtele.lineal",
                ),
            )

    conn.commit()
    conn.close()

    conn2 = sqlite3.connect(DB_PATH)
    filas = conn2.execute(
        """SELECT e.tipo, em.metodo, em.convergio, rr.valor_json,
                  (SELECT COUNT(*) FROM iteracion_resultado ir WHERE ir.ejecucion_id = em.id)
           FROM ejecucion_metodo em
           JOIN simulacion s ON em.simulacion_id = s.id
           JOIN escenario e ON s.escenario_id = e.id
           JOIN resultado_resumen rr ON rr.ejecucion_id = em.id"""
    ).fetchall()

    assert len(filas) == 12, f"Se esperaban 12 ejecuciones, hay {len(filas)}"
    print(f"{'ESCENARIO':<18} | {'METODO':<20} | {'ITER':<5} | {'CONV':<5} | {'MAX DIF'}")
    print("-" * 68)

    for tipo, metodo, conv_db, valor_json, n_iters in filas:
        A, b, xs = escenarios[tipo]
        x_guardado = np.array(json.loads(valor_json))
        x_recalculado, conv_rec, _, _ = ejecutar_desde_nucleo(metodo, A, b, xs)
        diff = float(np.linalg.norm(x_guardado - x_recalculado))
        assert bool(conv_db) == conv_rec
        assert diff < 1e-12, f"Diferencia fuera de tolerancia en {tipo}-{metodo}: {diff}"
        print(f"{tipo:<18} | {metodo:<20} | {n_iters:<5} | {str(bool(conv_db)):<5} | {diff:.2e}")

    conn2.close()
    os.remove(DB_PATH)
    print("\nPrueba de persistencia verificada con exito (diff < 1e-12 en las 12 ejecuciones).")


if __name__ == "__main__":
    test_ciclo_persistencia()