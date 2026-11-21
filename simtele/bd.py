"""Persistencia (SQLite): guarda una simulación completa (parámetros, ejecuciones, iteraciones, resultados y errores),
la recupera tras reiniciar la aplicación y permite relanzarla comparando con la original."""
import hashlib
import json
import os
import sqlite3

from .casos import CASOS, POR_NOMBRE
from .casos.comun import limpio

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_PATH = os.path.join(RAIZ, "db", "schema.sql")
MAX_FILAS = 200
TOL_RELANZAR = 1e-12


def conectar(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def inicializar(conn, schema_path=SCHEMA_PATH):
    if not conn.execute("SELECT name FROM sqlite_master WHERE name='caso_estudio'").fetchone():
        with open(schema_path, encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.commit()


def _caso_id(conn, nombre):
    r = conn.execute("SELECT id FROM caso_estudio WHERE nombre=?", (nombre,)).fetchone()
    return r["id"] if r else conn.execute("INSERT INTO caso_estudio(nombre, modelo_version) VALUES (?,?)", (nombre, "v2.0")).lastrowid


def guardar_simulacion(conn, caso_key, tipo, params, resultado, usuario="web"):
    modulo = CASOS[caso_key]
    pj = json.dumps(limpio(dict(caso=caso_key, params=params)), sort_keys=True)
    caso_id = _caso_id(conn, modulo.NOMBRE)
    esc = conn.execute("SELECT id FROM escenario WHERE caso_id=? AND tipo=? AND parametros_json=?", (caso_id, tipo, pj)).fetchone()
    esc_id = esc["id"] if esc else conn.execute(
        "INSERT INTO escenario(caso_id,tipo,parametros_json,fuente_datos) VALUES (?,?,?,?)",
        (caso_id, tipo, pj, "sintético (datos y parámetros supuestos declarados)")).lastrowid
    sim_id = conn.execute(
        "INSERT INTO simulacion(escenario_id,usuario_equipo,estado,semilla,hash_parametros) VALUES (?,?,?,?,?)",
        (esc_id, usuario, "COMPLETADA", int(params.get("semilla", 42)), hashlib.sha256(pj.encode()).hexdigest()[:16])).lastrowid
    for e in resultado["ejecuciones"]:
        ej = conn.execute(
            "INSERT INTO ejecucion_metodo(simulacion_id,metodo,tolerancia,paso,max_iter,convergio,tiempo_ms) VALUES (?,?,?,?,?,?,?)",
            (sim_id, e["metodo"], e["tolerancia"], e["paso"], e["max_iter"], int(e["convergio"]), e["tiempo_ms"])).lastrowid
        for f in e["iteraciones"][:MAX_FILAS]:
            conn.execute("INSERT INTO iteracion_resultado(ejecucion_id,numero,aproximacion_json,residual,error_abs,error_rel) VALUES (?,?,?,?,?,?)",
                         (ej, f["numero"], json.dumps(f["aprox"]), f["residual"], f["error_abs"], f["error_rel"]))
        conn.execute("INSERT INTO resultado_resumen(ejecucion_id,referencia,valor_json,metrica_error_json,interpretacion) VALUES (?,?,?,?,?)",
                     (ej, json.dumps(e["referencia"]), json.dumps(e["valor"]), json.dumps(e["metricas"]), e["texto"]))
    conn.commit()
    return sim_id


def cargar_simulacion(conn, sim_id):
    s = conn.execute("""SELECT s.id, s.fecha, s.estado, s.usuario_equipo, s.semilla, s.hash_parametros, e.tipo, e.parametros_json,
                               c.nombre AS caso FROM simulacion s JOIN escenario e ON e.id=s.escenario_id
                        JOIN caso_estudio c ON c.id=e.caso_id WHERE s.id=?""", (sim_id,)).fetchone()
    if not s:
        return None
    d = dict(s)
    pj = json.loads(d.pop("parametros_json"))
    d["caso_key"], d["params"] = pj["caso"], pj["params"]
    d["ejecuciones"] = []
    for em in conn.execute("SELECT * FROM ejecucion_metodo WHERE simulacion_id=? ORDER BY id", (sim_id,)):
        its = [dict(numero=r["numero"], aprox=json.loads(r["aproximacion_json"]), residual=r["residual"],
                    error_abs=r["error_abs"], error_rel=r["error_rel"])
               for r in conn.execute("SELECT * FROM iteracion_resultado WHERE ejecucion_id=? ORDER BY numero", (em["id"],))]
        rr = conn.execute("SELECT * FROM resultado_resumen WHERE ejecucion_id=?", (em["id"],)).fetchone()
        d["ejecuciones"].append(dict(id=em["id"], metodo=em["metodo"], tolerancia=em["tolerancia"], paso=em["paso"],
                                     max_iter=em["max_iter"], convergio=bool(em["convergio"]), tiempo_ms=em["tiempo_ms"],
                                     iteraciones=its, referencia=json.loads(rr["referencia"] or "{}"),
                                     valor=json.loads(rr["valor_json"] or "{}"), metricas=json.loads(rr["metrica_error_json"] or "{}"),
                                     texto=rr["interpretacion"]))
    return d


def listar_historial(conn, q=""):
    sql = """SELECT s.id, s.fecha, s.estado, s.usuario_equipo, e.tipo, c.nombre AS caso,
                    (SELECT group_concat(metodo, ', ') FROM ejecucion_metodo WHERE simulacion_id=s.id) AS metodos
             FROM simulacion s JOIN escenario e ON e.id=s.escenario_id JOIN caso_estudio c ON c.id=e.caso_id"""
    args = []
    if q:
        sql += """ WHERE c.nombre LIKE ? OR e.tipo LIKE ? OR s.usuario_equipo LIKE ? OR s.fecha LIKE ? OR CAST(s.id AS TEXT)=?
                   OR s.id IN (SELECT simulacion_id FROM ejecucion_metodo WHERE metodo LIKE ?)"""
        like = f"%{q}%"; args = [like, like, like, like, q.strip(), like]
    return [dict(r) for r in conn.execute(sql + " ORDER BY s.id DESC", args)]


def ultimas_por_escenario(conn):
    """{(caso_key, tipo): id de la simulación más reciente}."""
    out = {}
    for r in conn.execute("""SELECT s.id, e.tipo, c.nombre FROM simulacion s JOIN escenario e ON e.id=s.escenario_id
                             JOIN caso_estudio c ON c.id=e.caso_id ORDER BY s.id"""):
        k = POR_NOMBRE.get(r["nombre"])
        if k:
            out[(k, r["tipo"])] = r["id"]
    return out


def _huella(e):
    h = (e.get("valor") or {}).get("huella") or []
    return [float(x) for x in h if x is not None]


def comparar(orig, nuevo):
    """Compara las ejecuciones de dos simulaciones por método. Devuelve [{metodo, diff, coincide}]."""
    out = []
    por_nombre = {e["metodo"]: e for e in nuevo["ejecuciones"]}
    for a in orig["ejecuciones"]:
        b = por_nombre.get(a["metodo"])
        if b is None:
            out.append(dict(metodo=a["metodo"], diff=None, coincide=False)); continue
        ha, hb = _huella(a), _huella(b)
        if len(ha) != len(hb):
            out.append(dict(metodo=a["metodo"], diff=None, coincide=False)); continue
        esc = max([1.0] + [abs(x) for x in ha])
        diff = max([abs(x - y) for x, y in zip(ha, hb)] + [0.0]) / esc
        out.append(dict(metodo=a["metodo"], diff=diff, coincide=diff <= TOL_RELANZAR and a["convergio"] == b["convergio"]))
    return out


def relanzar(conn, sim_id, usuario="web"):
    """Recupera los parámetros guardados, repite el cálculo, guarda la nueva simulación y la compara con la original."""
    orig = cargar_simulacion(conn, sim_id)
    mod = CASOS[orig["caso_key"]]
    res = mod.ejecutar(orig["tipo"], orig["params"])
    nuevo_id = guardar_simulacion(conn, orig["caso_key"], orig["tipo"], orig["params"], res,
                                  usuario=f"{usuario} (relanzada de #{sim_id})")
    return nuevo_id, comparar(orig, cargar_simulacion(conn, nuevo_id))
