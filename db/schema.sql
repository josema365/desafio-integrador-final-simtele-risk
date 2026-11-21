-- SIMTELE-RISK · esquema inicial (SQLite; portable a PostgreSQL cambiando AUTOINCREMENT por SERIAL)
PRAGMA foreign_keys = ON;

CREATE TABLE caso_estudio (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre          TEXT NOT NULL UNIQUE,
    modelo_version  TEXT NOT NULL
);

CREATE TABLE escenario (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    caso_id         INTEGER NOT NULL REFERENCES caso_estudio(id),
    tipo            TEXT NOT NULL CHECK (tipo IN ('NORMAL','ESTRES','MAL_CONDICIONADO')),
    parametros_json TEXT NOT NULL,
    fuente_datos    TEXT NOT NULL,
    UNIQUE (caso_id, tipo, parametros_json)
);

CREATE TABLE simulacion (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    escenario_id    INTEGER NOT NULL REFERENCES escenario(id),
    fecha           TEXT NOT NULL DEFAULT (datetime('now')),
    usuario_equipo  TEXT,
    estado          TEXT NOT NULL DEFAULT 'PENDIENTE'
                    CHECK (estado IN ('PENDIENTE','COMPLETADA','ERROR')),
    semilla         INTEGER,
    hash_parametros TEXT
);

CREATE TABLE ejecucion_metodo (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    simulacion_id   INTEGER NOT NULL REFERENCES simulacion(id) ON DELETE CASCADE,
    metodo          TEXT NOT NULL,
    tolerancia      REAL,
    paso            REAL,
    max_iter        INTEGER,
    convergio       INTEGER NOT NULL CHECK (convergio IN (0,1)),
    tiempo_ms       REAL
);

CREATE TABLE iteracion_resultado (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ejecucion_id    INTEGER NOT NULL REFERENCES ejecucion_metodo(id) ON DELETE CASCADE,
    numero          INTEGER NOT NULL,
    aproximacion_json TEXT NOT NULL,
    residual        REAL,
    error_abs       REAL,
    error_rel       REAL,           -- NULL si la referencia es ~0 (evita dividir por cero)
    UNIQUE (ejecucion_id, numero)
);

CREATE TABLE resultado_resumen (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ejecucion_id    INTEGER NOT NULL UNIQUE REFERENCES ejecucion_metodo(id) ON DELETE CASCADE,
    referencia      TEXT,
    valor_json      TEXT,
    metrica_error_json TEXT,
    interpretacion  TEXT
);

CREATE INDEX idx_escenario_caso   ON escenario(caso_id);
CREATE INDEX idx_sim_escenario    ON simulacion(escenario_id);
CREATE INDEX idx_ejec_sim         ON ejecucion_metodo(simulacion_id);
CREATE INDEX idx_iter_ejec        ON iteracion_resultado(ejecucion_id);
