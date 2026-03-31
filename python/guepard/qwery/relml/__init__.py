import duckdb
from dataclasses import dataclass, field
from typing import Optional
from . import _relml_core


# ---------------------------------------------------------------------------
# The only thing a user needs to define: the SQL task query and what to predict.
# ---------------------------------------------------------------------------
@dataclass
class TaskSpec:
    sql: str                          # any SPJA query — result becomes training table
    task_table_name: str              # name for the materialized result
    target_column: str                # which column in the result is the label
    task_type: str = "binary_classification"  # or "regression" / "multiclass_classification"
    label_transform: dict = field(default_factory=lambda: {"kind": "threshold", "threshold": 0.5})
    split_strategy: str = "random"    # or "temporal"
    time_col: Optional[str] = None    # required when split_strategy="temporal"
    inference_mode: str = "row_based" # or "entity_synthesis"
    inference_agg: str = "none"       # "mean", "fraction", "count"
    entity_refs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DuckDB type string → relml type string
# ---------------------------------------------------------------------------
def _duck_type_to_relml(duck_type: str) -> str:
    t = duck_type.upper()
    if any(x in t for x in ("INT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "HUGEINT")):
        return "numerical"
    if "TIMESTAMP" in t or "DATE" in t:
        return "timestamp"
    if "BOOL" in t or "ENUM" in t:
        return "categorical"
    return "categorical"   # VARCHAR — cardinality check below will refine


def _classify_varchar(values: list) -> str:
    non_null = [v for v in values if v is not None]
    if not non_null:
        return "text"
    unique_ratio = len(set(non_null)) / len(non_null)
    return "categorical" if unique_ratio <= 0.05 else "text"


# ---------------------------------------------------------------------------
# Load a table from DuckDB into the dict format expected by the C++ binding.
# ---------------------------------------------------------------------------
def _load_table_from_conn(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    query: str
) -> dict:
    result = conn.execute(query).fetchall()
    desc   = conn.execute(query).description   # (name, type_code, ...)

    columns = []
    for i, col_desc in enumerate(desc):
        col_name  = col_desc[0]
        duck_type = str(col_desc[1]) if col_desc[1] else "VARCHAR"
        values    = [row[i] for row in result]

        col_type = _duck_type_to_relml(duck_type)
        if col_type == "categorical":
            # DuckDB reports VARCHAR for all strings — refine by cardinality
            if "VARCHAR" in duck_type.upper() or "TEXT" in duck_type.upper():
                col_type = _classify_varchar(values)

        # Convert timestamps to unix seconds (DuckDB returns datetime objects)
        if col_type == "timestamp":
            import datetime
            converted = []
            for v in values:
                if v is None:
                    converted.append(None)
                elif isinstance(v, (datetime.datetime, datetime.date)):
                    converted.append(int(v.timestamp()) if isinstance(v, datetime.datetime)
                                     else int(datetime.datetime(v.year, v.month, v.day).timestamp()))
                else:
                    converted.append(int(v))
            values = converted

        columns.append({
            "name":   col_name,
            "type":   col_type,
            "values": values,
        })

    return {"columns": columns, "pkey_col": None, "time_col": None, "foreign_keys": []}


# ---------------------------------------------------------------------------
# Auto-detect PKs and FKs from DuckDB's information_schema.
# Falls back to naming convention when constraints are absent (e.g. CSV files).
# ---------------------------------------------------------------------------
def _detect_pks(conn: duckdb.DuckDBPyConnection, table_names: list) -> dict:
    pks = {}
    try:
        for name in table_names:
            pragma = conn.execute(f'PRAGMA table_info("{name}")').fetchall()
            for row in pragma:
                # row: (cid, name, type, notnull, dflt, pk)
                if row[5] and int(row[5]) > 0:
                    pks[name] = row[1]
                    break
    except Exception:
        pass

    # Naming convention fallback
    for name in table_names:
        if name in pks:
            continue
        try:
            cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{name}")').fetchall()]
        except Exception:
            continue
        singular = name.rstrip("s")
        candidates = ["id", f"{name}Id", f"{name}_id", f"{singular}Id", f"{singular}_id"]
        for c in candidates:
            if c.lower() in [x.lower() for x in cols]:
                # Verify uniqueness
                try:
                    count_q = f'SELECT COUNT(*) = COUNT(DISTINCT "{c}") FROM "{name}"'
                    if conn.execute(count_q).fetchone()[0]:
                        pks[name] = c
                        break
                except Exception:
                    pass
    return pks


def _detect_fks(conn: duckdb.DuckDBPyConnection, table_names: list) -> dict:
    """Returns {src_table: [(src_col, dst_table), ...]}"""
    fks = {t: [] for t in table_names}
    try:
        rows = conn.execute("""
            SELECT
                kcu.table_name,
                kcu.column_name,
                ccu.table_name AS referenced_table
            FROM information_schema.referential_constraints rc
            JOIN information_schema.key_column_usage kcu
                ON rc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage ccu
                ON rc.unique_constraint_name = ccu.constraint_name
        """).fetchall()
        for src_table, src_col, dst_table in rows:
            if src_table in fks:
                fks[src_table].append((src_col, dst_table))
    except Exception:
        pass
    return fks


def _detect_time_cols(conn: duckdb.DuckDBPyConnection, table_names: list) -> dict:
    time_cols = {}
    TIME_NAMES = {"timestamp", "date", "datetime", "created_at", "updated_at",
                  "t_dat", "event_time", "event_date"}
    for name in table_names:
        try:
            pragma = conn.execute(f'PRAGMA table_info("{name}")').fetchall()
        except Exception:
            continue
        for row in pragma:
            col_name  = row[1]
            col_type  = str(row[2]).upper()
            if "TIMESTAMP" in col_type or "DATE" in col_type:
                time_cols[name] = col_name
                break
            if col_name.lower() in TIME_NAMES:
                time_cols[name] = col_name
                break
    return time_cols


# ---------------------------------------------------------------------------
# The main public function.
# ---------------------------------------------------------------------------
def train(
    conn: duckdb.DuckDBPyConnection,
    task: TaskSpec,
    channels:   int   = 64,
    gnn_layers: int   = 2,
    hidden:     int   = 64,
    dropout:    float = 0.3,
    lr:         float = 3e-4,
    epochs:     int   = 30,
    batch_size: int   = 0,
):
    """
    Train a RelML model.

    Parameters
    ----------
    conn   : open DuckDB connection with the data already loaded
    task   : TaskSpec describing the SQL query and prediction target
    ...    : hyperparameters passed directly to the C++ Trainer

    Returns
    -------
    TrainedModel with predict_all() and predict_entity() methods
    """
    # 1. Materialize the task query as a named table in DuckDB
    conn.execute(f"""
        CREATE OR REPLACE TABLE "{task.task_table_name}" AS ({task.sql})
    """)

    # 2. Discover all tables now present in the connection
    all_tables = [
        r[0] for r in conn.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
        """).fetchall()
    ]

    # 3. Auto-detect schema metadata from DuckDB
    pks       = _detect_pks(conn, all_tables)
    fks       = _detect_fks(conn, all_tables)
    time_cols = _detect_time_cols(conn, all_tables)

    # Apply user-specified time_col for the task table
    if task.time_col:
        time_cols[task.task_table_name] = task.time_col

    # 4. Load all tables into the dict format
    db_tables = {}
    for tname in all_tables:
        tdict = _load_table_from_conn(conn, tname, f'SELECT * FROM "{tname}"')
        tdict["pkey_col"]    = pks.get(tname)
        tdict["time_col"]    = time_cols.get(tname)
        tdict["foreign_keys"] = fks.get(tname, [])

        # Annotate the time_col in the column list so C++ encoder skips it
        if tdict["time_col"]:
            for col in tdict["columns"]:
                if col["name"] == tdict["time_col"]:
                    col["type"] = "timestamp"

        db_tables[tname] = tdict

    # 5. Build the task dict for the C++ binding
    task_dict = {
        "target_table":   task.task_table_name,
        "target_column":  task.target_column,
        "task_type":      task.task_type,
        "label_transform": task.label_transform,
        "split_strategy": task.split_strategy,
        "time_col":       task.time_col,
        "inference_mode": task.inference_mode,
        "inference_agg":  task.inference_agg,
        "entity_refs":    task.entity_refs,
    }

    # 6. Call into C++
    return _relml_core.train(
        db_tables, task_dict,
        channels, gnn_layers, hidden,
        dropout, lr, epochs, batch_size
    )
