"""
Zero-infrastructure storage: SQLite for queryable run metadata + trend charts,
a JSON blob column for the full run (so nothing is ever lossy), and the
golden dataset/prompts stay as git-tracked files. No servers to stand up.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .config import DATA_DIR
from .eval_runner import EvalRun

DB_PATH = DATA_DIR / "eval_history.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    model TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    overall_pass_rate REAL NOT NULL,
    avg_summary_score REAL NOT NULL,
    avg_latency_ms REAL NOT NULL,
    total_tokens INTEGER NOT NULL,
    raw_json TEXT NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    return conn


def save_run(run: EvalRun) -> None:
    conn = _connect()
    with conn:
        conn.execute(
            """INSERT OR REPLACE INTO runs
               (run_id, timestamp, prompt_version, model, dataset_version,
                overall_pass_rate, avg_summary_score, avg_latency_ms, total_tokens, raw_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run.run_id, run.timestamp, run.prompt_version, run.model, run.dataset_version,
                run.overall_pass_rate, run.avg_summary_score, run.avg_latency_ms, run.total_tokens,
                json.dumps(run.to_dict()),
            ),
        )
    conn.close()


def get_latest_run(exclude_run_id: str | None = None) -> EvalRun | None:
    conn = _connect()
    query = "SELECT raw_json FROM runs"
    params: tuple = ()
    if exclude_run_id:
        query += " WHERE run_id != ?"
        params = (exclude_run_id,)
    query += " ORDER BY timestamp DESC LIMIT 1"
    row = conn.execute(query, params).fetchone()
    conn.close()
    if row is None:
        return None
    return EvalRun.from_dict(json.loads(row[0]))


def get_recent_runs(n: int = 7, exclude_run_id: str | None = None) -> list[EvalRun]:
    conn = _connect()
    query = "SELECT raw_json FROM runs"
    params: tuple = ()
    if exclude_run_id:
        query += " WHERE run_id != ?"
        params = (exclude_run_id,)
    query += " ORDER BY timestamp DESC LIMIT ?"
    rows = conn.execute(query, params + (n,)).fetchall()
    conn.close()
    runs = [EvalRun.from_dict(json.loads(r[0])) for r in rows]
    return list(reversed(runs))  # chronological order


def get_all_runs() -> list[EvalRun]:
    conn = _connect()
    rows = conn.execute("SELECT raw_json FROM runs ORDER BY timestamp ASC").fetchall()
    conn.close()
    return [EvalRun.from_dict(json.loads(r[0])) for r in rows]
