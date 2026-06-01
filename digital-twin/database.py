from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


DB_PATH = Path(__file__).resolve().parent / "detections.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    object_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    seen_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_detections_seen_at
ON detections(seen_at DESC);

CREATE INDEX IF NOT EXISTS idx_detections_object_type
ON detections(object_type);
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_detection(object_type: str, confidence: float) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO detections (object_type, confidence, seen_at)
            VALUES (?, ?, ?)
            """,
            (object_type, confidence, utc_now_iso()),
        )


def get_summary(active_seconds: int = 10, recent_limit: int = 25) -> dict:
    with connect() as conn:
        total = conn.execute("SELECT COUNT(*) AS count FROM detections").fetchone()["count"]

        last_seen = conn.execute(
            """
            SELECT
                object_type,
                MAX(seen_at) AS last_seen,
                COUNT(*) AS detection_count
            FROM detections
            GROUP BY object_type
            ORDER BY MAX(seen_at) DESC
            """
        ).fetchall()

        recent = conn.execute(
            """
            SELECT object_type, confidence, seen_at
            FROM detections
            ORDER BY seen_at DESC, id DESC
            LIMIT ?
            """,
            (recent_limit,),
        ).fetchall()

    now = datetime.now(timezone.utc)
    active_objects = []
    objects = []

    for row in last_seen:
        last_seen_dt = datetime.fromisoformat(row["last_seen"])
        age_seconds = max(0, int((now - last_seen_dt).total_seconds()))
        item = {
            "object_type": row["object_type"],
            "last_seen": row["last_seen"],
            "age_seconds": age_seconds,
            "detection_count": row["detection_count"],
        }
        objects.append(item)
        if age_seconds <= active_seconds:
            active_objects.append(item)

    return {
        "total_detections": total,
        "active_objects": active_objects,
        "objects": objects,
        "recent": [
            {
                "object_type": row["object_type"],
                "confidence": round(float(row["confidence"]), 3),
                "seen_at": row["seen_at"],
            }
            for row in recent
        ],
    }
