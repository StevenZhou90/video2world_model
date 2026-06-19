from __future__ import annotations

import sqlite3
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional


DB_PATH = Path(os.getenv("DB_PATH", Path(__file__).resolve().parent / "world_model.db"))


SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    item_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    bbox_x1 REAL NOT NULL,
    bbox_y1 REAL NOT NULL,
    bbox_x2 REAL NOT NULL,
    bbox_y2 REAL NOT NULL,
    mask_area REAL,
    map_x REAL,
    map_y REAL,
    map_z REAL
);

CREATE TABLE IF NOT EXISTS camera_poses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    pose_json TEXT NOT NULL,
    seen_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_camera_poses_seen
ON camera_poses(seen_at DESC);
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
        _drop_legacy_state_tables(conn)
        conn.executescript(SCHEMA)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def upsert_item(item: dict) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO items (
                item_id, label, confidence,
                bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                mask_area, map_x, map_y, map_z
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(item_id) DO UPDATE SET
                label = excluded.label,
                confidence = excluded.confidence,
                bbox_x1 = excluded.bbox_x1,
                bbox_y1 = excluded.bbox_y1,
                bbox_x2 = excluded.bbox_x2,
                bbox_y2 = excluded.bbox_y2,
                mask_area = excluded.mask_area,
                map_x = excluded.map_x,
                map_y = excluded.map_y,
                map_z = excluded.map_z
            """,
            _item_params(item),
        )


def get_world_items() -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM items
            ORDER BY label ASC, item_id ASC
            """
        ).fetchall()
    return [_row_to_item(row) for row in rows]


def record_camera_pose(source: str, pose_json: str) -> int:
    with connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO camera_poses (source, pose_json, seen_at)
            VALUES (?, ?, ?)
            """,
            (source, pose_json, utc_now_iso()),
        )
        return int(cursor.lastrowid)


def get_latest_camera_pose() -> Optional[dict]:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM camera_poses
            ORDER BY seen_at DESC, id DESC
            LIMIT 1
            """
        ).fetchone()
    if row is None:
        return None
    return {"id": row["id"], "source": row["source"], "pose_json": row["pose_json"], "seen_at": row["seen_at"]}


def _item_params(item: dict) -> tuple:
    bbox = item["bbox"]
    map_position = item.get("map_position") or {}
    return (
        item["item_id"],
        item["label"],
        item["confidence"],
        bbox[0],
        bbox[1],
        bbox[2],
        bbox[3],
        item.get("mask_area"),
        map_position.get("x"),
        map_position.get("y"),
        map_position.get("z"),
    )


def _row_to_item(row: sqlite3.Row) -> dict:
    map_position = None
    if row["map_x"] is not None or row["map_y"] is not None or row["map_z"] is not None:
        map_position = {"x": row["map_x"], "y": row["map_y"], "z": row["map_z"]}
    return {
        "item_id": row["item_id"],
        "label": row["label"],
        "confidence": round(float(row["confidence"]), 3),
        "bbox": [row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]],
        "mask_area": row["mask_area"],
        "map_position": map_position,
    }


def _drop_legacy_state_tables(conn: sqlite3.Connection) -> None:
    item_columns = _table_columns(conn, "items")
    expected_columns = {
        "item_id",
        "label",
        "confidence",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "mask_area",
        "map_x",
        "map_y",
        "map_z",
    }
    if item_columns and item_columns != expected_columns:
        conn.execute("DROP TABLE IF EXISTS items")
    conn.execute("DROP TABLE IF EXISTS observations")
    conn.execute("DROP TABLE IF EXISTS detections")


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row["name"] for row in rows}
