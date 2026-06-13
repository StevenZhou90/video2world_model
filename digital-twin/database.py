from __future__ import annotations

import sqlite3
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional


DB_PATH = Path(os.getenv("DB_PATH", Path(__file__).resolve().parent / "detections.db"))


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

CREATE TABLE IF NOT EXISTS items (
    item_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    bbox_x1 REAL NOT NULL,
    bbox_y1 REAL NOT NULL,
    bbox_x2 REAL NOT NULL,
    bbox_y2 REAL NOT NULL,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    observation_count INTEGER NOT NULL,
    state TEXT NOT NULL,
    mask_area REAL,
    map_x REAL,
    map_y REAL,
    map_z REAL
);

CREATE INDEX IF NOT EXISTS idx_items_state
ON items(state);

CREATE INDEX IF NOT EXISTS idx_items_last_seen
ON items(last_seen DESC);

CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id TEXT NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    bbox_x1 REAL NOT NULL,
    bbox_y1 REAL NOT NULL,
    bbox_x2 REAL NOT NULL,
    bbox_y2 REAL NOT NULL,
    seen_at TEXT NOT NULL,
    mask_area REAL,
    camera_pose_id INTEGER,
    FOREIGN KEY(item_id) REFERENCES items(item_id)
);

CREATE INDEX IF NOT EXISTS idx_observations_item_seen
ON observations(item_id, seen_at DESC);

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


def upsert_item(item: dict) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO items (
                item_id, label, confidence,
                bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                first_seen, last_seen, observation_count, state,
                mask_area, map_x, map_y, map_z
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(item_id) DO UPDATE SET
                label = excluded.label,
                confidence = excluded.confidence,
                bbox_x1 = excluded.bbox_x1,
                bbox_y1 = excluded.bbox_y1,
                bbox_x2 = excluded.bbox_x2,
                bbox_y2 = excluded.bbox_y2,
                first_seen = excluded.first_seen,
                last_seen = excluded.last_seen,
                observation_count = excluded.observation_count,
                state = excluded.state,
                mask_area = excluded.mask_area,
                map_x = excluded.map_x,
                map_y = excluded.map_y,
                map_z = excluded.map_z
            """,
            _item_params(item),
        )


def update_item_states(items: Iterable[dict]) -> None:
    with connect() as conn:
        conn.executemany(
            """
            UPDATE items
            SET state = ?, last_seen = ?
            WHERE item_id = ?
            """,
            ((item["state"], item["last_seen"], item["item_id"]) for item in items),
        )


def record_observation(item: dict, camera_pose_id: Optional[int] = None) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO observations (
                item_id, label, confidence,
                bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                seen_at, mask_area, camera_pose_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item["item_id"],
                item["label"],
                item["confidence"],
                item["bbox"][0],
                item["bbox"][1],
                item["bbox"][2],
                item["bbox"][3],
                item["last_seen"],
                item.get("mask_area"),
                camera_pose_id,
            ),
        )
        conn.execute(
            """
            INSERT INTO detections (object_type, confidence, seen_at)
            VALUES (?, ?, ?)
            """,
            (item["label"], item["confidence"], item["last_seen"]),
        )


def get_world_items() -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM items
            ORDER BY
                CASE state
                    WHEN 'active' THEN 0
                    WHEN 'stale' THEN 1
                    ELSE 2
                END,
                last_seen DESC
            """
        ).fetchall()
    return [_row_to_item(row) for row in rows]


def get_item_history(item_id: str, limit: int = 100) -> Optional[dict]:
    with connect() as conn:
        item = conn.execute("SELECT * FROM items WHERE item_id = ?", (item_id,)).fetchone()
        if item is None:
            return None
        observations = conn.execute(
            """
            SELECT *
            FROM observations
            WHERE item_id = ?
            ORDER BY seen_at DESC, id DESC
            LIMIT ?
            """,
            (item_id, limit),
        ).fetchall()
    payload = _row_to_item(item)
    payload["observations"] = [_row_to_observation(row) for row in observations]
    return payload


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
        item["first_seen"],
        item["last_seen"],
        item["observation_count"],
        item["state"],
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
        "first_seen": row["first_seen"],
        "last_seen": row["last_seen"],
        "observation_count": row["observation_count"],
        "state": row["state"],
        "mask_area": row["mask_area"],
        "map_position": map_position,
    }


def _row_to_observation(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "item_id": row["item_id"],
        "label": row["label"],
        "confidence": round(float(row["confidence"]), 3),
        "bbox": [row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]],
        "seen_at": row["seen_at"],
        "mask_area": row["mask_area"],
        "camera_pose_id": row["camera_pose_id"],
    }
