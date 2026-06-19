from __future__ import annotations

import database


def test_visual_item_persistence(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "world_model.db")
    database.init_db()

    item = {
        "item_id": "item-test",
        "label": "cup",
        "confidence": 0.87,
        "bbox": [1, 2, 30, 40],
        "mask_area": 900,
        "map_position": {"x": 1.0, "y": 2.0, "z": 0.0},
    }

    database.upsert_item(item)

    world = database.get_world_items()

    assert world[0]["item_id"] == "item-test"
    assert world[0]["map_position"] == {"x": 1.0, "y": 2.0, "z": 0.0}
    assert "first_seen" not in world[0]
    assert "last_seen" not in world[0]
    assert "observation_count" not in world[0]


def test_camera_pose_persistence(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "world_model.db")
    database.init_db()

    pose_id = database.record_camera_pose("test", '{"position": {"x": 1}}')
    pose = database.get_latest_camera_pose()

    assert pose is not None
    assert pose["id"] == pose_id
    assert pose["source"] == "test"


def test_legacy_item_schema_is_purged(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "world_model.db")
    with database.connect() as conn:
        conn.execute(
            """
            CREATE TABLE items (
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
                state TEXT NOT NULL
            )
            """
        )

    database.init_db()

    with database.connect() as conn:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(items)").fetchall()}
    assert "first_seen" not in columns
    assert "last_seen" not in columns
    assert "observation_count" not in columns
