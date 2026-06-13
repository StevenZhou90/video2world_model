from __future__ import annotations

import database


def test_item_persistence_and_history(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "detections.db")
    database.init_db()

    item = {
        "item_id": "item-test",
        "label": "cup",
        "confidence": 0.87,
        "bbox": [1, 2, 30, 40],
        "first_seen": "2026-06-13T00:00:00+00:00",
        "last_seen": "2026-06-13T00:00:01+00:00",
        "observation_count": 1,
        "state": "active",
        "mask_area": 900,
        "map_position": {"x": 1.0, "y": 2.0, "z": 0.0},
    }

    database.upsert_item(item)
    database.record_observation(item)

    world = database.get_world_items()
    history = database.get_item_history("item-test")
    summary = database.get_summary()

    assert world[0]["item_id"] == "item-test"
    assert world[0]["map_position"] == {"x": 1.0, "y": 2.0, "z": 0.0}
    assert history is not None
    assert history["observations"][0]["label"] == "cup"
    assert summary["total_detections"] == 1


def test_camera_pose_persistence(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "detections.db")
    database.init_db()

    pose_id = database.record_camera_pose("test", '{"position": {"x": 1}}')
    pose = database.get_latest_camera_pose()

    assert pose is not None
    assert pose["id"] == pose_id
    assert pose["source"] == "test"
