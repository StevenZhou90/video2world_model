from __future__ import annotations

from world_model import Detection, WorldModel, WorldModelConfig, bbox_iou


def test_bbox_iou_handles_overlap_and_miss():
    assert round(bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)), 3) == 0.143
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_world_model_updates_existing_item_by_label_and_iou():
    model = WorldModel(WorldModelConfig(match_iou=0.3))

    first = model.update([Detection("cup", 0.8, (10, 10, 40, 40))])
    second = model.update([Detection("cup", 0.9, (12, 12, 42, 42))])

    assert first.observed_items[0].item_id == second.observed_items[0].item_id
    assert second.observed_items[0].confidence == 0.9


def test_world_model_creates_new_item_when_label_differs():
    model = WorldModel(WorldModelConfig(match_iou=0.3))

    update = model.update(
        [
            Detection("cup", 0.8, (10, 10, 40, 40)),
            Detection("bottle", 0.7, (10, 10, 40, 40)),
        ]
    )

    assert len(update.observed_items) == 2
    assert update.observed_items[0].item_id != update.observed_items[1].item_id


def test_world_model_snapshot_keeps_visual_fields_only():
    model = WorldModel(WorldModelConfig(match_iou=0.3))
    item = model.update(
        [
            Detection(
                "cup",
                0.8,
                (10, 10, 40, 40),
                mask_area=900.0,
                map_position={"x": 1.0, "y": 2.0, "z": 0.0},
            )
        ]
    ).observed_items[0]

    assert item.as_dict().keys() == {
        "item_id",
        "label",
        "confidence",
        "bbox",
        "mask_area",
        "map_position",
    }
