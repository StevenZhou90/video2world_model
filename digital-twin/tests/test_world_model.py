from __future__ import annotations

from world_model import Detection, WorldModel, WorldModelConfig, bbox_iou


def test_bbox_iou_handles_overlap_and_miss():
    assert round(bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)), 3) == 0.143
    assert bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_world_model_updates_existing_item_by_label_and_iou():
    times = iter(["2026-06-13T00:00:00+00:00", "2026-06-13T00:00:01+00:00"])
    model = WorldModel(WorldModelConfig(active_seconds=10, lost_seconds=60, match_iou=0.3), clock=lambda: next(times))

    first = model.update([Detection("cup", 0.8, (10, 10, 40, 40))])
    second = model.update([Detection("cup", 0.9, (12, 12, 42, 42))])

    assert first.observed_items[0].item_id == second.observed_items[0].item_id
    assert second.observed_items[0].observation_count == 2
    assert second.observed_items[0].confidence == 0.9


def test_world_model_creates_new_item_when_label_differs():
    model = WorldModel(WorldModelConfig(match_iou=0.3), clock=lambda: "2026-06-13T00:00:00+00:00")

    update = model.update(
        [
            Detection("cup", 0.8, (10, 10, 40, 40)),
            Detection("bottle", 0.7, (10, 10, 40, 40)),
        ]
    )

    assert len(update.observed_items) == 2
    assert update.observed_items[0].item_id != update.observed_items[1].item_id


def test_world_model_marks_items_stale_and_lost():
    times = iter(
        [
            "2026-06-13T00:00:00+00:00",
            "2026-06-13T00:00:06+00:00",
            "2026-06-13T00:00:12+00:00",
        ]
    )
    model = WorldModel(WorldModelConfig(active_seconds=5, lost_seconds=10), clock=lambda: next(times))
    item = model.update([Detection("cup", 0.8, (10, 10, 40, 40))]).observed_items[0]

    stale = model.refresh_states()
    assert stale == [item]
    assert item.state == "stale"

    lost = model.refresh_states()
    assert lost == [item]
    assert item.state == "lost"
