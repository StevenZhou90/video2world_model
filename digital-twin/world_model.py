from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable, Optional
from uuid import uuid4


BBox = tuple[float, float, float, float]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: BBox
    mask_area: Optional[float] = None
    map_position: Optional[dict] = None


@dataclass
class TrackedItem:
    item_id: str
    label: str
    confidence: float
    bbox: BBox
    first_seen: str
    last_seen: str
    observation_count: int = 1
    state: str = "active"
    mask_area: Optional[float] = None
    map_position: Optional[dict] = None

    def update(self, detection: Detection, seen_at: str) -> None:
        self.confidence = detection.confidence
        self.bbox = detection.bbox
        self.last_seen = seen_at
        self.observation_count += 1
        self.state = "active"
        self.mask_area = detection.mask_area
        self.map_position = detection.map_position

    def as_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "label": self.label,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "observation_count": self.observation_count,
            "state": self.state,
            "mask_area": self.mask_area,
            "map_position": self.map_position,
        }


@dataclass(frozen=True)
class WorldModelConfig:
    active_seconds: int = 10
    lost_seconds: int = 60
    match_iou: float = 0.35


@dataclass
class WorldModelUpdate:
    observed_items: list[TrackedItem] = field(default_factory=list)
    state_changed_items: list[TrackedItem] = field(default_factory=list)


class WorldModel:
    def __init__(
        self,
        config: WorldModelConfig,
        clock: Callable[[], str] = utc_now_iso,
    ):
        self.config = config
        self.clock = clock
        self.items: dict[str, TrackedItem] = {}

    def update(self, detections: Iterable[Detection]) -> WorldModelUpdate:
        seen_at = self.clock()
        observed: list[TrackedItem] = []

        for detection in detections:
            match = self._find_match(detection)
            if match is None:
                match = TrackedItem(
                    item_id=f"item-{uuid4().hex[:8]}",
                    label=detection.label,
                    confidence=detection.confidence,
                    bbox=detection.bbox,
                    first_seen=seen_at,
                    last_seen=seen_at,
                    mask_area=detection.mask_area,
                    map_position=detection.map_position,
                )
                self.items[match.item_id] = match
            else:
                match.update(detection, seen_at)
            observed.append(match)

        state_changed = self.refresh_states(seen_at)
        return WorldModelUpdate(observed_items=observed, state_changed_items=state_changed)

    def refresh_states(self, now_iso: Optional[str] = None) -> list[TrackedItem]:
        now = _parse_iso(now_iso or self.clock())
        changed: list[TrackedItem] = []
        for item in self.items.values():
            age = max(0, int((now - _parse_iso(item.last_seen)).total_seconds()))
            next_state = "active"
            if age > self.config.lost_seconds:
                next_state = "lost"
            elif age > self.config.active_seconds:
                next_state = "stale"
            if item.state != next_state:
                item.state = next_state
                changed.append(item)
        return changed

    def snapshot(self) -> list[dict]:
        self.refresh_states()
        return [item.as_dict() for item in sorted(self.items.values(), key=lambda value: value.last_seen, reverse=True)]

    def _find_match(self, detection: Detection) -> Optional[TrackedItem]:
        candidates = [
            item
            for item in self.items.values()
            if item.label == detection.label and item.state != "lost"
        ]
        if not candidates:
            return None

        best = max(candidates, key=lambda item: bbox_iou(item.bbox, detection.bbox))
        if bbox_iou(best.bbox, detection.bbox) < self.config.match_iou:
            return None
        return best


def bbox_iou(left: BBox, right: BBox) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    inter_x1 = max(lx1, rx1)
    inter_y1 = max(ly1, ry1)
    inter_x2 = min(lx2, rx2)
    inter_y2 = min(ly2, ry2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0

    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)
