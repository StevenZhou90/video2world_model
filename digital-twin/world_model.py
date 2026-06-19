from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional
from uuid import uuid4


BBox = tuple[float, float, float, float]


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
    mask_area: Optional[float] = None
    map_position: Optional[dict] = None

    def update(self, detection: Detection) -> None:
        self.confidence = detection.confidence
        self.bbox = detection.bbox
        self.mask_area = detection.mask_area
        self.map_position = detection.map_position

    def as_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "label": self.label,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "mask_area": self.mask_area,
            "map_position": self.map_position,
        }


@dataclass(frozen=True)
class WorldModelConfig:
    match_iou: float = 0.35


@dataclass
class WorldModelUpdate:
    observed_items: list[TrackedItem] = field(default_factory=list)


class WorldModel:
    def __init__(self, config: WorldModelConfig):
        self.config = config
        self.items: dict[str, TrackedItem] = {}

    def update(self, detections: Iterable[Detection]) -> WorldModelUpdate:
        observed: list[TrackedItem] = []

        for detection in detections:
            match = self._find_match(detection)
            if match is None:
                match = TrackedItem(
                    item_id=f"item-{uuid4().hex[:8]}",
                    label=detection.label,
                    confidence=detection.confidence,
                    bbox=detection.bbox,
                    mask_area=detection.mask_area,
                    map_position=detection.map_position,
                )
                self.items[match.item_id] = match
            else:
                match.update(detection)
            observed.append(match)

        return WorldModelUpdate(observed_items=observed)

    def snapshot(self) -> list[dict]:
        return [item.as_dict() for item in sorted(self.items.values(), key=lambda value: value.item_id)]

    def _find_match(self, detection: Detection) -> Optional[TrackedItem]:
        candidates = [
            item
            for item in self.items.values()
            if item.label == detection.label
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
