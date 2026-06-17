from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import urlopen

import cv2
from ultralytics import YOLO

from world_model import Detection


LOGGER = logging.getLogger("video2world_model")


@dataclass(frozen=True)
class PerceptionConfig:
    confidence: float
    yolo_model: str
    segmentation_model: str = os.getenv("SEGMENTATION_MODEL", "off")
    slam_pose_file: Optional[str] = os.getenv("SLAM_POSE_FILE")
    slam_pose_url: Optional[str] = os.getenv("SLAM_POSE_URL")


class PerceptionPipeline:
    def __init__(self, config: PerceptionConfig):
        self.config = config
        self.detector = YOLO(config.yolo_model)
        self.segmenter = Segmenter(config.segmentation_model)
        self.pose_source = SlamPoseSource(config.slam_pose_file, config.slam_pose_url)

    def infer(self, frame) -> tuple[list[Detection], Optional[dict]]:
        pose = self.pose_source.latest_pose()
        results = self.detector.predict(frame, conf=self.config.confidence, verbose=False)
        if not results or results[0].boxes is None:
            return [], pose

        result = results[0]
        detections: list[Detection] = []
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence < self.config.confidence:
                continue

            class_id = int(box.cls[0])
            label = result.names.get(class_id, str(class_id))
            bbox = tuple(float(value) for value in box.xyxy[0].tolist())
            mask_area = self.segmenter.mask_area(frame, bbox)
            detections.append(
                Detection(
                    label=label,
                    confidence=confidence,
                    bbox=bbox,
                    mask_area=mask_area,
                    map_position=_image_to_map_position(bbox, pose),
                )
            )

        return detections, pose


class Segmenter:
    def __init__(self, model_name: str):
        self.model_name = model_name.lower().strip()
        self.enabled = self.model_name not in {"", "off", "none", "false", "0"}
        self.model = None
        if self.enabled:
            LOGGER.warning(
                "SEGMENTATION_MODEL=%s requested, but SAM is not bundled in this baseline. "
                "Using bounding-box area as a mask summary.",
                model_name,
            )

    def mask_area(self, frame, bbox: tuple[float, float, float, float]) -> Optional[float]:
        if not self.enabled:
            return None
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = _clip_bbox(bbox, width, height)
        return float(max(0, x2 - x1) * max(0, y2 - y1))


class SlamPoseSource:
    def __init__(self, pose_file: Optional[str], pose_url: Optional[str]):
        self.pose_file = Path(pose_file).expanduser() if pose_file else None
        self.pose_url = pose_url
        self.latest: Optional[dict] = None

    def latest_pose(self) -> Optional[dict]:
        pose = self._read_url_pose() if self.pose_url else None
        if pose is None and self.pose_file is not None:
            pose = self._read_file_pose()
        if pose is not None:
            self.latest = pose
        return self.latest

    def _read_file_pose(self) -> Optional[dict]:
        if self.pose_file is None or not self.pose_file.exists():
            return None
        try:
            return json.loads(self.pose_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOGGER.warning("Could not read SLAM pose file %s: %s", self.pose_file, exc)
            return None

    def _read_url_pose(self) -> Optional[dict]:
        if not self.pose_url:
            return None
        try:
            with urlopen(self.pose_url, timeout=0.05) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, URLError, json.JSONDecodeError) as exc:
            LOGGER.debug("Could not read SLAM pose URL %s: %s", self.pose_url, exc)
            return None


def annotate_frame(frame, items: list[dict]):
    annotated = frame.copy()
    for item in items:
        x1, y1, x2, y2 = [int(value) for value in item["bbox"]]
        color = _state_color(item["state"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"{item['item_id']} {item['label']} {item['confidence']:.2f}"
        _draw_label(annotated, text, x1, y1, color)
    return annotated


def _draw_label(frame, text: str, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    text_w, text_h = text_size
    y_text = max(y1, text_h + baseline + 6)
    cv2.rectangle(frame, (x1, y_text - text_h - baseline - 6), (x1 + text_w + 8, y_text), color, -1)
    cv2.putText(
        frame,
        text,
        (x1 + 4, y_text - baseline - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _state_color(state: str) -> tuple[int, int, int]:
    if state == "lost":
        return (91, 95, 106)
    if state == "stale":
        return (0, 180, 216)
    return (42, 157, 143)


def _clip_bbox(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width, int(x1))),
        max(0, min(height, int(y1))),
        max(0, min(width, int(x2))),
        max(0, min(height, int(y2))),
    )


def _image_to_map_position(bbox: tuple[float, float, float, float], pose: Optional[dict]) -> Optional[dict]:
    if not pose:
        return None
    position = pose.get("position") if isinstance(pose, dict) else None
    if not isinstance(position, dict):
        return None
    x1, y1, x2, y2 = bbox
    return {
        "x": position.get("x"),
        "y": position.get("y"),
        "z": position.get("z"),
        "image_cx": round((x1 + x2) / 2, 2),
        "image_cy": round((y1 + y2) / 2, 2),
    }
