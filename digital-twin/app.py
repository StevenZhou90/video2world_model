from __future__ import annotations

import logging
import os
import signal
import threading
import time
import math
import json
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template

from database import (
    get_latest_camera_pose,
    get_world_items,
    init_db,
    record_camera_pose,
    upsert_item,
)
from perception import PerceptionConfig, PerceptionPipeline, annotate_frame
from world_model import Detection, WorldModel, WorldModelConfig


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
LOGGER = logging.getLogger("video2world_model")


def env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    camera_index: int = int(os.getenv("CAMERA_INDEX", "0"))
    frame_width: int = int(os.getenv("FRAME_WIDTH", "640"))
    frame_height: int = int(os.getenv("FRAME_HEIGHT", "480"))
    confidence: float = float(os.getenv("YOLO_CONFIDENCE", "0.5"))
    model_path: str = os.getenv("YOLO_MODEL", "yolov8n.pt")
    segmentation_model: str = os.getenv("SEGMENTATION_MODEL", "off")
    slam_pose_file: Optional[str] = os.getenv("SLAM_POSE_FILE")
    slam_pose_url: Optional[str] = os.getenv("SLAM_POSE_URL")
    detection_interval: float = float(os.getenv("DETECTION_INTERVAL", "0.15"))
    match_iou: float = float(os.getenv("MATCH_IOU", "0.35"))
    simulated_feed: bool = env_flag("SIMULATED_FEED")
    mock_detections: bool = env_flag("MOCK_DETECTIONS")


class CameraDetector:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pipeline: Optional[PerceptionPipeline] = None
        if not settings.mock_detections:
            self.pipeline = PerceptionPipeline(
                PerceptionConfig(
                    confidence=settings.confidence,
                    yolo_model=settings.model_path,
                    segmentation_model=settings.segmentation_model,
                    slam_pose_file=settings.slam_pose_file,
                    slam_pose_url=settings.slam_pose_url,
                )
            )
        self.world_model = WorldModel(
            WorldModelConfig(
                match_iou=settings.match_iou,
            )
        )
        self.capture: Optional[cv2.VideoCapture] = None
        self.latest_jpeg: Optional[bytes] = None
        self.latest_error: Optional[str] = None
        self.latest_pose: Optional[dict] = None
        self.frame_count = 0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="camera-detector", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)
        if self.capture is not None:
            self.capture.release()

    def get_frame(self) -> Optional[bytes]:
        with self.lock:
            return self.latest_jpeg

    def get_status(self) -> dict:
        with self.lock:
            return {
                "camera_index": self.settings.camera_index,
                "model": self.settings.model_path,
                "confidence": self.settings.confidence,
                "segmentation_model": self.settings.segmentation_model,
                "slam_enabled": self._pose_enabled(),
                "simulated_feed": self.settings.simulated_feed,
                "mock_detections": self.settings.mock_detections,
                "error": self.latest_error,
                "has_frame": self.latest_jpeg is not None,
                "world_items": len(self.world_model.items),
            }

    def _open_capture(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self.settings.camera_index, cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.frame_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.frame_height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _run(self) -> None:
        if self.settings.mock_detections:
            LOGGER.info("Using mock detections")
        else:
            LOGGER.info("Loading YOLO model: %s", self.settings.model_path)
        if not self.settings.simulated_feed:
            self.capture = self._open_capture()
            if not self.capture.isOpened():
                self._set_error(f"Could not open camera index {self.settings.camera_index}")
                return

            LOGGER.info("Camera opened on index %s", self.settings.camera_index)
        else:
            LOGGER.info("Using simulated camera feed")
        last_detection_time = 0.0
        annotated = None

        while not self.stop_event.is_set():
            ok, frame = self._read_frame()
            if not ok or frame is None:
                self._set_error("Camera frame read failed")
                time.sleep(0.25)
                continue

            now = time.monotonic()
            if annotated is None or now - last_detection_time >= self.settings.detection_interval:
                annotated = self._update_world_and_annotate(frame)
                last_detection_time = now
            else:
                annotated = annotate_frame(frame, self.world_model.snapshot())

            encoded_ok, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if encoded_ok:
                with self.lock:
                    self.latest_jpeg = buffer.tobytes()
                    self.latest_error = None

            if self.settings.simulated_feed:
                time.sleep(0.03)

    def _read_frame(self):
        if self.settings.simulated_feed:
            self.frame_count += 1
            return True, self._synthetic_frame()
        if self.capture is None:
            return False, None
        self.frame_count += 1
        return self.capture.read()

    def _update_world_and_annotate(self, frame):
        if self.settings.mock_detections:
            detections, pose = self._mock_detections(), self._mock_pose()
        else:
            if self.pipeline is None:
                raise RuntimeError("Perception pipeline is not initialized")
            detections, pose = self.pipeline.infer(frame)
        if pose is not None:
            record_camera_pose("slam-adapter", json.dumps(pose, sort_keys=True))
            with self.lock:
                self.latest_pose = pose

        update = self.world_model.update(detections)
        for item in update.observed_items:
            payload = item.as_dict()
            upsert_item(payload)

        return annotate_frame(frame, [item.as_dict() for item in update.observed_items])

    def _synthetic_frame(self):
        height = self.settings.frame_height
        width = self.settings.frame_width
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (18, 21, 28)

        cup_bbox, bottle_bbox = self._synthetic_bboxes()
        cup_bbox = tuple(int(value) for value in cup_bbox)
        bottle_bbox = tuple(int(value) for value in bottle_bbox)

        cv2.rectangle(frame, (cup_bbox[0], cup_bbox[1]), (cup_bbox[2], cup_bbox[3]), (40, 150, 210), -1)
        cv2.circle(frame, (cup_bbox[0] + 45, cup_bbox[1] + 22), 22, (65, 180, 235), -1)
        cv2.rectangle(frame, (bottle_bbox[0], bottle_bbox[1]), (bottle_bbox[2], bottle_bbox[3]), (65, 170, 120), -1)
        cv2.rectangle(frame, (bottle_bbox[0] + 18, bottle_bbox[1] - 34), (bottle_bbox[2] - 18, bottle_bbox[1]), (80, 200, 145), -1)
        cv2.putText(frame, "SIMULATED FEED", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 235, 240), 2, cv2.LINE_AA)
        return frame

    def _mock_detections(self) -> list[Detection]:
        cup_bbox, bottle_bbox = self._synthetic_bboxes()
        pose = self._mock_pose()
        return [
            Detection(
                "synthetic-cup",
                0.92,
                cup_bbox,
                mask_area=9000.0,
                map_position={**pose["position"], "image_cx": round((cup_bbox[0] + cup_bbox[2]) / 2, 2), "image_cy": 170.0},
            ),
            Detection(
                "synthetic-bottle",
                0.88,
                bottle_bbox,
                mask_area=12180.0,
                map_position={**pose["position"], "image_cx": round((bottle_bbox[0] + bottle_bbox[2]) / 2, 2), "image_cy": 313.0},
            ),
        ]

    def _mock_pose(self) -> dict:
        t = self.frame_count / 30.0
        return {
            "position": {
                "x": round(math.sin(t) * 0.25, 3),
                "y": round(math.cos(t) * 0.25, 3),
                "z": 0.0,
            },
            "orientation": {"yaw": round(t, 3)},
            "source": "simulated",
        }

    def _synthetic_bboxes(self):
        width = self.settings.frame_width
        t = self.frame_count / 30.0
        cup_center = (width * 0.35) + math.sin(t * 0.8) * 55
        bottle_center = (width * 0.68) + math.cos(t * 0.6) * 45
        cup_x = max(20.0, min(width - 110.0, cup_center - 45.0))
        bottle_x = max(20.0, min(width - 90.0, bottle_center - 35.0))
        return (
            (cup_x, 120.0, cup_x + 90.0, 220.0),
            (bottle_x, 226.0, bottle_x + 70.0, 400.0),
        )

    def _set_error(self, message: str) -> None:
        LOGGER.error(message)
        with self.lock:
            self.latest_error = message

    def _pose_enabled(self) -> bool:
        return bool(self.settings.slam_pose_file or self.settings.slam_pose_url or self.settings.mock_detections)


settings = Settings()
init_db()
detector = CameraDetector(settings)
detector.start()

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html", settings=settings)


@app.get("/video_feed")
def video_feed():
    return Response(_mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/status")
def status():
    return jsonify(detector.get_status())


@app.get("/api/world")
def world():
    return jsonify({"items": get_world_items()})


@app.get("/api/map")
def map_state():
    return jsonify(
        {
            "slam_enabled": detector._pose_enabled(),
            "camera_pose": get_latest_camera_pose(),
            "items": [item for item in get_world_items() if item.get("map_position") is not None],
        }
    )


def _mjpeg_stream():
    while True:
        frame = detector.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.03)


def _shutdown(*_args):
    detector.stop()
    raise SystemExit(0)


signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
