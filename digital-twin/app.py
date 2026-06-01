from __future__ import annotations

import logging
import os
import signal
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
from flask import Flask, Response, jsonify, render_template
from ultralytics import YOLO

from database import get_summary, init_db, record_detection


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
LOGGER = logging.getLogger("digital-twin")


@dataclass(frozen=True)
class Settings:
    camera_index: int = int(os.getenv("CAMERA_INDEX", "0"))
    frame_width: int = int(os.getenv("FRAME_WIDTH", "640"))
    frame_height: int = int(os.getenv("FRAME_HEIGHT", "480"))
    confidence: float = float(os.getenv("YOLO_CONFIDENCE", "0.5"))
    model_path: str = os.getenv("YOLO_MODEL", "yolov8n.pt")
    detection_interval: float = float(os.getenv("DETECTION_INTERVAL", "0.15"))
    active_seconds: int = int(os.getenv("ACTIVE_SECONDS", "10"))


class CameraDetector:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = YOLO(settings.model_path)
        self.capture: Optional[cv2.VideoCapture] = None
        self.latest_jpeg: Optional[bytes] = None
        self.latest_error: Optional[str] = None
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
                "error": self.latest_error,
                "has_frame": self.latest_jpeg is not None,
            }

    def _open_capture(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self.settings.camera_index, cv2.CAP_V4L2)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.frame_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.frame_height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _run(self) -> None:
        LOGGER.info("Loading YOLO model: %s", self.settings.model_path)
        self.capture = self._open_capture()
        if not self.capture.isOpened():
            self._set_error(f"Could not open camera index {self.settings.camera_index}")
            return

        LOGGER.info("Camera opened on index %s", self.settings.camera_index)
        last_detection_time = 0.0
        annotated = None

        while not self.stop_event.is_set():
            ok, frame = self.capture.read()
            if not ok or frame is None:
                self._set_error("Camera frame read failed")
                time.sleep(0.25)
                continue

            now = time.monotonic()
            if annotated is None or now - last_detection_time >= self.settings.detection_interval:
                annotated = self._detect_and_annotate(frame)
                last_detection_time = now
            else:
                annotated = frame

            encoded_ok, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if encoded_ok:
                with self.lock:
                    self.latest_jpeg = buffer.tobytes()
                    self.latest_error = None

    def _detect_and_annotate(self, frame):
        results = self.model.predict(frame, conf=self.settings.confidence, verbose=False)
        annotated = frame.copy()

        if not results:
            return annotated

        result = results[0]
        names = result.names
        boxes = result.boxes
        if boxes is None:
            return annotated

        for box in boxes:
            confidence = float(box.conf[0])
            if confidence < self.settings.confidence:
                continue

            class_id = int(box.cls[0])
            label = names.get(class_id, str(class_id))
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]

            record_detection(label, confidence)
            self._draw_box(annotated, label, confidence, x1, y1, x2, y2)

        return annotated

    def _draw_box(self, frame, label: str, confidence: float, x1: int, y1: int, x2: int, y2: int) -> None:
        color = (42, 157, 143)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {confidence:.2f}"
        text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_w, text_h = text_size
        y_text = max(y1, text_h + baseline + 6)
        cv2.rectangle(frame, (x1, y_text - text_h - baseline - 6), (x1 + text_w + 8, y_text), color, -1)
        cv2.putText(
            frame,
            text,
            (x1 + 4, y_text - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def _set_error(self, message: str) -> None:
        LOGGER.error(message)
        with self.lock:
            self.latest_error = message


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


@app.get("/api/summary")
def summary():
    return jsonify(get_summary(active_seconds=settings.active_seconds))


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
