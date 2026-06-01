# Raspberry Pi 5 Digital Twin Demo

This proof-of-concept runs a Flask dashboard on a Raspberry Pi 5, reads a USB webcam or Pi camera through OpenCV, detects objects with Ultralytics YOLOv8n, overlays bounding boxes on a live MJPEG stream, and records detections in SQLite.

## Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

## Create the virtual environment

```bash
cd /home/steven/pi-camera/digital-twin
uv venv
source .venv/bin/activate
```

## Install dependencies

```bash
uv pip install --torch-backend cpu -r requirements.txt
```

The first launch downloads `yolov8n.pt` automatically if it is not already present.

## Run the demo

```bash
cd /home/steven/pi-camera/digital-twin
source .venv/bin/activate
python app.py
```

The Flask server listens on `0.0.0.0:5000`.

Open the dashboard on the Pi:

```text
http://localhost:5000
```

Open the dashboard from another machine on the same network:

```bash
hostname -I
```

Use the first IP address shown:

```text
http://<PI_IP_ADDRESS>:5000
```

## Camera and runtime settings

Defaults are tuned for a USB camera at `/dev/video0`.

```bash
CAMERA_INDEX=0 FRAME_WIDTH=640 FRAME_HEIGHT=480 YOLO_CONFIDENCE=0.5 python app.py
```

Useful variables:

- `CAMERA_INDEX`: OpenCV camera index, usually `0` for `/dev/video0`.
- `FRAME_WIDTH`: requested camera width.
- `FRAME_HEIGHT`: requested camera height.
- `YOLO_CONFIDENCE`: minimum confidence recorded in SQLite and displayed on video.
- `DETECTION_INTERVAL`: seconds between YOLO inference passes. Increase this if the Pi is too slow.
- `ACTIVE_SECONDS`: how recently a class must have been seen to count as active.

## SQLite data

The app creates `detections.db` automatically on startup.

```bash
sqlite3 detections.db "select object_type, confidence, seen_at from detections order by seen_at desc limit 10;"
```
