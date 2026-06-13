# Pi Digital Twin

Flask + OpenCV demo for a Pi 5 or Jetson Nano. It ingests camera frames, detects objects, tracks stable item IDs in a lightweight world model, and stores item state plus observation history in SQLite.

![Simulated dashboard](docs/simulated-dashboard.png)

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`.

## Test Without A Camera

```bash
source .venv/bin/activate
DB_PATH=/tmp/digital-twin-fake.db SIMULATED_FEED=1 MOCK_DETECTIONS=1 python app.py
```

This generates synthetic frames, fake detections, and simulated pose data so the full dashboard/API loop can be tested on any machine.

## World Model

The current world model is a pragmatic baseline, not a full neural scene model:

- YOLO detections, or mock detections in test mode
- label + bounding-box IoU association
- stable `item_id`
- `active`, `stale`, `lost` lifecycle
- SQLite tables for `items`, `observations`, and `camera_poses`
- optional SLAM pose input through `SLAM_POSE_FILE` or `SLAM_POSE_URL`

## Useful Config

- `CAMERA_INDEX=0`
- `FRAME_WIDTH=640`
- `FRAME_HEIGHT=480`
- `YOLO_CONFIDENCE=0.5`
- `DETECTION_INTERVAL=0.15`
- `ACTIVE_SECONDS=10`
- `LOST_SECONDS=60`
- `MATCH_IOU=0.35`
- `SEGMENTATION_MODEL=off`
- `DB_PATH=/path/to/detections.db`

## APIs

- `GET /api/status`
- `GET /api/summary`
- `GET /api/world`
- `GET /api/items/<item_id>`
- `GET /api/map`
