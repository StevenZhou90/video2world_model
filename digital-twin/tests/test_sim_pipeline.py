from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from sim_pipeline import build_map, export_usd, extract_frames, preview_scene, reconstruct_scene, run_pipeline


def test_extract_frames_writes_metadata(tmp_path):
    video = _make_video(tmp_path / "walkthrough.avi", frame_count=8, fps=4)
    job = tmp_path / "job"

    metadata = extract_frames(video, job, sample_fps=2)

    assert metadata.frame_count == 4
    assert (job / "metadata.json").exists()
    assert (job / metadata.frames[0].path).exists()


def test_reconstruct_map_and_usd_export(tmp_path):
    video = _make_video(tmp_path / "walkthrough.avi", frame_count=6, fps=3)
    job = tmp_path / "job"
    extract_frames(video, job, sample_fps=1)

    reconstruction = reconstruct_scene(job)
    scene = build_map(job)
    usd = export_usd(job)
    preview = preview_scene(job)

    assert reconstruction.points
    assert (job / "point_cloud.ply").exists()
    assert (job / "map" / "occupancy_grid.pgm").exists()
    assert scene["grid"]["width"] > 0
    assert scene["obstacle_cells"] > 0
    assert usd.exists()
    assert preview.exists()
    assert (job / "preview.html").exists()
    assert 'def Xform "RobotSpawn"' in usd.read_text(encoding="utf-8")
    assert 'def Xform "CollisionObstacles"' in usd.read_text(encoding="utf-8")


def test_run_pipeline_creates_expected_artifacts(tmp_path):
    video = _make_video(tmp_path / "walkthrough.avi", frame_count=5, fps=5)
    job = tmp_path / "job"

    result = run_pipeline(video, job, sample_fps=1)

    assert Path(result["usd"]).exists()
    assert Path(result["preview"]).exists()
    assert (job / "scene.json").exists()


def _make_video(path: Path, frame_count: int, fps: int) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:] = (20 + index * 5, 25, 30)
        cv2.rectangle(frame, (8 + index, 12), (30 + index, 32), (180, 180, 180), -1)
        writer.write(frame)
    writer.release()
    return path
