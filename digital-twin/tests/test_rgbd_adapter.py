from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from rgbd_adapter import reconstruct_rgbd_frame


def test_reconstruct_rgbd_frame_uses_dark_foreground_and_depth(tmp_path):
    rgb_video = _write_video(tmp_path / "rgb.avi", color=True)
    depth_video = _write_video(tmp_path / "depth.avi", color=False)

    points = reconstruct_rgbd_frame(
        rgb_video=rgb_video,
        depth_video=depth_video,
        frame_index=0,
        crop=(10, 10, 54, 54),
        brightness_threshold=120,
        depth_scale=0.05,
        min_depth=0.1,
        max_depth=4.0,
        stride=2,
    )

    assert points
    assert all(point["confidence"] == 1.0 for point in points)
    assert all(point["z"] >= 0.0 for point in points)
    assert all(25 <= point["r"] <= 35 for point in points)


def _write_video(path: Path, color: bool) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 64))
    assert writer.isOpened()
    frame = np.full((64, 64, 3), 230, dtype=np.uint8)
    if color:
        frame[16:48, 16:48] = (50, 40, 30)
    else:
        frame[:] = (0, 0, 0)
        frame[16:48, 16:48] = (30, 30, 30)
    writer.write(frame)
    writer.release()
    return path
