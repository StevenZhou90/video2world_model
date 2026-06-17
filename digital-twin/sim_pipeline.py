from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameMetadata:
    frame_id: int
    source_frame: int
    timestamp_seconds: float
    path: str


@dataclass(frozen=True)
class VideoMetadata:
    source_video: str
    source_fps: float
    width: int
    height: int
    sampled_fps: float
    frame_count: int
    frames: list[FrameMetadata]


@dataclass(frozen=True)
class ReconstructionPoint:
    x: float
    y: float
    z: float
    confidence: float
    r: int = 255
    g: int = 255
    b: int = 255


@dataclass(frozen=True)
class CameraPose:
    frame_id: int
    timestamp_seconds: float
    position: dict[str, float]
    rotation_quat_xyzw: list[float]


@dataclass(frozen=True)
class Reconstruction:
    source: str
    scale: str
    camera_poses: list[CameraPose]
    points: list[ReconstructionPoint]


def extract_frames(video_path: Path, output_dir: Path, sample_fps: float = 2.0) -> VideoMetadata:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    stride = max(1, int(round(source_fps / sample_fps)))

    frames: list[FrameMetadata] = []
    source_frame = 0
    sampled_id = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if source_frame % stride == 0:
                path = frames_dir / f"frame_{sampled_id:06d}.jpg"
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"Could not write frame: {path}")
                frames.append(
                    FrameMetadata(
                        frame_id=sampled_id,
                        source_frame=source_frame,
                        timestamp_seconds=round(source_frame / source_fps, 6),
                        path=str(path.relative_to(output_dir)),
                    )
                )
                sampled_id += 1
            source_frame += 1
    finally:
        capture.release()

    if not frames:
        raise RuntimeError("No frames extracted from video")

    metadata = VideoMetadata(
        source_video=str(video_path),
        source_fps=source_fps,
        width=width,
        height=height,
        sampled_fps=sample_fps,
        frame_count=len(frames),
        frames=frames,
    )
    _write_json(output_dir / "metadata.json", _dataclass_to_json(metadata))
    return metadata


def reconstruct_scene(job_dir: Path) -> Reconstruction:
    external = os.getenv("VGGT_COMMAND")
    if external:
        return _run_external_vggt(external, job_dir)
    return _fallback_reconstruction(job_dir)


def build_map(job_dir: Path, resolution: float = 0.1, obstacle_height: float = 0.25) -> dict:
    reconstruction = _read_json(job_dir / "reconstruction.json")
    points = np.array(
        [[point["x"], point["y"], point["z"], point["confidence"]] for point in reconstruction["points"]],
        dtype=np.float32,
    )
    if points.size == 0:
        raise RuntimeError("Reconstruction has no points")

    floor_z = float(np.percentile(points[:, 2], 10))
    obstacle_points = points[(points[:, 2] > floor_z + obstacle_height) & (points[:, 3] >= 0.35)]
    if obstacle_points.size == 0:
        obstacle_points = points[points[:, 3] >= 0.35]

    xy = points[:, :2]
    min_xy = xy.min(axis=0) - 0.5
    max_xy = xy.max(axis=0) + 0.5
    size = np.maximum(np.ceil((max_xy - min_xy) / resolution).astype(int), 1)
    width, height = int(size[0]), int(size[1])
    grid = np.full((height, width), 254, dtype=np.uint8)

    for point in obstacle_points:
        gx = int((point[0] - min_xy[0]) / resolution)
        gy = int((point[1] - min_xy[1]) / resolution)
        if 0 <= gx < width and 0 <= gy < height:
            grid[height - gy - 1, gx] = 0

    grid = _inflate_obstacles(grid, radius_cells=2)
    map_dir = job_dir / "map"
    map_dir.mkdir(exist_ok=True)
    pgm_path = map_dir / "occupancy_grid.pgm"
    yaml_path = map_dir / "occupancy_grid.yaml"
    cv2.imwrite(str(pgm_path), grid)

    origin = [round(float(min_xy[0]), 4), round(float(min_xy[1]), 4), 0.0]
    yaml_text = (
        "image: occupancy_grid.pgm\n"
        f"resolution: {resolution}\n"
        f"origin: [{origin[0]}, {origin[1]}, {origin[2]}]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")

    scene = {
        "source": reconstruction["source"],
        "coordinate_system": reconstruction.get("coordinate_system", "unknown"),
        "scale": reconstruction["scale"],
        "floor_z": floor_z,
        "resolution": resolution,
        "origin": origin,
        "grid": {"width": width, "height": height, "image": "map/occupancy_grid.pgm", "yaml": "map/occupancy_grid.yaml"},
        "spawn": _choose_spawn(grid, origin, resolution),
        "obstacle_cells": int(np.count_nonzero(grid == 0)),
    }
    _write_json(job_dir / "scene.json", scene)
    return scene


def export_usd(job_dir: Path) -> Path:
    scene = _read_json(job_dir / "scene.json")
    grid = cv2.imread(str(job_dir / scene["grid"]["image"]), cv2.IMREAD_GRAYSCALE)
    if grid is None:
        raise RuntimeError("Could not read occupancy grid")

    resolution = float(scene["resolution"])
    origin_x, origin_y, _ = scene["origin"]
    height, width = grid.shape
    obstacles = _grid_to_boxes(grid, origin_x, origin_y, resolution)

    usd_dir = job_dir / "sim"
    usd_dir.mkdir(exist_ok=True)
    usd_path = usd_dir / "scene.usda"
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        "    metersPerUnit = 1",
        ")",
        "",
        'def Xform "World" {',
        '    def Cube "Floor" {',
        f"        double3 xformOp:scale = ({width * resolution / 2:.4f}, {height * resolution / 2:.4f}, 0.025)",
        f"        double3 xformOp:translate = ({origin_x + width * resolution / 2:.4f}, {origin_y + height * resolution / 2:.4f}, -0.025)",
        '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
        "    }",
    ]
    lines.extend(_usd_point_cloud_lines(job_dir))
    hide_collision_blocks = scene["source"].startswith("vggt:") and not _env_flag("SHOW_COLLISION_BLOCKS")
    lines.append('    def Xform "CollisionObstacles" {')
    if hide_collision_blocks:
        lines.append('        token visibility = "invisible"')
    for index, (x, y, sx, sy) in enumerate(obstacles):
        lines.extend(
            [
                f'        def Cube "Obstacle_{index:04d}" {{',
                f"            double3 xformOp:scale = ({sx / 2:.4f}, {sy / 2:.4f}, 0.5)",
                f"            double3 xformOp:translate = ({x:.4f}, {y:.4f}, 0.5)",
                '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
                "        }",
            ]
        )
    lines.append("    }")
    spawn = scene["spawn"]
    lines.extend(
        [
            '    def Xform "RobotSpawn" {',
            f"        double3 xformOp:translate = ({spawn['x']:.4f}, {spawn['y']:.4f}, 0)",
            '        uniform token[] xformOpOrder = ["xformOp:translate"]',
            "    }",
            "}",
            "",
        ]
    )
    usd_path.write_text("\n".join(lines), encoding="utf-8")
    return usd_path


def preview_scene(job_dir: Path, output_path: Path | None = None) -> Path:
    scene = _read_json(job_dir / "scene.json")
    grid = cv2.imread(str(job_dir / scene["grid"]["image"]), cv2.IMREAD_GRAYSCALE)
    if grid is None:
        raise RuntimeError("Could not read occupancy grid")

    output_path = output_path or job_dir / "preview.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scale = max(2, min(12, 900 // max(grid.shape)))
    preview = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
    preview[grid == 0] = (55, 55, 55)
    preview[grid > 200] = (236, 239, 244)

    spawn = scene["spawn"]
    resolution = float(scene["resolution"])
    origin_x, origin_y, _ = scene["origin"]
    col = int((spawn["x"] - origin_x) / resolution)
    row = grid.shape[0] - int((spawn["y"] - origin_y) / resolution) - 1
    if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]:
        cv2.circle(preview, (col, row), 4, (40, 70, 230), -1)

    preview = cv2.resize(preview, (grid.shape[1] * scale, grid.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    cv2.putText(preview, "occupied", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (55, 55, 55), 2, cv2.LINE_AA)
    cv2.putText(preview, "spawn", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 70, 230), 2, cv2.LINE_AA)

    if not cv2.imwrite(str(output_path), preview):
        raise RuntimeError(f"Could not write preview: {output_path}")
    _write_preview_html(job_dir, output_path, scene)
    return output_path


def run_pipeline(video_path: Path, output_dir: Path, sample_fps: float = 2.0) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = extract_frames(video_path, output_dir, sample_fps)
    reconstruction = reconstruct_scene(output_dir)
    scene = build_map(output_dir)
    usd = export_usd(output_dir)
    preview = preview_scene(output_dir)
    return {
        "metadata": _dataclass_to_json(metadata),
        "reconstruction": _dataclass_to_json(reconstruction),
        "scene": scene,
        "usd": str(usd),
        "preview": str(preview),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert video into a coarse robotics simulation scene.")
    parser.add_argument("video", type=Path, help="Input walkthrough video")
    parser.add_argument("--out", type=Path, default=Path("artifacts/video-to-sim"), help="Output job directory")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Frame sampling rate")
    args = parser.parse_args(argv)

    result = run_pipeline(args.video, args.out, args.sample_fps)
    print(json.dumps({"scene": str(args.out / "scene.json"), "usd": result["usd"], "preview": result["preview"]}, indent=2))
    return 0


def preview_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a quick browser/PNG preview of a generated sim scene.")
    parser.add_argument("job_dir", type=Path, help="Pipeline output directory containing scene.json")
    parser.add_argument("--out", type=Path, default=None, help="Preview PNG path, defaults to JOB_DIR/preview.png")
    args = parser.parse_args(argv)

    preview = preview_scene(args.job_dir, args.out)
    print(json.dumps({"preview": str(preview), "html": str(args.job_dir / "preview.html")}, indent=2))
    return 0


def _run_external_vggt(command: str, job_dir: Path) -> Reconstruction:
    subprocess.run(shlex.split(command.format(frames=job_dir / "frames", out=job_dir)), check=True)
    payload = _read_json(job_dir / "reconstruction.json")
    return Reconstruction(
        source=payload.get("source", "vggt"),
        scale=payload.get("scale", "relative"),
        camera_poses=[CameraPose(**pose) for pose in payload["camera_poses"]],
        points=[ReconstructionPoint(**point) for point in payload["points"]],
    )


def _fallback_reconstruction(job_dir: Path) -> Reconstruction:
    metadata = _read_json(job_dir / "metadata.json")
    points: list[ReconstructionPoint] = []
    camera_poses: list[CameraPose] = []
    frames = metadata["frames"]
    width = max(1, int(metadata["width"]))
    height = max(1, int(metadata["height"]))

    for frame in frames:
        camera_x = frame["frame_id"] * 0.25
        camera_poses.append(
            CameraPose(
                frame_id=frame["frame_id"],
                timestamp_seconds=frame["timestamp_seconds"],
                position={"x": round(camera_x, 4), "y": 0.0, "z": 1.5},
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            )
        )
        image = cv2.imread(str(job_dir / frame["path"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        for y in range(0, height, max(1, height // 12)):
            for x in range(0, width, max(1, width // 16)):
                intensity = float(image[min(y, image.shape[0] - 1), min(x, image.shape[1] - 1)]) / 255.0
                world_x = camera_x + (x / width - 0.5) * 4.0
                world_y = (0.5 - y / height) * 3.0
                z = 0.0 if y > height * 0.58 else 0.35 + intensity
                shade = int(round(intensity * 255))
                points.append(
                    ReconstructionPoint(
                        round(world_x, 4),
                        round(world_y, 4),
                        round(z, 4),
                        round(0.4 + intensity * 0.6, 4),
                        shade,
                        shade,
                        shade,
                    )
                )

    reconstruction = Reconstruction(source="fallback-frame-depth", scale="relative", camera_poses=camera_poses, points=points)
    _write_json(job_dir / "reconstruction.json", _dataclass_to_json(reconstruction))
    _write_ply(job_dir / "point_cloud.ply", points)
    return reconstruction


def _inflate_obstacles(grid: np.ndarray, radius_cells: int) -> np.ndarray:
    if radius_cells <= 0:
        return grid
    kernel = np.ones((radius_cells * 2 + 1, radius_cells * 2 + 1), dtype=np.uint8)
    occupied = (grid == 0).astype(np.uint8)
    inflated = cv2.dilate(occupied, kernel, iterations=1)
    result = grid.copy()
    result[inflated > 0] = 0
    return result


def _choose_spawn(grid: np.ndarray, origin: list[float], resolution: float) -> dict[str, float]:
    free = np.argwhere(grid > 200)
    if free.size == 0:
        return {"x": origin[0], "y": origin[1], "theta": 0.0}
    row, col = free[len(free) // 2]
    height = grid.shape[0]
    return {
        "x": round(origin[0] + col * resolution, 4),
        "y": round(origin[1] + (height - row - 1) * resolution, 4),
        "theta": 0.0,
    }


def _grid_to_boxes(grid: np.ndarray, origin_x: float, origin_y: float, resolution: float) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    occupied = np.argwhere(grid == 0)
    height = grid.shape[0]
    max_boxes = 250
    stride = max(1, int(math.ceil(len(occupied) / max_boxes))) if len(occupied) else 1
    for row, col in occupied[::stride]:
        x = origin_x + (col + 0.5) * resolution
        y = origin_y + (height - row - 0.5) * resolution
        boxes.append((round(x, 4), round(y, 4), resolution, resolution))
    return boxes


def _write_ply(path: Path, points: Iterable[ReconstructionPoint]) -> None:
    point_list = list(points)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(point_list)}",
        "property float x",
        "property float y",
        "property float z",
        "property float confidence",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    lines.extend(f"{p.x} {p.y} {p.z} {p.confidence} {p.r} {p.g} {p.b}" for p in point_list)
    path.write_text("\n".join(lines), encoding="utf-8")


def _usd_point_cloud_lines(job_dir: Path, max_points: int | None = None) -> list[str]:
    max_points = max_points or int(os.getenv("USD_POINT_LIMIT", "100000"))
    reconstruction_path = job_dir / "reconstruction.json"
    if not reconstruction_path.exists():
        return []
    reconstruction = _read_json(reconstruction_path)
    points = reconstruction.get("points") or []
    if not points:
        return []

    stride = max(1, math.ceil(len(points) / max_points))
    sampled = points[::stride]
    usd_points = []
    usd_colors = []
    for point in sampled:
        x = float(point["x"])
        y = float(point["y"])
        z = float(point["z"])
        if not all(math.isfinite(value) for value in (x, y, z)):
            continue
        r = int(point.get("r", 255)) / 255.0
        g = int(point.get("g", 255)) / 255.0
        b = int(point.get("b", 255)) / 255.0
        usd_points.append(f"({x:.4f}, {y:.4f}, {z:.4f})")
        usd_colors.append(f"({r:.4f}, {g:.4f}, {b:.4f})")
    if not usd_points:
        return []

    return [
        '    def Points "ReconstructionPoints" {',
        f"        point3f[] points = [{', '.join(usd_points)}]",
        f"        color3f[] primvars:displayColor = [{', '.join(usd_colors)}]",
        '        uniform token primvars:displayColor:interpolation = "vertex"',
        f"        float[] widths = [{', '.join([os.getenv('USD_POINT_WIDTH', '0.0125')] * len(usd_points))}]",
        "    }",
    ]


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def _write_preview_html(job_dir: Path, preview_path: Path, scene: dict) -> None:
    image_src = preview_path.name if preview_path.parent == job_dir else str(preview_path)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Digital Twin Sim Preview</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #11151a; color: #e8edf2; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 24px; }}
    img {{ width: 100%; image-rendering: pixelated; background: #fff; border: 1px solid #2c333d; }}
    dl {{ display: grid; grid-template-columns: max-content 1fr; gap: 8px 16px; }}
    dt {{ color: #aab4c0; }}
    dd {{ margin: 0; }}
    code {{ color: #f2d57e; }}
  </style>
</head>
<body>
  <main>
    <h1>Digital Twin Sim Preview</h1>
    <img src="{image_src}" alt="Top-down occupancy preview">
    <dl>
      <dt>USD</dt><dd><code>sim/scene.usda</code></dd>
      <dt>Grid</dt><dd>{scene["grid"]["width"]} x {scene["grid"]["height"]} cells at {scene["resolution"]} m/cell</dd>
      <dt>Source</dt><dd>{scene["source"]}</dd>
      <dt>Scale</dt><dd>{scene["scale"]}</dd>
      <dt>Spawn</dt><dd>x={scene["spawn"]["x"]}, y={scene["spawn"]["y"]}, theta={scene["spawn"]["theta"]}</dd>
    </dl>
  </main>
</body>
</html>
"""
    (job_dir / "preview.html").write_text(html, encoding="utf-8")


def _dataclass_to_json(value):
    return asdict(value)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
