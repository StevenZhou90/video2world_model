from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from sim_pipeline import build_map, export_usd, preview_scene
from vggt_adapter import clean_points


FREIBURG3_INTRINSICS = {
    "fx": 535.4,
    "fy": 539.2,
    "cx": 320.1,
    "cy": 247.6,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a point-cloud USD scene from aligned RGB-D video frames.")
    parser.add_argument("--rgb-video", type=Path, required=True)
    parser.add_argument("--depth-video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=416, help="Video frame index to reconstruct")
    parser.add_argument("--crop", default=None, help="Optional crop x1,y1,x2,y2")
    parser.add_argument("--brightness-threshold", type=int, default=170, help="Keep darker foreground pixels")
    parser.add_argument("--depth-scale", type=float, default=256.0 / 5000.0, help="Convert raw depth pixel values to meters")
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=4.2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--outlier-radius", type=float, default=0.035)
    parser.add_argument("--min-neighbors", type=int, default=3)
    parser.add_argument("--usd-point-width", default="0.006")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    reference_frame_path = args.out / "reference_frame.jpg"
    cv2.imwrite(str(reference_frame_path), _read_frame(args.rgb_video, args.frame))
    points = reconstruct_rgbd_frame(
        rgb_video=args.rgb_video,
        depth_video=args.depth_video,
        frame_index=args.frame,
        crop=_parse_crop(args.crop),
        brightness_threshold=args.brightness_threshold,
        depth_scale=args.depth_scale,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        stride=args.stride,
    )
    points = clean_points(
        points,
        voxel_size=args.voxel_size,
        outlier_radius=args.outlier_radius,
        min_neighbors=args.min_neighbors,
    )
    reconstruction = {
        "source": f"tum-rgbd:{args.rgb_video.name}:{args.depth_video.name}:frame-{args.frame}",
        "coordinate_system": "isaac-z-up",
        "scale": "metric-ish",
        "camera_poses": [
            {
                "frame_id": args.frame,
                "timestamp_seconds": round(args.frame / 30.0, 6),
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        ],
        "points": points,
    }
    _write_json(args.out / "reconstruction.json", reconstruction)
    _write_ply(args.out / "point_cloud.ply", points)

    build_map(args.out)
    import os

    previous_width = os.environ.get("USD_POINT_WIDTH")
    os.environ["USD_POINT_WIDTH"] = args.usd_point_width
    try:
        usd = export_usd(args.out)
    finally:
        if previous_width is None:
            os.environ.pop("USD_POINT_WIDTH", None)
        else:
            os.environ["USD_POINT_WIDTH"] = previous_width
    preview = preview_scene(args.out)

    print(
        json.dumps(
            {
                "points": len(points),
                "reference_frame": str(reference_frame_path),
                "usd": str(usd),
                "preview": str(preview),
            },
            indent=2,
        )
    )
    return 0


def reconstruct_rgbd_frame(
    rgb_video: Path,
    depth_video: Path,
    frame_index: int,
    crop: tuple[int, int, int, int] | None,
    brightness_threshold: int,
    depth_scale: float,
    min_depth: float,
    max_depth: float,
    stride: int,
) -> list[dict]:
    rgb = _read_frame(rgb_video, frame_index)
    depth_frame = _read_frame(depth_video, frame_index)
    depth_raw = depth_frame[:, :, 0] if depth_frame.ndim == 3 else depth_frame
    depth_m = depth_raw.astype(np.float32) * depth_scale

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    mask = (gray <= brightness_threshold) & (depth_m >= min_depth) & (depth_m <= max_depth)
    if crop is not None:
        x1, y1, x2, y2 = crop
        crop_mask = np.zeros(mask.shape, dtype=bool)
        crop_mask[max(0, y1) : min(mask.shape[0], y2), max(0, x1) : min(mask.shape[1], x2)] = True
        mask &= crop_mask
    mask = _largest_component(mask)
    if stride > 1:
        stride_mask = np.zeros(mask.shape, dtype=bool)
        stride_mask[::stride, ::stride] = True
        mask &= stride_mask

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise RuntimeError("RGB-D mask produced no points")

    z_cam = depth_m[ys, xs]
    x_cam = (xs.astype(np.float32) - FREIBURG3_INTRINSICS["cx"]) * z_cam / FREIBURG3_INTRINSICS["fx"]
    y_cam = (ys.astype(np.float32) - FREIBURG3_INTRINSICS["cy"]) * z_cam / FREIBURG3_INTRINSICS["fy"]
    xyz = np.column_stack((z_cam, -x_cam, -y_cam))
    xyz[:, 0] -= float(np.median(xyz[:, 0]))
    xyz[:, 1] -= float(np.median(xyz[:, 1]))
    xyz[:, 2] -= float(np.min(xyz[:, 2]))

    colors = rgb[ys, xs, ::-1]
    confidence = np.ones(len(xs), dtype=np.float32)
    return _arrays_to_points(xyz, confidence, colors)


def _read_frame(video_path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    return frame


def _largest_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8) * 255
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    if count <= 1:
        return mask_u8 > 0
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def _parse_crop(value: str | None) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--crop must be x1,y1,x2,y2")
    return tuple(parts)  # type: ignore[return-value]


def _arrays_to_points(xyz: np.ndarray, confidence: np.ndarray, colors: np.ndarray) -> list[dict]:
    output: list[dict] = []
    colors = np.clip(np.rint(colors), 0, 255).astype(np.uint8)
    for index, point in enumerate(xyz):
        output.append(
            {
                "x": round(float(point[0]), 5),
                "y": round(float(point[1]), 5),
                "z": round(float(point[2]), 5),
                "confidence": round(float(confidence[index]), 5),
                "r": int(colors[index, 0]),
                "g": int(colors[index, 1]),
                "b": int(colors[index, 2]),
            }
        )
    return output


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_ply(path: Path, points: list[dict]) -> None:
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property float confidence",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    lines.extend(f"{p['x']} {p['y']} {p['z']} {p['confidence']} {p['r']} {p['g']} {p['b']}" for p in points)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
