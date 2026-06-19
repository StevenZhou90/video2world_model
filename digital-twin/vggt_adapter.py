from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run VGGT over extracted frames and write reconstruction artifacts.")
    parser.add_argument("--frames", type=Path, required=True, help="Directory of extracted video frames")
    parser.add_argument("--out", type=Path, required=True, help="Pipeline job output directory")
    parser.add_argument("--checkpoint", default="facebook/VGGT-1B", help="Hugging Face checkpoint")
    parser.add_argument("--max-frames", type=int, default=12, help="Evenly sample at most this many frames")
    parser.add_argument("--max-points", type=int, default=60000, help="Maximum reconstructed points to write")
    parser.add_argument("--confidence-percentile", type=float, default=55.0, help="Drop points below this confidence percentile")
    parser.add_argument("--voxel-size", type=float, default=0.025, help="Voxel size for visual point fusion; set 0 to disable")
    parser.add_argument("--outlier-radius", type=float, default=0.08, help="Radius for local-density outlier filtering; set 0 to disable")
    parser.add_argument("--min-neighbors", type=int, default=4, help="Minimum neighboring points required inside outlier radius")
    parser.add_argument("--preprocess-mode", choices=["crop", "pad"], default="crop", help="VGGT image preprocessing mode")
    parser.add_argument("--use-point-map", action="store_true", help="Use VGGT point-map output instead of depth unprojection")
    parser.add_argument(
        "--coordinate-system",
        choices=["isaac-z-up", "opencv"],
        default="isaac-z-up",
        help="Output coordinate convention. Isaac Z-up maps OpenCV x-right/y-down/z-forward to x-forward/y-left/z-up.",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    args = parser.parse_args(argv)

    frame_paths = _select_frames(_image_files(args.frames), args.max_frames)
    if not frame_paths:
        raise SystemExit(f"No image frames found in {args.frames}")

    reconstruction = run_vggt(
        frame_paths=frame_paths,
        output_dir=args.out,
        checkpoint=args.checkpoint,
        max_points=args.max_points,
        confidence_percentile=args.confidence_percentile,
        voxel_size=args.voxel_size,
        outlier_radius=args.outlier_radius,
        min_neighbors=args.min_neighbors,
        preprocess_mode=args.preprocess_mode,
        use_point_map=args.use_point_map,
        coordinate_system=args.coordinate_system,
        requested_device=args.device,
    )
    print(json.dumps({"source": reconstruction["source"], "frames": len(frame_paths), "points": len(reconstruction["points"])}, indent=2))
    return 0


def clean_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean an existing reconstruction point cloud and rebuild sim artifacts.")
    parser.add_argument("job_dir", type=Path, help="Existing job directory containing reconstruction.json")
    parser.add_argument("--out", type=Path, default=None, help="Output job directory, defaults to overwriting job_dir")
    parser.add_argument("--voxel-size", type=float, default=0.025, help="Voxel size for visual point fusion; set 0 to disable")
    parser.add_argument("--outlier-radius", type=float, default=0.08, help="Radius for local-density outlier filtering; set 0 to disable")
    parser.add_argument("--min-neighbors", type=int, default=4, help="Minimum neighboring points required inside outlier radius")
    args = parser.parse_args(argv)

    output_dir = args.out or args.job_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads((args.job_dir / "reconstruction.json").read_text(encoding="utf-8"))
    before = len(payload.get("points") or [])
    payload["points"] = clean_points(
        payload.get("points") or [],
        voxel_size=args.voxel_size,
        outlier_radius=args.outlier_radius,
        min_neighbors=args.min_neighbors,
    )
    _write_json(output_dir / "reconstruction.json", payload)
    _write_ply(output_dir / "point_cloud.ply", payload["points"])

    from sim_pipeline import build_map, export_usd, preview_scene

    build_map(output_dir)
    usd = export_usd(output_dir)
    preview = preview_scene(output_dir)
    print(json.dumps({"input_points": before, "output_points": len(payload["points"]), "usd": str(usd), "preview": str(preview)}, indent=2))
    return 0


def run_vggt(
    frame_paths: list[Path],
    output_dir: Path,
    checkpoint: str,
    max_points: int,
    confidence_percentile: float,
    voxel_size: float,
    outlier_radius: float,
    min_neighbors: int,
    preprocess_mode: str,
    use_point_map: bool,
    coordinate_system: str,
    requested_device: str,
) -> dict:
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = _resolve_device(torch, requested_device)
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    model = VGGT.from_pretrained(checkpoint).to(device)
    model.eval()

    images = load_and_preprocess_images([str(path) for path in frame_paths], mode=preprocess_mode).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=device == "cuda", dtype=dtype):
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    extrinsic_np = _tensor_to_numpy(extrinsic).squeeze(0)
    intrinsic_np = _tensor_to_numpy(intrinsic).squeeze(0)

    if use_point_map:
        world_points = _tensor_to_numpy(predictions["world_points"]).squeeze(0)
        confidence = _tensor_to_numpy(predictions["world_points_conf"]).squeeze(0)
    else:
        depth = _tensor_to_numpy(predictions["depth"]).squeeze(0)
        world_points = unproject_depth_map_to_point_map(depth, extrinsic_np, intrinsic_np)
        confidence = _tensor_to_numpy(predictions["depth_conf"]).squeeze(0)
    camera_to_world = closed_form_inverse_se3(extrinsic_np)[:, :3, :]

    if coordinate_system == "isaac-z-up":
        world_points = _opencv_points_to_isaac_z_up(world_points)
        camera_to_world = _opencv_camera_to_world_to_isaac_z_up(camera_to_world)

    colors = _tensor_to_numpy(images).transpose(0, 2, 3, 1)
    points = _sample_points(world_points, confidence, colors, max_points, confidence_percentile)
    points = clean_points(points, voxel_size=voxel_size, outlier_radius=outlier_radius, min_neighbors=min_neighbors)
    camera_poses = _camera_poses(frame_paths, output_dir, camera_to_world)

    reconstruction = {
        "source": f"vggt:{checkpoint}",
        "coordinate_system": coordinate_system,
        "scale": "relative",
        "camera_poses": camera_poses,
        "points": points,
    }
    _write_json(output_dir / "reconstruction.json", reconstruction)
    _write_ply(output_dir / "point_cloud.ply", points)
    return reconstruction


def _image_files(frames_dir: Path) -> list[Path]:
    return sorted(path for path in frames_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def _select_frames(frame_paths: list[Path], max_frames: int) -> list[Path]:
    if max_frames <= 0 or len(frame_paths) <= max_frames:
        return frame_paths
    indices = np.linspace(0, len(frame_paths) - 1, max_frames).round().astype(int)
    return [frame_paths[int(index)] for index in indices]


def _resolve_device(torch, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested not in {"cuda", "cpu"}:
        raise ValueError("--device must be auto, cuda, or cpu")
    return requested


def _opencv_points_to_isaac_z_up(points: np.ndarray) -> np.ndarray:
    # OpenCV/VGGT-style: x right, y down, z forward.
    # Isaac/USD-style: x forward, y left, z up.
    return np.stack((points[..., 2], -points[..., 0], -points[..., 1]), axis=-1)


def _opencv_camera_to_world_to_isaac_z_up(camera_to_world: np.ndarray) -> np.ndarray:
    basis = np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    converted = camera_to_world.copy()
    converted[:, :3, :3] = basis @ converted[:, :3, :3] @ basis.T
    converted[:, :3, 3] = (basis @ converted[:, :3, 3][..., None]).squeeze(-1)
    return converted


def _sample_points(
    world_points: np.ndarray,
    confidence: np.ndarray,
    colors: np.ndarray,
    max_points: int,
    confidence_percentile: float,
) -> list[dict]:
    flat_points = world_points.reshape(-1, 3)
    flat_conf = confidence.reshape(-1)
    flat_colors = colors.reshape(-1, 3)
    valid = np.isfinite(flat_points).all(axis=1) & np.isfinite(flat_conf) & (flat_conf > 0)
    if not valid.any():
        return []

    threshold = np.percentile(flat_conf[valid], confidence_percentile)
    selected = np.flatnonzero(valid & (flat_conf >= threshold))
    if len(selected) > max_points:
        selected = selected[np.linspace(0, len(selected) - 1, max_points).round().astype(int)]

    output: list[dict] = []
    for index in selected:
        point = flat_points[index]
        color = np.clip(flat_colors[index] * 255, 0, 255).astype(np.uint8)
        output.append(
            {
                "x": round(float(point[0]), 5),
                "y": round(float(point[1]), 5),
                "z": round(float(point[2]), 5),
                "confidence": round(float(flat_conf[index]), 5),
                "r": int(color[0]),
                "g": int(color[1]),
                "b": int(color[2]),
            }
        )
    return output


def clean_points(
    points: list[dict],
    voxel_size: float,
    outlier_radius: float,
    min_neighbors: int,
) -> list[dict]:
    cleaned = points
    if voxel_size > 0:
        cleaned = _voxel_downsample(cleaned, voxel_size)
    if outlier_radius > 0 and min_neighbors > 0:
        cleaned = _radius_outlier_filter(cleaned, outlier_radius, min_neighbors)
    return cleaned


def _voxel_downsample(points: list[dict], voxel_size: float) -> list[dict]:
    if not points:
        return []

    xyz, confidence, colors = _points_to_arrays(points)
    voxels = np.floor(xyz / voxel_size).astype(np.int64)
    unique_voxels, inverse = np.unique(voxels, axis=0, return_inverse=True)
    weights = np.clip(confidence, 1e-6, None)
    weight_sum = np.bincount(inverse, weights=weights)

    downsampled_xyz = np.column_stack(
        [
            np.bincount(inverse, weights=xyz[:, axis] * weights) / weight_sum
            for axis in range(3)
        ]
    )
    downsampled_colors = np.column_stack(
        [
            np.bincount(inverse, weights=colors[:, axis] * weights) / weight_sum
            for axis in range(3)
        ]
    )
    downsampled_confidence = np.bincount(inverse, weights=confidence * weights) / weight_sum

    order = np.lexsort((unique_voxels[:, 2], unique_voxels[:, 1], unique_voxels[:, 0]))
    return _arrays_to_points(downsampled_xyz[order], downsampled_confidence[order], downsampled_colors[order])


def _radius_outlier_filter(points: list[dict], radius: float, min_neighbors: int) -> list[dict]:
    if not points:
        return []

    xyz, confidence, colors = _points_to_arrays(points)
    cells = np.floor(xyz / radius).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, cell in enumerate(cells):
        buckets.setdefault((int(cell[0]), int(cell[1]), int(cell[2])), []).append(index)

    keep = np.zeros(len(points), dtype=bool)
    radius_sq = radius * radius
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for index, cell in enumerate(cells):
        candidate_indices: list[int] = []
        base = (int(cell[0]), int(cell[1]), int(cell[2]))
        for dx, dy, dz in offsets:
            candidate_indices.extend(buckets.get((base[0] + dx, base[1] + dy, base[2] + dz), []))
        if len(candidate_indices) <= min_neighbors:
            continue
        candidates = np.asarray(candidate_indices, dtype=np.int64)
        delta = xyz[candidates] - xyz[index]
        neighbor_count = int(np.count_nonzero(np.einsum("ij,ij->i", delta, delta) <= radius_sq)) - 1
        keep[index] = neighbor_count >= min_neighbors

    if not keep.any():
        return points
    return _arrays_to_points(xyz[keep], confidence[keep], colors[keep])


def _points_to_arrays(points: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.array([[point["x"], point["y"], point["z"]] for point in points], dtype=np.float32)
    confidence = np.array([point.get("confidence", 1.0) for point in points], dtype=np.float32)
    colors = np.array([[point.get("r", 255), point.get("g", 255), point.get("b", 255)] for point in points], dtype=np.float32)
    return xyz, confidence, colors


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


def _camera_poses(frame_paths: list[Path], output_dir: Path, camera_to_world: np.ndarray) -> list[dict]:
    metadata = _read_metadata(output_dir)
    poses: list[dict] = []
    for index, frame_path in enumerate(frame_paths):
        frame_meta = metadata.get(frame_path.name, {})
        matrix = camera_to_world[index]
        position = matrix[:3, 3]
        poses.append(
            {
                "frame_id": int(frame_meta.get("frame_id", index)),
                "timestamp_seconds": float(frame_meta.get("timestamp_seconds", 0.0)),
                "position": {"x": round(float(position[0]), 5), "y": round(float(position[1]), 5), "z": round(float(position[2]), 5)},
                "rotation_quat_xyzw": _rotation_matrix_to_quat_xyzw(matrix[:3, :3]),
            }
        )
    return poses


def _read_metadata(output_dir: Path) -> dict[str, dict]:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {Path(frame["path"]).name: frame for frame in metadata.get("frames", [])}


def _rotation_matrix_to_quat_xyzw(matrix: np.ndarray) -> list[float]:
    trace = float(np.trace(matrix))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        qw = 0.25 * s
        qx = (matrix[2, 1] - matrix[1, 2]) / s
        qy = (matrix[0, 2] - matrix[2, 0]) / s
        qz = (matrix[1, 0] - matrix[0, 1]) / s
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            qw = (matrix[2, 1] - matrix[1, 2]) / s
            qx = 0.25 * s
            qy = (matrix[0, 1] + matrix[1, 0]) / s
            qz = (matrix[0, 2] + matrix[2, 0]) / s
        elif axis == 1:
            s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            qw = (matrix[0, 2] - matrix[2, 0]) / s
            qx = (matrix[0, 1] + matrix[1, 0]) / s
            qy = 0.25 * s
            qz = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            qw = (matrix[1, 0] - matrix[0, 1]) / s
            qx = (matrix[0, 2] + matrix[2, 0]) / s
            qy = (matrix[1, 2] + matrix[2, 1]) / s
            qz = 0.25 * s
    return [round(float(qx), 6), round(float(qy), 6), round(float(qz), 6), round(float(qw), 6)]


def _tensor_to_numpy(value) -> np.ndarray:
    return value.detach().cpu().float().numpy()


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
