from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MODEL = "gpt-5.5"
SUPPORTED_PRIMITIVES = {"box", "panel", "cylinder"}


MESH_SPEC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["object_label", "units", "primitives"],
    "properties": {
        "object_label": {"type": "string"},
        "units": {"type": "string", "enum": ["meters"]},
        "primitives": {
            "type": "array",
            "minItems": 1,
            "maxItems": 32,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "type", "center", "size", "rotation_euler", "color"],
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": sorted(SUPPORTED_PRIMITIVES)},
                    "center": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 3,
                        "items": {"type": "number"},
                    },
                    "size": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 3,
                        "items": {"type": "number"},
                    },
                    "rotation_euler": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 3,
                        "items": {"type": "number"},
                    },
                    "color": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 3,
                        "items": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
            },
        },
    },
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a Blender mesh from a visual reconstruction and reference image.")
    parser.add_argument("--job-dir", type=Path, required=True, help="Existing job directory containing reconstruction.json")
    parser.add_argument("--reference-image", type=Path, required=True, help="Reference RGB image sent to the VLM")
    parser.add_argument("--out", type=Path, default=None, help="Output mesh directory, defaults to JOB_DIR/mesh")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL), help="OpenAI vision model")
    parser.add_argument("--mesh-spec", type=Path, default=None, help="Use an existing mesh spec JSON instead of calling OpenAI")
    parser.add_argument("--no-run-blender", action="store_true", help="Generate files without executing Blender")
    args = parser.parse_args(argv)

    output_dir = args.out or args.job_dir / "mesh"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_geometry_metrics(args.job_dir / "reconstruction.json")
    (output_dir / "geometry_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    if args.mesh_spec:
        mesh_spec = json.loads(args.mesh_spec.read_text(encoding="utf-8"))
    else:
        mesh_spec = request_mesh_spec(args.reference_image, metrics, args.model)
    validate_mesh_spec(mesh_spec)
    mesh_spec_path = output_dir / "mesh_spec.json"
    mesh_spec_path.write_text(json.dumps(mesh_spec, indent=2, sort_keys=True), encoding="utf-8")

    script_path = output_dir / "generated_mesh.py"
    script_path.write_text(generate_blender_script(mesh_spec, output_dir), encoding="utf-8")
    usd_path = write_mesh_usd(args.job_dir, output_dir, mesh_spec)

    ran_blender = False
    if not args.no_run_blender:
        run_blender(script_path)
        ran_blender = True

    result = {
        "mesh_spec": str(mesh_spec_path),
        "script": str(script_path),
        "usd": str(usd_path),
        "ran_blender": ran_blender,
    }
    print(json.dumps(result, indent=2))
    return 0


def compute_geometry_metrics(reconstruction_path: Path) -> dict[str, Any]:
    payload = json.loads(reconstruction_path.read_text(encoding="utf-8"))
    raw_points = payload.get("points") or []
    if not raw_points:
        raise RuntimeError(f"Reconstruction has no points: {reconstruction_path}")

    xyz = np.array([[point["x"], point["y"], point["z"]] for point in raw_points], dtype=np.float32)
    colors = np.array(
        [[point.get("r", 255), point.get("g", 255), point.get("b", 255)] for point in raw_points],
        dtype=np.float32,
    )
    min_xyz = xyz.min(axis=0)
    max_xyz = xyz.max(axis=0)
    center = (min_xyz + max_xyz) / 2
    span = max_xyz - min_xyz
    centered = xyz - xyz.mean(axis=0)
    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    explained = (singular * singular) / np.sum(singular * singular)

    return {
        "source": payload.get("source", "unknown"),
        "coordinate_system": payload.get("coordinate_system", "unknown"),
        "point_count": int(len(xyz)),
        "bounds": {
            "min": _round_list(min_xyz),
            "max": _round_list(max_xyz),
            "center": _round_list(center),
            "span": _round_list(span),
        },
        "pca": {
            "axes": [_round_list(row) for row in axes[:3]],
            "explained_variance": _round_list(explained[:3]),
        },
        "height_percentiles": _percentiles(xyz[:, 2]),
        "depth_percentiles": _percentiles(xyz[:, 0]),
        "dominant_color_rgb": _round_list(np.median(colors, axis=0), digits=0),
    }


def request_mesh_spec(reference_image: Path, metrics: dict[str, Any], model: str) -> dict[str, Any]:
    from openai import OpenAI

    api_key = _load_openai_key()
    client = OpenAI(api_key=api_key)
    prompt = (
        "You convert a point-cloud reconstruction and a reference image into a clean primitive mesh spec for Blender. "
        "Use the image to infer object category and visible parts, but keep geometry conservative and centered around "
        "the provided metric bounds. Return only the requested JSON schema. Prefer composed primitives over arbitrary detail. "
        "For unknown objects, produce a simple proxy using boxes, panels, and cylinders."
        "\n\nGeometry metrics:\n"
        f"{json.dumps(metrics, indent=2, sort_keys=True)}"
    )
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": _image_data_url(reference_image)},
                ],
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "object_mesh_spec",
                "strict": True,
                "schema": MESH_SPEC_SCHEMA,
            }
        },
    )
    return json.loads(response.output_text)


def validate_mesh_spec(spec: dict[str, Any]) -> None:
    if not isinstance(spec, dict):
        raise ValueError("Mesh spec must be an object")
    if spec.get("units") != "meters":
        raise ValueError("Mesh spec units must be meters")
    primitives = spec.get("primitives")
    if not isinstance(primitives, list) or not primitives:
        raise ValueError("Mesh spec must contain at least one primitive")

    for primitive in primitives:
        if not isinstance(primitive, dict):
            raise ValueError("Primitive must be an object")
        if primitive.get("type") not in SUPPORTED_PRIMITIVES:
            raise ValueError(f"Unsupported primitive type: {primitive.get('type')}")
        _validate_vector(primitive.get("center"), "center", allow_zero=True)
        size = _validate_vector(primitive.get("size"), "size", allow_zero=False)
        if any(value <= 0 for value in size):
            raise ValueError("Primitive size values must be positive")
        if "rotation_euler" in primitive:
            _validate_vector(primitive["rotation_euler"], "rotation_euler", allow_zero=True)
        color = _validate_vector(primitive.get("color"), "color", allow_zero=True)
        if any(value < 0 or value > 1 for value in color):
            raise ValueError("Primitive color values must be in [0, 1]")


def generate_blender_script(spec: dict[str, Any], output_dir: Path) -> str:
    validate_mesh_spec(spec)
    obj_path = output_dir / "generated.obj"
    glb_path = output_dir / "generated.glb"
    lines = [
        "import bpy",
        "from mathutils import Vector",
        "",
        "bpy.ops.object.select_all(action='SELECT')",
        "bpy.ops.object.delete()",
        "",
        "def material(name, color):",
        "    mat = bpy.data.materials.new(name)",
        "    mat.diffuse_color = (color[0], color[1], color[2], 1.0)",
        "    return mat",
        "",
        "def add_box(name, center, size, rotation, color):",
        "    bpy.ops.mesh.primitive_cube_add(size=1, location=center, rotation=rotation)",
        "    obj = bpy.context.object",
        "    obj.name = name",
        "    obj.dimensions = size",
        "    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)",
        "    obj.data.materials.append(material(name + '_mat', color))",
        "    return obj",
        "",
        "def add_cylinder(name, center, size, rotation, color):",
        "    radius = max(size[0], size[1]) / 2.0",
        "    depth = size[2]",
        "    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=radius, depth=depth, location=center, rotation=rotation)",
        "    obj = bpy.context.object",
        "    obj.name = name",
        "    obj.scale.x = max(size[0], 1e-6) / max(size[1], 1e-6)",
        "    obj.data.materials.append(material(name + '_mat', color))",
        "    return obj",
        "",
    ]
    for index, primitive in enumerate(spec["primitives"]):
        name = _safe_name(str(primitive.get("name") or f"primitive_{index}"))
        center = _float_tuple(primitive["center"])
        size = _float_tuple(primitive["size"])
        rotation = _float_tuple(primitive.get("rotation_euler", [0.0, 0.0, 0.0]))
        color = _float_tuple(primitive["color"])
        if primitive["type"] == "cylinder":
            lines.append(f"add_cylinder({name!r}, {center}, {size}, {rotation}, {color})")
        else:
            lines.append(f"add_box({name!r}, {center}, {size}, {rotation}, {color})")
    lines.extend(
        [
            "",
            "bpy.ops.object.select_all(action='SELECT')",
            "try:",
            f"    bpy.ops.wm.obj_export(filepath={str(obj_path)!r})",
            "except Exception:",
            f"    bpy.ops.export_scene.obj(filepath={str(obj_path)!r})",
            f"bpy.ops.export_scene.gltf(filepath={str(glb_path)!r}, export_format='GLB')",
            "",
        ]
    )
    return "\n".join(lines)


def write_mesh_usd(job_dir: Path, output_dir: Path, spec: dict[str, Any]) -> Path:
    validate_mesh_spec(spec)
    sim_dir = output_dir / "sim"
    sim_dir.mkdir(parents=True, exist_ok=True)
    usd_path = sim_dir / "mesh_scene.usda"
    point_cloud = os.path.relpath(job_dir / "sim" / "scene.usda", sim_dir)
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "World" {',
        f'    string objectLabel = "{_usd_escape(str(spec.get("object_label", "object")))}"',
        '    def Xform "GeneratedMesh" {',
    ]
    for index, primitive in enumerate(spec["primitives"]):
        lines.extend(_usd_primitive_lines(primitive, index))
    lines.extend(
        [
            "    }",
            '    def Xform "SourcePointCloud" (',
            "        active = false",
            f"        prepend references = @{point_cloud}@",
            "    ) {",
            "    }",
            "}",
            "",
        ]
    )
    usd_path.write_text("\n".join(lines), encoding="utf-8")
    return usd_path


def run_blender(script_path: Path) -> None:
    blender = shutil.which("blender")
    if blender is None:
        raise RuntimeError(
            f"Blender is not installed or not on PATH. Generated script is available at {script_path}."
        )
    subprocess.run([blender, "--background", "--python", str(script_path)], check=True)


def _load_openai_key() -> str:
    for key_name in ("OPENAI_API_KEY", "OPENAI_KEY"):
        value = os.getenv(key_name)
        if value:
            return value
    for env_path in (Path.cwd() / ".env", Path(__file__).resolve().parent / ".env", Path.cwd().parent / ".env"):
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if not line or line.lstrip().startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() in {"OPENAI_API_KEY", "OPENAI_KEY"} and value.strip():
                return value.strip().strip("'\"")
    raise RuntimeError("Set OPENAI_KEY or OPENAI_API_KEY in the environment or .env")


def _image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"


def _validate_vector(value: Any, name: str, allow_zero: bool) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a 3-number array")
    output = [float(item) for item in value]
    if not allow_zero and any(item == 0 for item in output):
        raise ValueError(f"{name} values must be non-zero")
    if not all(math.isfinite(item) for item in output):
        raise ValueError(f"{name} values must be finite")
    return output


def _round_list(values: np.ndarray, digits: int = 4) -> list[float]:
    return [round(float(value), digits) for value in values.tolist()]


def _percentiles(values: np.ndarray) -> dict[str, float]:
    labels = ["p01", "p10", "p25", "p50", "p75", "p90", "p99"]
    percentiles = np.percentile(values, [1, 10, 25, 50, 75, 90, 99])
    return {label: round(float(value), 4) for label, value in zip(labels, percentiles)}


def _float_tuple(values: list[Any]) -> tuple[float, float, float]:
    return tuple(round(float(value), 6) for value in values)  # type: ignore[return-value]


def _usd_primitive_lines(primitive: dict[str, Any], index: int) -> list[str]:
    prim_type = "Cylinder" if primitive["type"] == "cylinder" else "Cube"
    name = _safe_name(str(primitive.get("name") or f"primitive_{index}"))
    center = _float_tuple(primitive["center"])
    size = _float_tuple(primitive["size"])
    rotation = tuple(math.degrees(value) for value in _float_tuple(primitive.get("rotation_euler", [0, 0, 0])))
    color = _float_tuple(primitive["color"])
    return [
        f'        def {prim_type} "{name}_{index:02d}" {{',
        f"            color3f[] primvars:displayColor = [({_fmt(color[0])}, {_fmt(color[1])}, {_fmt(color[2])})]",
        '            uniform token primvars:displayColor:interpolation = "constant"',
        f"            double3 xformOp:scale = ({_fmt(size[0] / 2)}, {_fmt(size[1] / 2)}, {_fmt(size[2] / 2)})",
        f"            double3 xformOp:rotateXYZ = ({_fmt(rotation[0])}, {_fmt(rotation[1])}, {_fmt(rotation[2])})",
        f"            double3 xformOp:translate = ({_fmt(center[0])}, {_fmt(center[1])}, {_fmt(center[2])})",
        '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]',
        "        }",
    ]


def _fmt(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".") or "0"


def _usd_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in value.strip())
    if cleaned and cleaned[0].isdigit():
        cleaned = f"Primitive_{cleaned}"
    return cleaned or "primitive"


if __name__ == "__main__":
    raise SystemExit(main())
