import json

import pytest

from vlm_mesh_adapter import (
    compute_geometry_metrics,
    generate_blender_script,
    validate_mesh_spec,
    write_mesh_usd,
    _load_openai_key,
)


def _valid_spec():
    return {
        "object_label": "cabinet",
        "units": "meters",
        "primitives": [
            {
                "name": "cabinet_body",
                "type": "box",
                "center": [0.0, 0.0, 0.5],
                "size": [0.8, 0.4, 1.0],
                "rotation_euler": [0.0, 0.0, 0.0],
                "color": [0.72, 0.7, 0.66],
            },
            {
                "name": "round_handle",
                "type": "cylinder",
                "center": [0.0, -0.22, 0.75],
                "size": [0.08, 0.08, 0.04],
                "rotation_euler": [1.5708, 0.0, 0.0],
                "color": [0.08, 0.08, 0.08],
            },
        ],
    }


def test_compute_geometry_metrics_summarizes_reconstruction(tmp_path):
    reconstruction = {
        "source": "tum-rgbd:test",
        "coordinate_system": "isaac-z-up",
        "points": [
            {"x": -1.0, "y": 0.0, "z": 0.0, "r": 255, "g": 240, "b": 230},
            {"x": 0.0, "y": 1.0, "z": 0.5, "r": 240, "g": 235, "b": 220},
            {"x": 1.0, "y": -1.0, "z": 1.0, "r": 230, "g": 220, "b": 210},
            {"x": 0.5, "y": 0.5, "z": 1.5, "r": 220, "g": 210, "b": 200},
        ],
    }
    path = tmp_path / "reconstruction.json"
    path.write_text(json.dumps(reconstruction), encoding="utf-8")

    metrics = compute_geometry_metrics(path)

    assert metrics["point_count"] == 4
    assert metrics["bounds"]["min"] == [-1.0, -1.0, 0.0]
    assert metrics["bounds"]["max"] == [1.0, 1.0, 1.5]
    assert metrics["bounds"]["span"] == [2.0, 2.0, 1.5]
    assert metrics["coordinate_system"] == "isaac-z-up"
    assert len(metrics["pca"]["axes"]) == 3


def test_validate_mesh_spec_rejects_non_visual_primitives():
    spec = _valid_spec()
    validate_mesh_spec(spec)

    bad_spec = _valid_spec()
    bad_spec["primitives"][0]["type"] = "triangle_mesh"

    with pytest.raises(ValueError, match="Unsupported primitive"):
        validate_mesh_spec(bad_spec)


def test_generate_blender_script_builds_primitives_and_exports(tmp_path):
    script = generate_blender_script(_valid_spec(), tmp_path)

    assert "add_box('cabinet_body'" in script
    assert "add_cylinder('round_handle'" in script
    assert "bpy.ops.wm.obj_export" in script
    assert "bpy.ops.export_scene.gltf" in script


def test_write_mesh_usd_writes_direct_isaac_primitives(tmp_path):
    job_dir = tmp_path / "job"
    (job_dir / "sim").mkdir(parents=True)
    (job_dir / "sim" / "scene.usda").write_text("#usda 1.0\n", encoding="utf-8")
    output_dir = job_dir / "mesh"

    usd = write_mesh_usd(job_dir, output_dir, _valid_spec())
    text = usd.read_text(encoding="utf-8")

    assert 'upAxis = "Z"' in text
    assert 'def Cube "cabinet_body_00"' in text
    assert 'def Cylinder "round_handle_01"' in text
    assert 'def Xform "SourcePointCloud" (' in text
    assert "active = false" in text
    assert "prepend references = @../../sim/scene.usda@" in text


def test_load_openai_key_reads_local_dotenv_without_printing_secret(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OPENAI_KEY='sk-test-fake'\n", encoding="utf-8")

    assert _load_openai_key() == "sk-test-fake"
