from __future__ import annotations

from vggt_adapter import clean_points


def test_clean_points_voxel_fuses_dense_points_and_removes_outlier():
    points = [
        {"x": 0.0, "y": 0.0, "z": 0.0, "confidence": 1.0, "r": 10, "g": 20, "b": 30},
        {"x": 0.01, "y": 0.0, "z": 0.0, "confidence": 1.0, "r": 30, "g": 40, "b": 50},
        {"x": 0.08, "y": 0.0, "z": 0.0, "confidence": 1.0, "r": 50, "g": 60, "b": 70},
        {"x": 0.09, "y": 0.0, "z": 0.0, "confidence": 1.0, "r": 70, "g": 80, "b": 90},
        {"x": 4.0, "y": 4.0, "z": 4.0, "confidence": 1.0, "r": 255, "g": 255, "b": 255},
    ]

    cleaned = clean_points(points, voxel_size=0.05, outlier_radius=0.12, min_neighbors=1)

    assert len(cleaned) == 2
    assert cleaned[0]["x"] == 0.005
    assert cleaned[0]["r"] == 20
    assert cleaned[1]["x"] == 0.085
    assert all(point["x"] < 1.0 for point in cleaned)


def test_clean_points_returns_original_when_outlier_filter_would_drop_everything():
    points = [
        {"x": 0.0, "y": 0.0, "z": 0.0, "confidence": 1.0},
        {"x": 1.0, "y": 1.0, "z": 1.0, "confidence": 1.0},
    ]

    cleaned = clean_points(points, voxel_size=0.0, outlier_radius=0.1, min_neighbors=2)

    assert cleaned == points
