import sys
import os
import numpy as np
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from corner_detection import curvature_from_closed_contour, detect_corners_by_curvature, is_line_segment

def test_curvature_circle():
    R = 5
    N = 100
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    s_u, kappa = curvature_from_closed_contour(x, y, resample_points=N)
    assert np.allclose(np.mean(kappa), 1/R, rtol=0.05)

def test_detect_corners_polygon():
    # 正六角形のコーナー検出
    N = 6
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    result = detect_corners_by_curvature(x, y, resample_points=60, angle_range_deg=(50, 130))
    # コーナー数が6個前後
    if isinstance(result, dict):
        corners = result['corner_indices']
    else:
        corners = result[0]  # tupleの最初がcorner_indices
    assert 4 <= len(corners) <= 8

def test_is_line_segment():
    # 直線区間と曲線区間の判定
    # 完全な直線のみでテスト
    x = np.linspace(0, 1, 50)
    y = np.zeros(50)
    s_u, kappa = curvature_from_closed_contour(x, y, resample_points=50)
    corners = np.array([0, 49])
    flags = is_line_segment(s_u, kappa, corners, threshold=0.1)
    assert flags[0]  # 唯一の区間は直線

def test_invalid_input():
    # 入力長さ不一致
    x = np.arange(10)
    y = np.arange(9)
    with pytest.raises(ValueError):
        curvature_from_closed_contour(x, y)
    # 点数不足
    x = np.arange(4)
    y = np.arange(4)
    with pytest.raises(ValueError):
        curvature_from_closed_contour(x, y)
