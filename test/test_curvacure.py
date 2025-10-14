
import sys
import os
import numpy as np
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from curvature import curvature_from_closed_contour

def test_curvature_circle():
    # 半径Rの円の理論曲率は1/R
    R = 10
    N = 100
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    s_u, kappa = curvature_from_closed_contour(x, y, resample_points=N)
    # 曲率の平均が理論値1/Rに近い
    assert np.allclose(np.mean(kappa), 1/R, rtol=0.05)

def test_curvature_polygon():
    # 正六角形の曲率（角部以外はほぼ0、角部で大きい）
    N = 6
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    s_u, kappa = curvature_from_closed_contour(x, y, resample_points=60)
    # 角部で大きな曲率、辺部で小さい
    assert np.max(np.abs(kappa)) > 1
    assert np.median(np.abs(kappa)) < 0.5

def test_invalid_input_length():
    # x, y長さ不一致
    x = np.arange(10)
    y = np.arange(9)
    with pytest.raises(ValueError):
        curvature_from_closed_contour(x, y)

def test_too_few_points():
    # 点数が少なすぎる
    x = np.arange(4)
    y = np.arange(4)
    with pytest.raises(ValueError):
        curvature_from_closed_contour(x, y)
