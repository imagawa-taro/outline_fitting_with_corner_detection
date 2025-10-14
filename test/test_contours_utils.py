import sys
import os
import numpy as np
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import contours_utils

def test_drop_small_contours():
    # 2つの輪郭: 1つは大きい矩形, 1つは小さい矩形
    big = np.array([[[0,0]], [[0,10]], [[10,10]], [[10,0]]], dtype=np.int32)
    small = np.array([[[0,0]], [[0,2]], [[2,2]], [[2,0]]], dtype=np.int32)
    contours = [big, small]
    filtered = contours_utils.drop_small_contours(contours, min_area=20)
    assert any(np.array_equal(big, c) for c in filtered)
    assert not any(np.array_equal(small, c) for c in filtered)
    # min_areaを下げれば両方残る
    filtered2 = contours_utils.drop_small_contours(contours, min_area=1)
    assert any(np.array_equal(big, c) for c in filtered2)
    assert any(np.array_equal(small, c) for c in filtered2)

def test_simplify_contour_with_corners_linearity():
    # 直線的な輪郭（矩形）
    rect = np.array([[[0,0]], [[0,10]], [[10,10]], [[10,0]], [[0,0]]], dtype=np.int32)
    corners = [0, 1, 2, 3, 4]
    simplified = contours_utils.simplify_contour_with_corners(rect, corners, linearity_threshold=0.95)
    # 簡略化しても矩形のまま
    assert simplified.shape[0] <= rect.shape[0]

def test_simplify_contour_with_corners_approx():
    # 曲線的な輪郭（半円＋直線）
    theta = np.linspace(0, np.pi, 10)
    arc = np.stack([np.cos(theta), np.sin(theta)], axis=1) * 10
    line = np.array([[10,0],[0,0]])
    contour = np.vstack([arc, line]).astype(np.int32).reshape(-1,1,2)
    corners = [0, 5, len(contour)-1]
    simplified = contours_utils.simplify_contour_with_corners(contour, corners, linearity_threshold=0.8)
    # 近似で点数が減る
    assert simplified.shape[0] < contour.shape[0]

def test_simplify_contour_with_few_corners():
    # コーナーが1つしかない場合はそのまま返す
    rect = np.array([[[0,0]], [[0,10]], [[10,10]], [[10,0]], [[0,0]]], dtype=np.int32)
    corners = [0]
    simplified = contours_utils.simplify_contour_with_corners(rect, corners)
    assert np.all(simplified == rect)
