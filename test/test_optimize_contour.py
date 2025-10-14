import sys
import os
import numpy as np
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from optimize_contour import Param, cost_fn2, optimize_loop, optimize_contour

def test_param_class():
    p = Param(lambda_data=2, lambda_pos=3, lambda_angle=4)
    assert p.lambda_data == 2
    assert p.lambda_pos == 3
    assert p.lambda_angle == 4

def test_cost_fn2_basic():
    img = np.zeros((20, 20), dtype=np.uint8)
    pts = np.array([[5,5],[5,15],[15,15],[15,5]])
    params = Param(lambda_data=1, lambda_pos=1, lambda_angle=1)
    cost = cost_fn2(pts.ravel(), 4, img, pts, params)
    assert isinstance(cost, float)

def test_optimize_loop_runs():
    img = np.zeros((20, 20), dtype=np.uint8)
    cv2 = __import__('cv2')
    cv2.rectangle(img, (6,6), (13,13), 255, -1)
    pts = np.array([[6,6],[6,13],[13,13],[13,6]])
    params = Param(lambda_data=1, lambda_pos=0.1, lambda_angle=1)
    opt_pts, result = optimize_loop(pts, img, params, maxiter=10)
    assert opt_pts.shape == pts.shape
    assert hasattr(result, 'success')

def test_optimize_contour_shape():
    img = np.zeros((20, 20), dtype=np.uint8)
    cv2 = __import__('cv2')
    cv2.rectangle(img, (6,6), (13,13), 255, -1)
    pts = np.array([[6,6],[6,13],[13,13],[13,6]])
    params = Param(lambda_data=1, lambda_pos=0.1, lambda_angle=1)
    opt_contours, result = optimize_contour(pts, img, params)
    assert isinstance(opt_contours, list)
    assert opt_contours[0].ndim == 3
    assert hasattr(result, 'success')
