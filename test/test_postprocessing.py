import sys
import os
import numpy as np
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import postprocessing

def test_postprocessing_rectangle():
    # 10x10の矩形画像と輪郭
    img = np.zeros((10, 10), dtype=np.uint8)
    img[2:8, 2:8] = 255
    contour = np.array([[[2,2]], [[2,7]], [[7,7]], [[7,2]]], dtype=np.int32)
    silhouette = [np.ones((10,10), dtype=np.uint8)*255]
    result = postprocessing.postprocessing([contour], img, silhouette)
    assert isinstance(result, list)
    assert result[0].shape[1:] == (1,2)

def test_postprocessing_min_edge_length():
    # min_edge_lengthを大きくして全スキップ
    img = np.zeros((10, 10), dtype=np.uint8)
    contour = np.array([[[2,2]], [[2,3]], [[3,3]], [[3,2]]], dtype=np.int32)
    silhouette = [np.ones((10,10), dtype=np.uint8)*255]
    result = postprocessing.postprocessing([contour], img, silhouette, min_edge_length=100)
    assert isinstance(result, list)

def test_edge_alignment_nan_skip():
    # v_sum, h_sumに全ゼロを含めてnanスキップ分岐を通す
    contours = [np.array([[[2,2]], [[2,7]], [[7,7]], [[7,2]]], dtype=np.int32)]
    index_x = [(0,0)]
    index_y = [(0,1)]
    v_peaks = [0]
    h_peaks = [0]
    v_sum = np.zeros(10)
    h_sum = np.zeros(10)
    with np.errstate(invalid='ignore', divide='ignore'):
        new_contours, v_means, h_means = postprocessing.edge_alignment(contours, index_x, index_y, v_peaks, h_peaks, v_sum, h_sum)
    assert isinstance(new_contours, list)
    assert isinstance(v_means, list)
    assert isinstance(h_means, list)
