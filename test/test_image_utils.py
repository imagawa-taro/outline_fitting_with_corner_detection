import sys
import os
import numpy as np
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import image_utils
import cv2

def test_get_blur_image():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[5:15, 5:15] = 255
    blurred = image_utils.get_blur_image(img, kernel=(5,5))
    assert blurred.shape == img.shape
    assert blurred.dtype == np.uint8
    assert blurred.max() <= 255 and blurred.min() >= 0

def test_get_silhouette():
    cnt = np.array([[[0,0]], [[0,9]], [[9,9]], [[9,0]]], dtype=np.int32)
    mask_list = image_utils.get_silhouette([cnt], (10,10), kernel_size=(3,3))
    assert isinstance(mask_list, list)
    mask = mask_list[0]
    assert mask.shape == (10,10)
    assert mask.dtype == np.uint8
    assert mask.max() == 255

def test_extract_room_contours():
    img = np.zeros((20, 20), dtype=np.uint8)
    cv2.rectangle(img, (2,2), (17,17), 255, -1)
    contours = image_utils.extract_room_contours(img, threshold=127)
    assert isinstance(contours, list)
    assert all(isinstance(c, np.ndarray) for c in contours)
    assert any(len(c) >= 4 for c in contours)

def test_load_image_grayscale(tmp_path):
    # 一時ファイルに画像を書き込んでテスト
    img = np.zeros((10,10), dtype=np.uint8)
    img[2:8,2:8] = 200
    path = tmp_path / 'testimg.png'
    cv2.imwrite(str(path), img)
    loaded = image_utils.load_image_grayscale(str(path))
    assert loaded.shape == img.shape
    assert loaded.dtype == np.uint8
    assert np.allclose(loaded, img, atol=10)  # PNG保存で微妙な差が出る場合あり
    # 存在しないファイル
    with pytest.raises(FileNotFoundError):
        image_utils.load_image_grayscale(str(tmp_path / 'notfound.png'))
