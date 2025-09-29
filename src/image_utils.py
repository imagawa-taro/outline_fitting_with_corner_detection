import cv2
import numpy as np
from typing import List

def load_image_grayscale(image_path: str) -> np.ndarray:
    """
    画像をグレースケールで読み込む
    Args:
        image_path (str): 画像ファイルのパス
    Returns:
        ndarray: グレースケール画像
    Raises:
        FileNotFoundError: 画像ファイルが見つからない場合
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"画像ファイルが見つかりません。画像ファイル名を確認してください: {image_path}")
    return img

def extract_room_contours(image: np.ndarray, threshold: int = 225) -> List[np.ndarray]:
    """
    画像から部屋の輪郭を抽出する
    Args:
        image (ndarray): グレースケール画像
        threshold (int): 2値化の閾値
    Returns:
        list: 部屋の輪郭リスト
    """
    _, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        leaf_contours = [cnt for cnt, h in zip(contours, hierarchy) if h[2] == -1]
        return leaf_contours
    else:
        return contours
