import cv2
import numpy as np
from typing import List

def drop_small_contours(contours: List[np.ndarray], min_area: float = 100.0) -> List[np.ndarray]:
    """
    面積が小さい輪郭を削除する
    Args:
        contours (list): 元の輪郭リスト
        min_area (float): 保持する輪郭の最小面積（ピクセル）
    Returns:
        list: フィルタリングされた輪郭リスト
    """
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            filtered_contours.append(contour)
    return filtered_contours
