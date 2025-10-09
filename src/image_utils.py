import cv2
import numpy as np
from typing import List

def get_blur_image(image: np.ndarray, kernel: tuple = (9, 9)) -> np.ndarray:
    """
    画像をガウシアンブラー＆正規化して返す
    Args:
        image (np.ndarray): 入力画像
        kernel (tuple): ガウシアンブラーのカーネルサイズ
    Returns:
        np.ndarray: ブラー＆正規化後の画像
    """
    out = cv2.GaussianBlur(image.copy(), kernel, 0)
    out = cv2.normalize(out, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    return out


def get_silhouette(contours: List[np.ndarray], image_shape: tuple, kernel_size: tuple = (5, 5)) -> List[np.ndarray]:
    """
    輪郭リストから全体のconvex hull（シルエット）を求める
    Args:
        contours (List[np.ndarray]): 輪郭リスト
        image_shape (tuple): 画像の形状 (高さ, 幅)
    Returns:
        List[np.ndarray]: 凸包（シルエット）を1要素とするリスト
    """
    all_points = np.vstack(contours)
    hull = cv2.convexHull(all_points)
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    # maskをシュリンク
    kernel = np.ones(kernel_size, np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    return [mask]


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
