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


# 閉じた輪郭点列とコーナーインデックスを受け取り、下記の条件で新たな輪郭点列を生成する
# - 隣り合うコーナー点間の輪郭点が直線的である場合、その間の頂点を削除
# - 直線的でない場合は、cv2.approxPolyDPで近似した頂点に置換
def simplify_contour_with_corners(contour: np.ndarray, corner_indices: List[int],
                                linearity_threshold: float = 0.95,
                                approx_epsilon_ratio: float = 0.02) -> np.ndarray:
        """
        コーナー点に基づいて輪郭を簡略化する
        Args:
            contour (ndarray): 元の閉じた輪郭点列 (N,1,2)
            corner_indices (list): コーナー点のインデックスリスト
            linearity_threshold (float): 直線性の閾値（0〜1）。1に近いほど厳密に直線とみなす。
            approx_epsilon_ratio (float): cv2.approxPolyDPのepsilonの割合。輪郭長に対する比率。
        Returns:
            ndarray: 簡略化された輪郭点列 (M,1,2)
        """ 
        # コーナーが2つ未満の場合はcv2.approxPolyDPのみ適用
        if len(corner_indices) < 2:
            contour_for_arc = contour.astype(np.float32)
            return cv2.approxPolyDP(contour_for_arc, approx_epsilon_ratio/2 * cv2.arcLength(contour_for_arc, closed=False), closed=False)

        N = len(contour)
        simplified_points = [contour[corner_indices[0]][0]]  # 最初のコーナー点
        for i in range(len(corner_indices)):
            start_idx = corner_indices[i]
            end_idx = corner_indices[(i + 1) % len(corner_indices)]  # 次のコーナー、最後は最初に戻る
            if start_idx < end_idx:
                segment = contour[start_idx:end_idx + 1]
            else:
                segment = np.vstack([contour[start_idx:], contour[:end_idx + 1]])   
            # segmentを[M,1,2]から[M,2]に変換
            segment = segment.reshape(-1, 2)

            if segment.shape[0] == 0 or segment.shape[1] < 2:
                continue
            # 直線性の評価
            start_point = segment[0, :] # (2,)
            end_point = segment[-1, :]  # (2,)
            segment_vec = end_point - start_point
            segment_length = np.linalg.norm(segment_vec)    
            if segment_length < 1e-5:
                linearity = 1.0  # 始点と終点がほぼ同じ場合は直線とみなす
            else:
                segment_dir = segment_vec / segment_length
                vecs_to_start = segment - start_point  # (M,2)
                projections = np.dot(vecs_to_start, segment_dir)[:, np.newaxis] * segment_dir  # (M,2)
                dists = np.linalg.norm(vecs_to_start - projections, axis=1)  # (M,)
                max_dist = np.max(dists)
                linearity = 1.0 - (max_dist / (segment_length / 2))  # 正規化された直線性指標
            if linearity >= linearity_threshold:
                # 直線的なら終点のみ追加
                simplified_points.append(end_point)
            else:
                # 直線的でないなら近似を適用
                segment_for_arc = segment.astype(np.float32)
                epsilon = approx_epsilon_ratio * cv2.arcLength(segment_for_arc, closed=False)
                approx = cv2.approxPolyDP(segment_for_arc, epsilon, closed=False)
                # 近似頂点を追加（始点は既に追加済みのため2番目の点から始める）
                for point in approx[1:]:
                    simplified_points.append(point[0])
        # 最後の点は重複するので削除
        simplified_points = simplified_points[1:]
        if len(simplified_points) == 0:
            return contour
        simplified_contour = np.array(simplified_points, dtype=contour.dtype).reshape(-1, 1, 2)
        return simplified_contour
