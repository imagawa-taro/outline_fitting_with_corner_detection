import numpy as np
import matplotlib.pyplot as plt
from typing import List

def postprocessing(new_contours: List[np.ndarray], image_shape: tuple) -> float:
    """
    new_contoursに含まれるエッジ情報の統計処理を行い、平均エッジ長を返す
    Args:
        new_contours (List[np.ndarray]): 輪郭リスト (各輪郭は (N, 1, 2) ndarray)
    Returns:
        float: 平均エッジ長（エッジが存在しない場合は0.0）
    """
    h, w = image_shape[:2]
    hist_x = np.zeros(w, dtype=np.float32)
    hist_y = np.zeros(h, dtype=np.float32)

    for cnt in new_contours:
        # エッジの向きが垂直に近い場合に始点から終点までの連続したx座標をhist_xに加算
        for i in range(len(cnt)):
            p1 = cnt[i][0]
            p2 = cnt[(i + 1) % len(cnt)][0]
            # エッジの長さを計算
            edge_length = np.linalg.norm(p2 - p1)
            #エッジの長さが5未満の場合はスキップ
            if edge_length < 5:
                continue
            # エッジの角度を計算し、閾値で判定
            angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi
            angle_margin = 8  # 角度の閾値マージン
            if abs(angle) < angle_margin or abs(angle) > 180 - angle_margin:
                # 水平に近いエッジの場合, hist_yのp1[0]からp2[0]まで区間に1加算
                y_start = int(min(p1[1], p2[1]))
                y_end = int(max(p1[1], p2[1]))
                hist_y[y_start:y_end + 1] += 1
            elif abs(angle - 90) < angle_margin or abs(angle + 90) < angle_margin:
                # 垂直に近いエッジの場合, hist_xのp1[0]からp2[0]まで区間に1加算
                x_start = int(min(p1[0], p2[0]))
                x_end = int(max(p1[0], p2[0]))
                hist_x[x_start:x_end + 1] += 1
    # histogramの描画
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Vertical Edge Histogram")
    plt.bar(range(w), hist_x, width=1)
    plt.subplot(1, 2, 2)
    plt.title("Horizontal Edge Histogram")
    plt.bar(range(h), hist_y, width=1)
    plt.show()

    return hist_x, hist_y
