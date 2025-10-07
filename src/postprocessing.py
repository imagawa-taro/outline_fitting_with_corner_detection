import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def postprocessing(contours: List[np.ndarray], image: np.ndarray, silhouette: List[np.ndarray]) -> float:
    """
    new_contoursに含まれるエッジ情報の統計処理を行う
    Args:
        new_contours (List[np.ndarray]): 輪郭リスト (各輪郭は (N, 1, 2) ndarray)
    Returns:
        float: 平均エッジ長（エッジが存在しない場合は0.0）
    """
    # 画像の輝度値の縦・横累積値を計算
    inv_image = 255 - image  # 輝度反転
    v_sum = np.sum(inv_image, axis=0).astype(float)  # 縦方向の輝度値の合計
    h_sum = np.sum(inv_image, axis=1).astype(float)  # 横方向の輝度値の合計
    v_sum /= 5000.0  # 輝度値を0-255から0-1に正規化
    h_sum /= 5000.0  # 輝度値を0-255から0-1に正規化

    # new_contoursの各エッジを走査し、hist_x, hist_yにエッジ情報を加算
    h, w = image.shape[:2]
    hist_x = np.zeros(w, dtype=np.float32)
    hist_y = np.zeros(h, dtype=np.float32)
    # 角度が条件に合うエッジの始点と終点のリスト
    index_x = []
    index_y = []
    for c_idx, cnt in enumerate(contours):
        # エッジの向きが垂直に近い場合に始点から終点までの連続したx座標をhist_xに加算
        for i in range(len(cnt)):
            p1 = cnt[i][0]
            p2 = cnt[(i + 1) % len(cnt)][0]
            # p1, p2がシルエットの内側にある場合のみ処理を行う
            if silhouette[0][int(p1[1]), int(p1[0])] == 0 or silhouette[0][int(p2[1]), int(p2[0])] == 0:
                continue

            # エッジの長さを計算
            edge_length = np.linalg.norm(p2 - p1)
            #エッジの長さが5未満の場合はスキップ
            if edge_length < 3:
                continue
            # エッジの角度を計算し、閾値で判定
            angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi
            angle_margin = 15  # 角度の閾値マージン
            if abs(angle) < angle_margin or abs(angle) > 180 - angle_margin:
                # 水平に近いエッジの場合, hist_yのp1[0]からp2[0]まで区間に1加算
                y_start = int(min(p1[1], p2[1]))
                y_end = int(max(p1[1], p2[1]))
                hist_y[y_start:y_end + 1] += 1
                index_y.append((c_idx, i))  # 輪郭インデックスとエッジインデックスを保存    
                index_y.append((c_idx, (i + 1) % len(cnt)))  # 輪郭インデックスとエッジインデックスを保存    
            elif abs(angle - 90) < angle_margin or abs(angle + 90) < angle_margin:
                # 垂直に近いエッジの場合, hist_xのp1[0]からp2[0]まで区間に1加算
                x_start = int(min(p1[0], p2[0]))
                x_end = int(max(p1[0], p2[0]))
                hist_x[x_start:x_end + 1] += 1
                index_x.append((c_idx, i))  # 輪郭インデックスとエッジインデックスを保存
                index_x.append((c_idx, (i + 1) % len(cnt)))  # 輪郭インデックスとエッジインデックスを保存

    # hist_x, hist_yのピークを検出
    from scipy.signal import find_peaks
    v_peaks, _ = find_peaks(hist_x, height=1, distance=2)
    h_peaks, _ = find_peaks(hist_y, height=1, distance=2)
    v_peak_coords = [(x, hist_x[x]) for x in v_peaks]
    h_peak_coords = [(y, hist_y[y]) for y in h_peaks]
    
    # edge_alignment
    new_contours = contours.copy()
    new_contours, v_means, h_means = edge_alignment(contours,index_x, index_y, v_peaks, h_peaks, v_sum, h_sum)

    # histogramの描画
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.title("Vertical Edge Histogram")

    # plt.bar(range(w), v_sum, width=1)
    # plt.bar(range(w), hist_x, width=1)
    # if v_peak_coords:
    #     plt.scatter(*zip(*v_peak_coords), color='red')
    # if v_means:
    #     plt.scatter(v_means, [0]*len(v_means), color='blue')

    # plt.subplot(1, 2, 2)
    # plt.title("Horizontal Edge Histogram")

    # plt.bar(range(h), h_sum, width=1)
    # plt.bar(range(h), hist_y, width=1)
    # if h_peak_coords:
    #     plt.scatter(*zip(*h_peak_coords), color='red')
    # if h_means:
    #     plt.scatter(h_means, [0]*len(h_means), color='blue')

    return new_contours


# edge_alignment
'''
全体のエッジを整列
'''
def edge_alignment(contours: List[np.ndarray], index_x: List[Tuple[int, int]], index_y: List[Tuple[int, int]], v_peaks: List[int], h_peaks: List[int], v_sum: np.ndarray, h_sum: np.ndarray):
    new_contours = contours.copy()
    window_size = 2
    # x方向の処理
    v_means = []
    for peak in v_peaks:
        # peak付近のv_sumの重心を計算
        v_mean = np.sum(np.arange(max(0, peak - window_size), min(len(v_sum), peak + window_size + 1)) * 
                        v_sum[max(0, peak - window_size):min(len(v_sum), peak + window_size + 1)]) / np.sum(
                        v_sum[max(0, peak - window_size):min(len(v_sum), peak + window_size + 1)])
        # v_meanがnanになる場合はスキップ
        if np.isnan(v_mean):
            continue
        v_means.append(v_mean)
        # x座標がpeak近傍のindex_xを探す
        for (c_ind, ii) in index_x:
            xx = new_contours[c_ind][ii][0][0]
            if abs(xx - peak) < 3:  # 5ピクセル以内なら
                new_contours[c_ind][ii][0][0] = v_mean

    # y方向の処理
    h_means = []
    for peak in h_peaks:
        # peak付近のh_sumの重心を計算
        h_mean = np.sum(np.arange(max(0, peak - window_size), min(len(h_sum), peak + window_size + 1)) * 
                        h_sum[max(0, peak - window_size):min(len(h_sum), peak + window_size + 1)]) / np.sum(
                        h_sum[max(0, peak - window_size):min(len(h_sum), peak + window_size + 1)])
        # h_meanがnanになる場合はスキップ
        if np.isnan(h_mean):
            continue
        h_means.append(h_mean)
        # y座標がpeak近傍のindex_yを探す
        for (c_ind, ii) in index_y:
            yy = new_contours[c_ind][ii][0][1]
            if abs(yy - peak) < 3:  # 3ピクセル以内なら
                new_contours[c_ind][ii][0][1] = h_mean
    return new_contours, v_means, h_means
