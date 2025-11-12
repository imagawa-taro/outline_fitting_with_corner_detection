'''フィッティングの最適化ルーチン'''
from typing import Tuple
import numpy as np
from scipy.optimize import minimize

class Param:
    """最適化パラメータクラス"""
    def __init__(self, lambda_data=1.0, lambda_pos=1.0, lambda_angle=100):
        self.lambda_data = lambda_data
        self.lambda_pos = lambda_pos
        self.lambda_angle = lambda_angle


def cost_fn2(xvec: np.ndarray, N: int, image: np.ndarray, init_points: np.ndarray, params: Param) -> float:
    pts = xvec.reshape((N, 2))
    # ptsの内分点を追加して補間点を増やす
    new_pts = []
    wp = []
    for i in range(N):
        p1 = pts[i]
        p2 = pts[(i+1)%N]
        new_pts.append(p1)
        wp.append(10)  # p1の重みは1
        mid_pt = (p1 + p2) / 2
        new_pts.append(mid_pt)
        wp.append(np.linalg.norm(p1 - p2))  # 中間点の重みはp1とp2の距離
    pts2 = np.array(new_pts, dtype=float)
    wp = np.array(wp, dtype=float)
    pts2 = np.nan_to_num(pts2)
    pts2[~np.isfinite(pts2)] = 0
    h, w = image.shape[:2]
    # pts2をクリップ処理
    pts2_clipped = np.clip(pts2, [0, 0], [w-1, h-1])
    pts2_rounded = np.round(pts2_clipped)
    pts2_int = pts2_rounded.astype(np.int32)
    coords = np.clip(pts2_int, [0, 0], [w-1, h-1])
    
    pixel_values = image[coords[:,1], coords[:,0]] / 255.0  # 白黒画像を想定
    # pixelvalueを２乗
    pixel_values = pixel_values**2
    
    # 重みを反映してコスト計算
    # cost1 = np.sum(pixel_values) * params.lambda_data
    cost1 = np.sum(pixel_values * wp) * params.lambda_data

    # 位置のコスト
    diffs = pts.reshape(-1, 2) - init_points.reshape(-1, 2)
    dist = np.linalg.norm(diffs, axis=1)
    cost2 = params.lambda_pos * dist.sum()

    # 90度単位に近い場合は寄せる項
    p_next = np.roll(pts, -1, axis=0)
    diffs = p_next - pts
    angles = np.arctan2(diffs[:,1], diffs[:,0])
    angles_mod = np.mod(angles, np.pi/2)
    angle_threshold = 10 * np.pi/180
    indices = np.where((angles_mod < angle_threshold) | (angles_mod > np.pi/2 - angle_threshold))[0]
    # indicesに対応するdiffsのx,y成分の小さい方を取得する
    angle_diffs = np.minimum(np.abs(diffs[indices, 0]), np.abs(diffs[indices, 1]))
    # angle_diffs = np.minimum(angles_mod[indices], np.pi/2 - angles_mod[indices])
    cost3 = params.lambda_angle * np.sum(angle_diffs)

    return cost1 + cost2 + cost3

# def optimize_loop(init_points: np.ndarray, image: np.ndarray, params: Param, method='L-BFGS-B', maxiter=500, ftol=1e-9):
def optimize_loop(init_points: np.ndarray, image: np.ndarray, params: Param, method='Powell', maxiter=100, ftol=1e-9):
    """
    2次元上のN点をループ状に最適化
    Args:
        reference_points (ndarray[N_ref, 2]): 固定参照点の座標集合
        init_points      (ndarray[N,     2]): 最適化開始時の点列(ループを想定)
        params          (Param): パラメータオブジェクト
        method           (str): optimize.minimize に渡す手法名
        maxiter          (int): 最大イテレーション数
        ftol             (float): 目的関数許容誤差"""
    N = init_points.shape[0]
    x0 = init_points.ravel()
    result = minimize(
        lambda xvec: cost_fn2(xvec, N, image, init_points, params), x0,
        method=method,
        options={'maxiter': maxiter, 'ftol': ftol}
    )
    opt_points = result.x.reshape((N, 2))
    return opt_points, result

def optimize_contour(init_points: np.ndarray, image: np.ndarray, params: Param) -> Tuple[np.ndarray, dict]:
    """点列をループ状に最適化するラッパー関数"""
    # opt_points, result = optimize_loop(reference_points, init_points, polygon_image, params=params)
    init_points = np.asarray(init_points)
    opt_points, result = optimize_loop(init_points, image, params=params, maxiter=0, ftol=1e-12)
    opt_points = [np.round(opt_points.reshape(-1, 1, 2)).astype(np.int32)]
    return opt_points, result
