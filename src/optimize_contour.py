'''フィッティングの最適化ルーチン'''
from typing import Tuple
import numpy as np
from scipy.optimize import minimize

class Param:
    """最適化パラメータクラス"""
    def __init__(self, lambda_data=1.0, lambda_smooth=0.0, lambda_angle=100000):
        self.lambda_data = lambda_data
        self.lambda_smooth = lambda_smooth
        self.lambda_angle = lambda_angle


def cost_fn2(xvec: np.ndarray, N: int, image: np.ndarray, params: Param) -> float:
    """フィッティングのコスト関数"""
    # データ項
    pts = xvec.reshape((N, 2))
    # ptsの内分点を追加して補間点を増やす
    new_pts = []
    for i in range(N):
        p1 = pts[i]
        p2 = pts[(i+1)%N]
        new_pts.append(p1)
        new_pts.append((p1 + p2)/3)
    pts2 = np.array(new_pts)
    # image上のptsの位置の画素値の和をcost1とする
    h, w = image.shape[:2]
    coords = np.clip(pts2.astype(np.int32), [0, 0], [w-1, h-1])
    pixel_values = image[coords[:,1], coords[:,0]] / 255.0  # 白黒画像を想定
    # dists = 1.0 - pixel_values  # 白に近いほどコストが小さい
    cost1 = np.sum(pixel_values)*params.lambda_data

    # スムージング項
    p_prev = np.roll(pts,  1, axis=0)
    p_next = np.roll(pts, -1, axis=0)
    p_next2 = np.roll(pts, -2, axis=0)
    third_diff = -p_prev + 3*pts - 3*p_next + p_next2
    cost2 = params.lambda_smooth * np.sum(np.abs(third_diff))
    
    # 90度単位に近い場合は寄せる項
    diffs = p_next - pts
    angles = np.arctan2(diffs[:,1], diffs[:,0])
    angles_mod = np.mod(angles, np.pi/2)
    # angle_mod < thresholdのindexを見つける
    angle_threshold = np.pi/8
    indices = np.where((angles_mod < angle_threshold) | (angles_mod > np.pi/2 - angle_threshold))[0]
    angle_diffs = np.minimum(angles_mod[indices], np.pi/2 - angles_mod[indices])
    cost3 = params.lambda_angle * np.sum(np.abs(angle_diffs))

    return cost1 +cost2 + cost3

def optimize_loop(init_points: np.ndarray, image: np.ndarray, params: Param, method='L-BFGS-B', maxiter=500, ftol=1e-9):
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
    # tree = cKDTree(reference_points)
    x0 = init_points.ravel()
    result = minimize(
        lambda xvec: cost_fn2(xvec, N, image, params), x0,
        method=method,
        options={'maxiter': maxiter, 'ftol': ftol}
    )
    opt_points = result.x.reshape((N, 2))
    return opt_points, result

def optimize_contour(init_points: np.ndarray, image: np.ndarray, params: Param) -> Tuple[np.ndarray, dict]:
    """点列をループ状に最適化するラッパー関数"""
    # opt_points, result = optimize_loop(reference_points, init_points, polygon_image, params=params)
    init_points = np.asarray(init_points)
    opt_points, result = optimize_loop(init_points, image, params=params, maxiter=2000, ftol=1e-12)
    opt_points = [opt_points.reshape(-1, 1, 2).astype(np.int32)]
    return opt_points, result
