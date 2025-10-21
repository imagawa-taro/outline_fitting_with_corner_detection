# 輪郭上のクロスエッジ処理モジュール
# remove_cross_edges関数の提供
import numpy as np
from typing import List, Tuple

import numpy as np
import cv2

def _orient(a, b, c):
    # 2Dベクトルの外積 z 成分（符号のみ使用）
    return np.cross(b - a, c - a)

def _on_segment(a, b, c, eps=1e-9):
    # c が線分 ab 上（端含む）にあるか
    return (min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps and
            min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps)

def segments_intersect(p1, p2, q1, q2, eps=1e-9, count_collinear_touch=True):
    """
    2線分 [p1,p2], [q1,q2] が交差するか。
    count_collinear_touch=True のとき、端点での接触や一直線上の端点接触も交差とみなす。
    """
    o1 = _orient(p1, p2, q1)
    o2 = _orient(p1, p2, q2)
    o3 = _orient(q1, q2, p1)
    o4 = _orient(q1, q2, p2)

    # 一般位置の交差（両側にある）
    if (o1 * o2 < -eps) and (o3 * o4 < -eps):
        return True

    if not count_collinear_touch:
        # 端点接触や共線の接触は交差としない
        return False

    # 以降は「接触・共線重なり」を交差とみなす場合
    # 共線ケース（o ~ 0）での区間重なり／端点接触
    if abs(o1) <= eps and _on_segment(p1, p2, q1, eps):
        return True
    if abs(o2) <= eps and _on_segment(p1, p2, q2, eps):
        return True
    if abs(o3) <= eps and _on_segment(q1, q2, p1, eps):
        return True
    if abs(o4) <= eps and _on_segment(q1, q2, p2, eps):
        return True

    return False

def _intersection_point(p1, p2, q1, q2, eps=1e-9):
    """
    交点の近似座標を返す（適用できるのは固有交差や端点接触）。
    共線重なりは None を返す。
    """
    r = p2 - p1
    s = q2 - q1
    rxs = np.cross(r, s)
    qmp = q1 - p1
    if abs(rxs) <= eps:
        # 平行（共線含む）→交点が一意に決まらないので None
        return None
    t = np.cross(qmp, s) / rxs
    u = np.cross(qmp, r) / rxs
    if -eps <= t <= 1 + eps and -eps <= u <= 1 + eps:
        return p1 + t * r
    return None

def find_self_intersections(contour, count_collinear_touch=True, eps=1e-9, return_points=True):
    """
    輪郭の自己交差を検出。
    - contour: shape (N,2) または (N,1,2) のnumpy配列
    - count_collinear_touch: 端点接触や共線接触も「交差」とみなすか
    - return_points: 交点座標を推定して返す（共線重なりは None）
    戻り値:
      has_cross, intersections
      has_cross: True/False
      intersections: list of dict {i, j, point}
        i, j は交差したエッジのインデックス（エッジ i は (i, i+1)）
        point は交点座標（推定）、共線重なりは None
    """
    # Nx1x2 → Nx2 へ
    cnt = np.asarray(contour)
    if cnt.ndim == 3 and cnt.shape[1] == 1 and cnt.shape[2] == 2:
        cnt = cnt[:, 0, :]
    cnt = cnt.astype(np.float64)
    n = len(cnt)
    if n < 3:
        return False, []

    # 閉曲線として最後と最初が一致している場合は重複点を除去
    if np.allclose(cnt[0], cnt[-1], atol=eps):
        cnt = cnt[:-1]
        n = len(cnt)
        if n < 3:
            return False, []

    intersections = []

    # エッジ i は (i, (i+1)%n)
    for i in range(n):
        p1 = cnt[i]
        p2 = cnt[(i + 1) % n]
        for j in range(i + 1, n):
            # 隣接エッジ（共有端点）を除外
            if j == i:
                continue
            if j == (i + 1) % n:
                continue
            if i == (j + 1) % n:
                continue

            q1 = cnt[j]
            q2 = cnt[(j + 1) % n]

            if segments_intersect(p1, p2, q1, q2, eps=eps, count_collinear_touch=count_collinear_touch):
                pt = _intersection_point(p1, p2, q1, q2, eps=eps) if return_points else None
                intersections.append({"i": i, "j": j, "point": pt})

    return (len(intersections) > 0), intersections

# デモ：自己交差ポリライン（ボウタイ型）
if __name__ == "__main__":
    new_points = 5
    size = 200
    pts = np.array([np.random.rand(new_points)*size, np.random.rand(new_points)*size], dtype=np.float64)
    # pts = np.array([[50, 50], [150, 150], [50, 150], [150, 50]], dtype=np.float64).T
        #最初の点を最後に追加して閉じる
    pts = np.hstack([pts, pts[:, :1]])
    pts = pts.T
    print("頂点座標:\n", pts)
    has_cross, inters = find_self_intersections(pts, count_collinear_touch=True, return_points=True)
    print("自己交差あり？:", has_cross)
    print("交差情報:", inters)

    # 可視化
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(pts[:, 0], pts[:, 1], '-o')
    for inter in inters:
        if inter["point"] is not None:
            plt.plot(inter["point"][0], inter["point"][1], 'rx', markersize=10)
    plt.title("Self-Intersecting Polyline with Intersections")
    plt.axis('equal')
    plt.savefig("self_intersection_demo.png")
    plt.close()