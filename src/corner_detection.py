import numpy as np
from scipy.ndimage import uniform_filter1d, maximum_filter1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import uniform_filter1d, maximum_filter1d
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm

plt.rcParams['font.family'] = 'MS Gothic'  # Windowsの場合
# または 'Yu Gothic', 'Meiryo', 'IPAexGothic' など

def curvature_from_closed_contour(
    x,
    y,
    resample_points=None,
    sg_polyorder=3,
    window_length_samples=None,
    window_length_arc=None,
    return_resampled_coords=False
):
    """
    閉じた2D輪郭点列 (x[i], y[i]) から、弧長 s に対する曲率 κ(s) を推定する。
    Savitzky–Golay フィルタを用いて周期的境界条件で平滑微分し、ノイズにロバストな推定を行う。

    Parameters
    ----------
    x, y : array-like, shape (N,)
        閉じた輪郭の点列。順序は輪郭に沿って並んでいること。
        入力が厳密に閉じていなくても（x[0],y[0]) と (x[-1],y[-1]) が異なっていても処理内で扱う。
    resample_points : int or None
        弧長に沿って一様間隔に再標本化するサンプル数 M。
        None の場合は元の点数に合わせる（M = N）。大きくし過ぎると過平滑・計算量増となる。
    sg_polyorder : int
        Savitzky–Golayフィルタの多項式次数（通常は3～4が推奨）。
    window_length_samples : int or None
        Savitzky–Golayフィルタの窓幅（サンプル数）。奇数である必要あり。
        None の場合はデフォルトで全体の約5%を奇数化して使用。
    window_length_arc : float or None
        Savitzky–Golayフィルタの窓幅（弧長単位）。指定された場合は ds で換算してサンプル窓幅に変換。
        window_length_samples と両方指定された場合は window_length_arc を優先。
    return_resampled_coords : bool
        True の場合、平滑化に用いた一様弧長再標本化後の座標 x_u, y_u も返す。

    Returns
    -------
    s_u : ndarray, shape (M,)
        一様弧長サンプルの弧長座標（0 から全長 L の手前まで、等間隔）。
    kappa : ndarray, shape (M,)
        s_u に対応する曲率 κ(s)（符号付き）。向き（反時計回り/時計回り）に依存して符号が変わる。
    x_u, y_u : ndarray, shape (M,), optional
        return_resampled_coords=True のとき返す。一様弧長に沿って再標本化した座標。

    Notes
    -----
    - 曲率の公式：κ = (x' y'' - y' x'') / ( (x'^2 + y'^2)^(3/2) )
    - 微分は Savitzky–Golay フィルタ (mode='wrap') による周期的平滑微分で計算。
    - 入力点間隔が不均一だと微分が不正確になりやすいので、必ず弧長一様へ再標本化してから微分します。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x, y は同じ長さの1次元配列である必要があります。")
    N = x.size
    if N < 5:
        raise ValueError("点数が少なすぎます（>=5 推奨）。")

    # 距離と弧長の計算（閉曲線として最後から最初のセグメントも考慮）
    dx = np.diff(x, append=x[0])
    dy = np.diff(y, append=y[0])
    seg_lengths = np.hypot(dx, dy)  # 各セグメント長（N個、最後は N-1 -> 0 の閉じセグメント）
    L = np.sum(seg_lengths)        # 全周長
    # 各点に対応する弧長（最初の点は s=0、最後の点は s=sum(seg_lengths[:-1])）
    s_points = np.concatenate(([0.0], np.cumsum(seg_lengths[:-1])))

    # 弧長0..L上での周期的補間のため、終端 (s=L) に始点座標を追加
    s_ext = np.concatenate([s_points, [L]])
    x_ext = np.concatenate([x, [x[0]]])
    y_ext = np.concatenate([y, [y[0]]])

    # 一様弧長へ再標本化
    M = int(resample_points) if (resample_points is not None) else N
    if M < 5:
        M = 5  # 最低限のサンプル数
    s_u = np.linspace(0.0, L, M, endpoint=False)  # 0..L を等間隔（Lは重複回避で含めない）
    x_u = np.interp(s_u, s_ext, x_ext)
    y_u = np.interp(s_u, s_ext, y_ext)

    # Savitzky–Golay 窓幅の決定（奇数、polyorder より大きい、M以下）
    ds = s_u[1] - s_u[0]
    if window_length_arc is not None:
        w = int(round(window_length_arc / ds))
    elif window_length_samples is not None:
        w = int(window_length_samples)
    else:
        w = int(round(M * 0.05))  # 全体の約5%をデフォルトに
    # 奇数化と境界調整
    if w < sg_polyorder + 2:
        w = sg_polyorder + 2  # 最低限 polyorder+2 程度
    if w % 2 == 0:
        w += 1
    if w > M:
        w = M if (M % 2 == 1) else (M - 1)
        if w <= sg_polyorder:
            w = sg_polyorder + 2
            if w % 2 == 0:
                w += 1
            if w > M:
                raise ValueError("サンプル数が少なすぎて SG フィルタの条件を満たせません。M や polyorder を調整してください。")

    # 一様弧長上での平滑微分（周期的 wrap）
    x1 = savgol_filter(x_u, window_length=w, polyorder=sg_polyorder, deriv=1, delta=ds, mode='wrap')
    y1 = savgol_filter(y_u, window_length=w, polyorder=sg_polyorder, deriv=1, delta=ds, mode='wrap')
    x2 = savgol_filter(x_u, window_length=w, polyorder=sg_polyorder, deriv=2, delta=ds, mode='wrap')
    y2 = savgol_filter(y_u, window_length=w, polyorder=sg_polyorder, deriv=2, delta=ds, mode='wrap')

    # 曲率計算（ゼロ除算を回避するための微小項）
    speed_sq = x1**2 + y1**2
    eps = 1e-12
    kappa = (x1 * y2 - y1 * x2) / (np.power(speed_sq + eps, 1.5))

    if return_resampled_coords:
        return s_u, kappa, x_u, y_u
    else:
        return s_u, kappa

def detect_corners_by_curvature(
    x,
    y,
    resample_points=None,
    sg_polyorder=3,
    window_length_arc=None,
    window_length_samples=None,
    integ_window_arc=None,
    angle_range_deg=(45.0, 135.0),
    angle_measure='turning',   # 'turning'（接線の回転角Δθ）or 'internal'（内部角）
    angle_mode='both',         # 'both' | 'unsigned' | 'ccw' | 'cw'
    angle_margin_deg=0.0,      # 範囲の緩和（±margin）
    min_kappa=None,
    min_kappa_factor=1.5,
    peak_neighborhood_arc=None,
    return_all=False
):
    """
    曲率 κ(s) の移動積分 Δθ ≈ ∫ κ ds を用いて、指定角度範囲のコーナーを検出する。

    Parameters
    ----------
    x, y : array-like
        閉じた輪郭点列。
    resample_points, sg_polyorder, window_length_arc, window_length_samples :
        curvature_from_closed_contour に渡す平滑微分の設定。
    integ_window_arc : float or None
        κ を積分する移動ウィンドウ長（弧長）。None の場合は window_length_arc（あれば）か、なければ周長の約5%。
    angle_range_deg : tuple(float, float)
        角度範囲（度）。angle_measure='turning' なら |Δθ| の範囲、
        'internal' なら内部角 α の範囲（α = 180° - |Δθ|）。
    angle_measure : str
        'turning'（接線の回転角Δθの大きさを使う）または 'internal'（内部角で範囲指定）。
    angle_mode : str
        'both'     : +Δθ と −Δθ の両方を許容（符号つきで両向き検出）
        'unsigned' : 符号を無視（|Δθ| のみで判定）
        'ccw'      : +Δθ のみ（反時計回り）
        'cw'       : −Δθ のみ（時計回り）
    angle_margin_deg : float
        範囲判定に追加する緩和幅（度）。範囲を [min−margin, max+margin] に広げます。
    min_kappa : float or None
        ピーク曲率の下限。None の場合は min_kappa_factor * (theta_min_turn / integ_window) を採用。
        theta_min_turn は許容範囲の最小「回転角」（degree→rad）です。
    min_kappa_factor : float
        上記の係数（1.5〜2程度が目安）。
    peak_neighborhood_arc : float or None
        局所ピーク判定の近傍幅（弧長）。None の場合は integ_window_arc/3 程度。
    return_all : bool
        True の場合、中間量（s_u, kappa, x_u, y_u, delta_theta, angle_deg）も返す。

    Returns
    -------
    corner_indices : ndarray
        再標本化系列上のコーナーインデックス。
    s_u_corners : ndarray
        コーナーの弧長位置。
    xy_corners : ndarray, shape (K, 2)
        コーナーの座標。
    (s_u, kappa, x_u, y_u, delta_theta, angle_deg_dict) : optional
        return_all=True のとき追加で返す。angle_deg_dict は
        {'turning': |Δθ|の度, 'internal': 内部角の度} を含む辞書。
    """
    # 曲率推定（再標本化＆平滑微分）
    s_u, kappa, x_u, y_u = curvature_from_closed_contour(
        x, y,
        resample_points=resample_points,
        sg_polyorder=sg_polyorder,
        window_length_arc=window_length_arc,
        window_length_samples=window_length_samples,
        return_resampled_coords=True
    )
    M = len(s_u)
    ds = s_u[1] - s_u[0]
    L = ds * M

    # 移動積分ウィンドウの決定
    if integ_window_arc is None:
        integ_window_arc = window_length_arc if (window_length_arc is not None) else (0.05 * L)
    w_int = int(round(integ_window_arc / ds))
    if w_int < 3:
        w_int = 3
    if w_int % 2 == 0:
        w_int += 1

    # Δθ の近似（移動平均 * ウィンドウ長）
    delta_theta = uniform_filter1d(kappa, size=w_int, mode='wrap') * (w_int * ds)

    # 角度の大きさ（deg）を計算
    turning_deg = np.rad2deg(np.abs(delta_theta))
    internal_deg = 180.0 - turning_deg  # 内部角 α = π - |Δθ|（度に換算）
    angle_deg = turning_deg if (angle_measure == 'turning') else internal_deg

    # 範囲判定
    amin, amax = angle_range_deg
    margin = angle_margin_deg
    mask_range = (angle_deg >= (amin - margin)) & (angle_deg <= (amax + margin))

    # 符号の扱い
    if angle_mode == 'both':
        mask_sign = np.ones(M, dtype=bool)
    elif angle_mode == 'unsigned':
        mask_sign = np.ones(M, dtype=bool)
    elif angle_mode == 'ccw':
        mask_sign = (delta_theta > 0)
    elif angle_mode == 'cw':
        mask_sign = (delta_theta < 0)
    else:
        raise ValueError("angle_mode は 'both' | 'unsigned' | 'ccw' | 'cw' を指定してください。")

    # 局所ピークの検出準備（|κ|）
    if peak_neighborhood_arc is None:
        peak_neighborhood_arc = integ_window_arc / 3.0
    w_peak = int(round(peak_neighborhood_arc / ds))
    if w_peak < 3:
        w_peak = 3
    if w_peak % 2 == 0:
        w_peak += 1
    abs_kappa = np.abs(kappa)
    local_max = maximum_filter1d(abs_kappa, size=w_peak, mode='wrap')
    mask_peak = abs_kappa >= local_max - 1e-12

    # 鋭さの規定：|κ| の下限（自動）
    if min_kappa is None:
        # 許容範囲に対応する最小回転角（rad）からしきい値を作る
        if angle_measure == 'turning':
            theta_min_turn = np.deg2rad(amin)
        else:
            # 内部角の最大が最小の回転角に対応（Δθ = π - α）
            theta_min_turn = np.deg2rad(180.0 - amax)
        min_kappa = min_kappa_factor * (theta_min_turn / (w_int * ds))
    mask_sharp = abs_kappa >= min_kappa

    # 候補インデックス
    candidates = np.where(mask_range & mask_sign & mask_peak & mask_sharp)[0]

    # 非最大抑制（周期）：|κ|が大きい順に採用し、近傍は抑制
    if candidates.size > 0:
        suppress_radius = max(w_int, w_peak) // 2
        order = np.argsort(-abs_kappa[candidates])
        selected = []
        for idx in candidates[order]:
            if not selected:
                selected.append(idx)
                continue
            dists = np.array([min(abs(idx - j), M - abs(idx - j)) for j in selected])
            if np.all(dists > suppress_radius):
                selected.append(idx)
        corner_indices = np.array(sorted(selected), dtype=int)
    else:
        corner_indices = np.array([], dtype=int)

    s_u_corners = s_u[corner_indices]
    xy_corners = np.column_stack([x_u[corner_indices], y_u[corner_indices]])

    angle_deg_dict = {'turning': turning_deg, 'internal': internal_deg}

    if return_all:
        return corner_indices, s_u_corners, xy_corners, (s_u, kappa, x_u, y_u, delta_theta, angle_deg_dict)
    else:
        return corner_indices, s_u_corners, xy_corners
    




# 画像を引数として輪郭(closed contour)抽出する関数
def get_contour_from_image(image):
    import cv2
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("画像から輪郭が検出できません。")
    # 面積最大の輪郭を選択
    main_contour = max(contours, key=cv2.contourArea)
    main_contour = main_contour.squeeze()  # (N,1,2) -> (N,2)
    if main_contour.ndim != 2 or main_contour.shape[1] != 2:
        raise ValueError("輪郭形状が不正")
    return main_contour[:,0], main_contour[:,1]  # x, y


# 隣り合うコーナー間の点列が直線か否かを判定する関数を定義
# 巡回条件を考慮して、各コーナー間の点列を抽出
def is_line_segment(s_u, kappa, corner_indices, threshold=0.01):
    M = len(s_u)
    K = len(corner_indices)
    line_flags = []
    for i in range(K):
        start_idx = corner_indices[i]
        end_idx = corner_indices[(i + 1) % K]
        if start_idx < end_idx:
            segment_kappa = kappa[start_idx:end_idx+1]
        else:
            segment_kappa = np.concatenate([kappa[start_idx:], kappa[:end_idx+1]])
        max_abs_kappa = np.max(np.abs(segment_kappa))
        line_flags.append(max_abs_kappa < threshold) # 閾値以下なら直線と判定
    return line_flags


if __name__ == "__main__":
    '''運用のポイント
angle_measure='turning' は接線の回転角（外角）による範囲指定です。内部角 α を使いたい場合は angle_measure='internal' を選んでください（α = 180° − |Δθ|）。
integ_window_arc（角度積分の窓）と window_length_arc（平滑微分の窓）は、コーナーの丸み半径よりやや大きく設定すると安定します。小さすぎるとノイズに敏感、大きすぎると角度が丸まり範囲から外れやすくなります。
min_kappa_factor を上げると「鈍いコーナー」を除外しやすくなります。誤検出がある場合は peak_neighborhood_arc を広げる、angle_margin_deg を少し増やす、窓長を見直すと改善します。
符号で凸角/凹角を分けたい場合は angle_mode を使うか、返り値の delta_theta の符号で分類してください。'''


    rng = np.random.default_rng(0)

    # 例1: 矩形（内部角=90°）を 45°〜135°の範囲で検出
    W, H = 120.0, 80.0
    pts_per_side = 80
    top = np.column_stack([np.linspace(-W/2, W/2, pts_per_side, endpoint=False), np.full(pts_per_side, H/2)])
    right = np.column_stack([np.full(pts_per_side, W/2), np.linspace(H/2, -H/2, pts_per_side, endpoint=False)])
    bottom = np.column_stack([np.linspace(W/2, -W/2, pts_per_side, endpoint=False), np.full(pts_per_side, -H/2)])
    left = np.column_stack([np.full(pts_per_side, -W/2), np.linspace(-H/2, H/2, pts_per_side, endpoint=False)])
    xy = np.vstack([top, right, bottom, left])
    xy_noisy = xy + rng.normal(scale=0.5, size=xy.shape)
    x, y = xy_noisy[:, 0], xy_noisy[:, 1]

    # 曲率推定（可視化用）
    s_u, kappa, x_u, y_u = curvature_from_closed_contour(
        x, y, resample_points=400, sg_polyorder=3, window_length_arc=0.03 * (2*(W+H)), return_resampled_coords=True
    )

    # コーナー検出：内部角で 45°〜135°
    corner_idx, s_corners, xy_corners, (s_all, kappa_all, x_all, y_all, dtheta, ang_dict) = detect_corners_by_curvature(
        x, y,
        resample_points=400,
        sg_polyorder=3,
        window_length_arc=0.03 * (2*(W+H)),
        integ_window_arc=0.02 * (2*(W+H)),
        angle_measure='internal',         # 内部角で範囲指定
        angle_range_deg=(45.0, 135.0),
        angle_mode='both',
        angle_margin_deg=5.0,             # 少し緩める
        return_all=True
    )
    print(f"矩形の検出コーナー数: {len(corner_idx)}")


    line_segment_flags = is_line_segment(s_u, kappa, corner_idx, threshold=1)
    for i, is_line in enumerate(line_segment_flags):
        print(f"コーナー {i} から次のコーナーまでの区間は {'直線' if is_line else '直線でない'} と判定。")

    # 可視化
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.scatter(x, y, s=10, c='k', alpha=0.5, label='入力点')
    pts = np.column_stack([x_u, y_u])
    segs = np.stack([pts, np.roll(pts, -1, axis=0)], axis=1)
    vmax = np.nanpercentile(np.abs(kappa), 99.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    lc = LineCollection(segs, cmap='coolwarm', norm=norm, linewidth=2.0)
    lc.set_array(kappa)
    ax.add_collection(lc)
    cbar = plt.colorbar(lc, ax=ax, fraction=0.046, pad=0.04); cbar.set_label('Curvature κ')

    ax.scatter(xy_corners[:, 0], xy_corners[:, 1], s=70, c='lime', edgecolor='r', marker='*', label='検出コーナー')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('内部角45°〜135°のコーナー検出（矩形）')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # 例2: 正五角形（内部角≈108°、回転角≈72°）
    R = 50.0
    N = 200
    t = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5  # 回転してお好みの向きに
    pent = np.column_stack([R*np.cos(t), R*np.sin(t)])
    # 各辺を線形補間して点列化（CCW）
    def interpolate_edges(poly, pts_per_edge=60):
        segs = []
        for i in range(len(poly)):
            p0 = poly[i]
            p1 = poly[(i+1) % len(poly)]
            u = np.linspace(0, 1, pts_per_edge, endpoint=False)
            segs.append(np.outer(1-u, p0) + np.outer(u, p1))
        return np.vstack(segs)
    xy2 = interpolate_edges(pent, pts_per_edge=60)
    xy2_noisy = xy2 + rng.normal(scale=0.3, size=xy2.shape)
    x2, y2 = xy2_noisy[:, 0], xy2_noisy[:, 1]

    corner_idx2, s_corners2, xy_corners2, (s2, kappa2, x_u2, y_u2, dtheta2, ang_dict2) = detect_corners_by_curvature(
        x2, y2,
        resample_points=500,
        sg_polyorder=3,
        window_length_arc=0.06 * (5 * 2 * R * np.sin(np.pi/5)),  # 周長の目安で設定
        integ_window_arc=0.04 * (5 * 2 * R * np.sin(np.pi/5)),
        angle_measure='turning',           # 回転角で範囲指定（例: 60°〜90°）
        angle_range_deg=(60.0, 90.0),
        angle_mode='ccw',
        angle_margin_deg=5.0,
        return_all=True
    )
    print(f"正五角形の検出コーナー数: {len(corner_idx2)}")
    line_segment_flags = is_line_segment(s2, kappa2, corner_idx2, threshold=1)
    for i, is_line in enumerate(line_segment_flags):
        print(f"コーナー {i} から次のコーナーまでの区間は {'直線' if is_line else '直線でない'} と判定。")

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.scatter(x2, y2, s=10, c='k', alpha=0.5, label='入力点')
    pts2 = np.column_stack([x_u2, y_u2])
    segs2 = np.stack([pts2, np.roll(pts2, -1, axis=0)], axis=1)
    vmax2 = np.nanpercentile(np.abs(kappa2), 99.0)
    norm2 = TwoSlopeNorm(vmin=-vmax2, vcenter=0.0, vmax=vmax2)
    lc2 = LineCollection(segs2, cmap='coolwarm', norm=norm2, linewidth=2.0)
    lc2.set_array(kappa2)
    ax.add_collection(lc2)
    cbar2 = plt.colorbar(lc2, ax=ax, fraction=0.046, pad=0.04); cbar2.set_label('Curvature κ')

    ax.scatter(xy_corners2[:, 0], xy_corners2[:, 1], s=70, c='orange', edgecolor='r', marker='*', label='検出コーナー')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('回転角60°〜90°のコーナー検出（正五角形）')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # 例3: 画像から輪郭を抽出してコーナー検出
    import cv2
    image_path = 'layout1.png'
    image = cv2.imread(image_path)

    # 画像から輪郭を抽出
    x3, y3 = get_contour_from_image(image)
    arc_lengths = np.hypot(np.diff(x3, append=x3[0]), np.diff(y3, append=y3[0]))
    contour_length=arc_lengths.sum()  # 周長の一定割合の窓
    # print(f"抽出輪郭点数: {len(x3)}, 周長: {contour_length:.1f} ピクセル")

    # 曲率推定（可視化で使うため再標本化座標も取得）
    s3, kappa3, x_u3, y_u3 = curvature_from_closed_contour(
        x3, y3,
        resample_points=256,
        sg_polyorder=3,
        window_length_arc=int(contour_length*0.05),  # 周長の一定割合の窓
        return_resampled_coords=True
    )

    # コーナー検出
    corner_idx3, s_corners3, xy_corners3, (s3, kappa3, x_u3, y_u3, dtheta3, ang_dict3) = detect_corners_by_curvature(
        x3, y3,
        resample_points=256,
        sg_polyorder=3,
        window_length_arc=0.04 * contour_length,  # 周長の目安で設定
        integ_window_arc=0.04 * contour_length,
        angle_measure='turning',           # 回転角で範囲指定（例: 60°〜90°）
        angle_range_deg=(45.0, 135.0),
        angle_mode='both',
        angle_margin_deg=10.0,
        return_all=True
    )
    print(f"検出コーナー数: {len(corner_idx3)}")
    line_segment_flags = is_line_segment(s3, kappa3, corner_idx3, threshold=1)
    for i, is_line in enumerate(line_segment_flags):
        print(f"コーナー {i} から次のコーナーまでの区間は {'直線' if is_line else '直線でない'} と判定。")

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.scatter(x3, y3, s=10, c='k', alpha=0.5, label='入力点')
    pts3 = np.column_stack([x_u3, y_u3])
    segs3 = np.stack([pts3, np.roll(pts3, -1, axis=0)], axis=1)
    vmax3 = np.nanpercentile(np.abs(kappa3), 99.0)
    norm3 = TwoSlopeNorm(vmin=-vmax3, vcenter=0.0, vmax=vmax3)
    lc3 = LineCollection(segs3, cmap='coolwarm', norm=norm3, linewidth=2.0)
    lc3.set_array(kappa3)
    ax.add_collection(lc3)
    cbar3 = plt.colorbar(lc3, ax=ax, fraction=0.046, pad=0.04); cbar3.set_label('Curvature κ')

    ax.scatter(xy_corners3[:, 0], xy_corners3[:, 1], s=70, c='orange', edgecolor='r', marker='*', label='検出コーナー')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('コーナー検出')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
