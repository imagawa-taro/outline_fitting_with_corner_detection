import numpy as np
from scipy.ndimage import uniform_filter1d, maximum_filter1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
# from matplotlib.collections import LineCollection
# from matplotlib.colors import TwoSlopeNorm
# from scipy.ndimage import uniform_filter1d, maximum_filter1d
from typing import Tuple

plt.rcParams['font.family'] = 'MS Gothic'  # Windowsの場合
# または 'Yu Gothic', 'Meiryo', 'IPAexGothic' など

def curvature_from_closed_contour(
    x: np.ndarray,    y: np.ndarray,
    resample_points: int = None,
    sg_polyorder: int = 3,
    window_length_samples: int = None,
    window_length_arc: float = None,
    return_resampled_coords: bool = False
):
    """
    閉じた2D輪郭点列 (x[i], y[i]) から、弧長 s に対する曲率 κ(s) を推定
    Savitzky-Golayフィルタを用いて周期的境界条件で平滑微分しロバスト推定
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
        一様弧長サンプルの弧長座標(0 から全長 L の手前まで、等間隔)
    kappa : ndarray, shape (M,)
        s_u に対応する曲率 κ(s)（符号付き）。向き（反時計回り/時計回り）に依存して符号が変わる。
    x_u, y_u : ndarray, shape (M,), optional
        return_resampled_coords=True のとき返す。一様弧長に沿って再標本化した座標。

    Notes
    -----
    - 曲率の公式：κ = (x' y'' - y' x'') / ( (x'^2 + y'^2)^(3/2) )
    - 微分は Savitzky-Golay フィルタ (mode='wrap') による周期的平滑微分で計算。
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

    # Savitzky-Golay 窓幅の決定(奇数、polyorder より大きい、M以下）
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
    x: np.ndarray,
    y: np.ndarray,
    resample_points: int = None,
    sg_polyorder: int = 3,
    window_length_arc: float = None,
    window_length_samples: int = None,
    integ_window_arc: float = None,
    angle_range_deg: Tuple[float, float] = (45.0, 135.0),
    angle_measure: str = 'turning',   # 'turning'（接線の回転角Δθ）or 'internal'（内部角）
    angle_mode: str = 'both',         # 'both' | 'unsigned' | 'ccw' | 'cw'
    angle_margin_deg: float = 0.0,      # 範囲の緩和（±margin）
    min_kappa: float = None,
    min_kappa_factor: float = 1.5,
    peak_neighborhood_arc: float = None,
    return_all: bool = False
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
def get_contour_from_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
def is_line_segment(s_u: np.ndarray, kappa: np.ndarray, corner_indices: np.ndarray, threshold: float = 0.01) -> list[bool]:
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

