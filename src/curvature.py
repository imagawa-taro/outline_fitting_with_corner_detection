import numpy as np
import cv2
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from typing import List

def curvature_from_closed_contour(
    x: np.ndarray,
    y: np.ndarray,
    resample_points: int = None,
    sg_polyorder: int = 3,
    window_length_samples: int = None,
    window_length_arc: float = None,
    return_resampled_coords: bool = False
) -> tuple:
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


# 画像を引数として輪郭(closed contour)抽出する関数
def get_contour_from_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


# 使い方の簡単な例（円周で検証）
if __name__ == "__main__":
    # 画像から輪郭を抽出して曲率推定
    image_path = 'layout.png'
    image = cv2.imread(image_path)

    # 画像から輪郭を抽出
    x, y = get_contour_from_image(image)
    arc_lengths = np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0])) # 各セグメント長
    contour_length=arc_lengths.sum()  # 周長の一定割合の窓
    # print(f"抽出輪郭点数: {len(x)}, 周長: {contour_length:.1f} ピクセル")

    # 曲率推定（可視化で使うため再標本化座標も取得）
    s, kappa, x_u, y_u = curvature_from_closed_contour(
        x, y,
        resample_points=256,
        sg_polyorder=3,
        window_length_arc=int(contour_length*0.05),  # 周長の一定割合の窓
        return_resampled_coords=True
    )

    # 可視化：入力点のプロット、直線で結び、曲率に応じて色付け
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)

    # 入力点列を散布図としてプロット
    ax.scatter(x, y, s=12, c='k', alpha=0.6, label='Input points')

    # 再標本化した点列を直線で結ぶためのセグメント作成（閉曲線なので終点->始点も結ぶ）
    pts = np.column_stack([x_u, y_u])
    segs = np.stack([pts, np.roll(pts, -1, axis=0)], axis=1)  # shape (M, 2, 2)

    # 曲率の色マッピング（0 を中心にした発散カラーマップ）
    vmax = np.nanpercentile(np.abs(kappa), 99.0)  # 外れ値の影響を緩和
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    lc = LineCollection(segs, cmap='jet', norm=norm, linewidth=2.0)
    lc.set_array(kappa)  # 各セグメントに対応する曲率（kappa[i]）を割り当て
    ax.add_collection(lc)


    # カラーバー
    cbar = plt.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Curvature κ (1/pixel)')

    # 体裁
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Closed contour with curvature-colored segments')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.tight_layout()


    # 可視化：曲率 κ(s) のプロット
    fig2, ax2 = plt.subplots(figsize=(8, 4), dpi=120)
    ax2.plot(s, kappa, '-b', label='Curvature κ(s)')
    ax2.axhline(0, color='k', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('Arc length s (pixels)')
    ax2.set_ylabel('Curvature κ (1/pixel)')
    ax2.set_title('Curvature along the contour')
    ax2.legend()
    plt.tight_layout()
    plt.show()