import matplotlib.pyplot as plt
import cv2
import numpy as np
from typing import List, Tuple, Optional

from corner_detection import detect_corners_by_curvature
from visualize import visualize_contours, display_results, plot_single_contour
from image_utils import load_image_grayscale, extract_room_contours
from contours_utils import drop_small_contours, simplify_contour_with_corners
from curvature import curvature_from_closed_contour
from optimize_contour import optimize_contour, Param


def main() -> None:
    """
    メイン処理関数
    """
    data_folder = 'D:/20250929_layout_fitting3/data/'
    # img_name = '001759.png' # 斜めの壁がある図面
    img_name = '000325.png' # 曲がった壁がある図面

    # 元画像を加工して作成した壁画像の読み込み
    wall_img = load_image_grayscale(f'{data_folder}{img_name}')
    est_image = wall_img.copy()
    # ガウスぼかし
    est_image = cv2.GaussianBlur(est_image, (11, 11), 0)
    # 正規化
    est_image = cv2.normalize(est_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    # 輪郭抽出
    contours = extract_room_contours(wall_img)
    initial_contours_img = visualize_contours(wall_img, contours)
    
    # 面積が小さい輪郭を削除
    contours = drop_small_contours(contours, min_area=100.0)
    
    # 全体輪郭の可視化
    # contours_on_org_img = visualize_contours(wall_img, contours)
    # display_results(initial_contours_img, contours_on_org_img)

    #ToDo:輪郭のスパイク・狭い領域除去


    # contoursに含まれる輪郭を個別表示
    new_contours = []
    for i, cnt in enumerate(contours):
        if True:  # 最初の1つだけ表示
            # plot_single_contour(cnt, i)

            # contour_length = cv2.arcLength(cnt, closed=True)
            x = cnt[:, 0, 0]
            y = cnt[:, 0, 1]
            # 曲率推定
            s, kappa, x_u, y_u = curvature_from_closed_contour(
                x, y,
                resample_points=64,
                sg_polyorder=3,
                window_length_arc=int(6),  # コーナー: 0.3〜0.6 × R_corner
                return_resampled_coords=True
            )

            corner_idx3, s_corners3, xy_corners3, (s3, kappa3, x_u3, y_u3, dtheta3, ang_dict3) = detect_corners_by_curvature(
                x, y,
                resample_points=256,
                sg_polyorder=3,
                window_length_arc=6,  # 周長の目安で設定
                integ_window_arc=6,
                angle_measure='turning',           # 回転角で範囲指定（例: 60°〜90°）
                angle_range_deg=(45.0, 135.0),
                angle_mode='both',
                angle_margin_deg=10.0,
                return_all=True
            )


            # print(f"検出されたコーナー数: {len(corner_idx3)}")
            # fig3, ax3 = plt.subplots(figsize=(8, 4), dpi=120)
            # ax3.plot(s3, kappa3, '-b', label='Curvature κ(s)')
            # ax3.axhline(0, color='k', linestyle='--', linewidth=0.8)
            # ax3.scatter(s_corners3, kappa3[corner_idx3], color='r', s=50, label='Detected corners')
            # ax3.set_xlabel('Arc length s (pixels)')
            # ax3.set_ylabel('Curvature κ (1/pixel)')
            # ax3.legend()
            # plt.tight_layout()

            # # 2D plot上にコーナー点を表示
            # plt.figure()
            # plt.plot(x, y, '-b', label='Contour')
            # plt.plot([x[-1], x[0]], [y[-1], y[0]], '-b')
            # plt.scatter(xy_corners3[:, 0], xy_corners3[:, 1], color='r', s=50, label='Detected corners')
            # plt.gca().set_aspect('equal', adjustable='box')
            # plt.title(f'Contour {i} with Detected Corners')
            # plt.xlabel('X') 
            # plt.ylabel('Y')
            # plt.gca().invert_yaxis()
            # plt.legend()
            # plt.show()  

            # corners3をx_u3, y_u3から生成
            corners3 = np.array([x_u3, y_u3]).T.reshape(-1, 1, 2)
            simplified_contours3 = simplify_contour_with_corners(corners3, corner_indices=corner_idx3,
                                            linearity_threshold=0.7,
                                            approx_epsilon_ratio=0.01)
            # plot_single_contour(simplified_contours3, i)

            # 最適化
            params = Param(
                lambda_data=0.1,
                lambda_smooth=0.000,
                lambda_angle=10
            )
            opt_points, result = optimize_contour(simplified_contours3, est_image, params)
            opt_points = opt_points[0].reshape(-1, 1, 2)
            # plot_single_contour(opt_points, i)


            new_contours.append(opt_points.astype(np.int32))

    contours_on_org_img = visualize_contours(wall_img, new_contours)
    display_results(initial_contours_img, contours_on_org_img)



if __name__ == "__main__":
    main()