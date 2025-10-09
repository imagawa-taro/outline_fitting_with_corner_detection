import matplotlib.pyplot as plt
import cv2
import numpy as np
from typing import List, Tuple, Optional

from corner_detection import detect_corners_by_curvature
from visualize import visualize_contours, display_results, plot_single_contour
from image_utils import load_image_grayscale, extract_room_contours, get_silhouette, get_blur_image
from contours_utils import drop_small_contours, simplify_contour_with_corners
from curvature import curvature_from_closed_contour

from optimize_contour import optimize_contour, Param
from postprocessing import postprocessing
from timing import section


def main() -> None:
    """
    メイン処理関数
    """
    data_folder = 'D:/20250929_layout_fitting3/data/'
    data_list = [263, 325, 406, 426, 547, 581, 1407, 1423, 1521, 1759]  # 処理する画像の番号リスト
    num = 9  # data_listのインデックスを指定★2, ★3,4, ★8
    img_name = f'{data_list[num]:06d}.png' 
    # img_name = '001759.png' # 斜めの壁がある図面9
    # img_name = '000325.png' # 曲がった壁がある図面
    org_name = 'image_org/'+img_name 


    # パラメータをdictでまとめる例
    params = {
        'edge_blur_kernel_size': (9, 9),  # エッジの勾配生成用
        'erosion_kernel_size': (5, 5),  # silhouetteのシュリンク幅
        'min_area': 100.0,  # drop_small_contoursの最小面積
        'curvature_resample_points': 64,  # コーナー検出の曲率計算の再標本化点数
        'curvature_window_length_arc': 6.0,  # コーナー検出の曲率計算の平滑化窓幅（弧長単位）        # 他のパラメータもここに追加可能
        'corner_detection_resample_points': 256,  # コーナー検出の曲率計算の再標本化点数
    }

    with section("preprocess"):
        wall_img = load_image_grayscale(f'{data_folder}{img_name}')
        org_img = load_image_grayscale(f'{data_folder}{org_name}')
        ref_image = get_blur_image(wall_img, params['edge_blur_kernel_size'])

    with section("contour_extraction"):
        contours = extract_room_contours(wall_img, threshold=225)
        silhouette = get_silhouette(contours, wall_img.shape, params['erosion_kernel_size'])
        contours = drop_small_contours(contours, params['min_area'])

    new_contours = []
    for i, cnt in enumerate(contours):
        with section("corner_detection"):
            x = cnt[:, 0, 0]
            y = cnt[:, 0, 1]
            s, kappa, x_u, y_u = curvature_from_closed_contour(
                x, y,
                resample_points=params['curvature_resample_points'],
                sg_polyorder=3,
                window_length_arc=params['curvature_window_length_arc'],
                return_resampled_coords=True
            )
            corner_idx3, s_corners3, xy_corners3, (s3, kappa3, x_u3, y_u3, dtheta3, ang_dict3) = detect_corners_by_curvature(
                x, y,
                resample_points=params['corner_detection_resample_points'],
                sg_polyorder=3,
                window_length_arc=6,
                integ_window_arc=6,
                angle_measure='turning',
                angle_range_deg=(45.0, 135.0),
                angle_mode='both',
                angle_margin_deg=10.0,
                return_all=True
            )
            corners3 = np.array([x_u3, y_u3]).T.reshape(-1, 1, 2)
            simplified_contours3 = simplify_contour_with_corners(corners3, corner_indices=corner_idx3,
                                            linearity_threshold=0.7,
                                            approx_epsilon_ratio=0.01)
            # plot_single_contour(simplified_contours3, i)

        with section("optimization"):
            params_opt = Param(
                lambda_data=1,
                lambda_pos=1.000,
                lambda_angle=100
            )
            opt_points, result = optimize_contour(simplified_contours3, ref_image, params_opt)
            opt_points = opt_points[0].reshape(-1, 1, 2)
            new_contours.append(opt_points.astype(np.int32))
            # plot_single_contour(opt_points, i)

    with section("postprocess"):
        aligned_contours = postprocessing(new_contours, ref_image, silhouette)

    with section("visualize"):
        initial_contours_img = visualize_contours(wall_img, contours)
        new_contours_img = visualize_contours(wall_img, new_contours)
        contours_on_org_img = visualize_contours(wall_img, aligned_contours)
        display_results(initial_contours_img, new_contours_img)
        display_results(new_contours_img, contours_on_org_img)
        plt.show()


if __name__ == "__main__":
    main()