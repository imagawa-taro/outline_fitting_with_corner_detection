import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Tuple, Optional

from corner_detection import detect_corners_by_curvature
from visualize import visualize_contours, display_results, plot_single_contour
from image_utils import load_image_grayscale, extract_room_contours, get_silhouette, get_blur_image
from contours_utils import drop_small_contours, simplify_contour_with_corners
from optimize_contour import optimize_contour, Param
from postprocessing import postprocessing
from timing import section


def main(data_folder) -> None:
    """
    メイン処理関数
    """
    results_folder = '../results/'
    os.makedirs(results_folder, exist_ok=True)
    data_list = [int(f.split('.')[0]) for f in os.listdir(data_folder) if f.endswith('.png')]
    # data_list = [260, 325, 406, 426, 547, 581, 1407, 1423, 1521, 1759]  # 処理する画像の番号リスト
    # num = 6  # data_listのインデックスを指定★2, ★3,4, ★8
    # img_name = f'{data_list[num]:06d}.png' 
    # img_name = '001759.png' # 斜めの壁がある図面9
    # img_name = '000325.png' # 曲がった壁がある図面
    # org_name = 'image_org/'+img_name 


    # パラメータをdictで集約
    params = {
        'edge_blur_kernel_size': (15, 15),  # エッジの勾配生成用
        'erosion_kernel_size': (5, 5),  # silhouetteのシュリンク幅
        'min_area': 100.0,  # drop_small_contoursの最小面積
        'curvature_window_length_arc': 6.0,  # コーナー検出の曲率計算の平滑化窓幅（弧長単位）
        'corner_detection_resample_points': 256,  # コーナー検出の曲率計算の再標本化点数
        'integ_window_arc': 6.0,  # コーナー検出の曲率計算の積分窓幅（弧長単位）
        'corner_angle_range_deg': (45.0, 135.0),  # コーナー検出の角度範囲（度）
        'angle_margin_deg': 5.0,  # コーナー検出の角度マージン（度）
        'linearity_threshold': 0.7,  # 輪郭簡略化の直線性閾値
        'approx_epsilon_ratio': 0.01,  # 輪郭簡略化の近似精度（輪郭長に対する比率）
        'opt_lambda_pos': 1.000,  # 輪郭最適化の位置ペナルティ
        'opt_lambda_angle': 100,  # 輪郭最適化の角度ペナルティ
        'min_edge_length': 3.0,  # 後処理の対象にするエッジの最小長さ
        'angle_margin_deg': 15.0,  # 後処理のコーナー検出の角度マージン（度）
        'edge_cumulative_window_size': 2,  # 後処理のエッジ累積の窓幅
        'neighbor_distance': 3,  # 後処理の画像ヒストグラムの近傍距離
    }

    # data_listのループ処理
    for img_number in data_list:
        img_name = f'{img_number:06d}.png'
        with section("preprocess"):
            wall_img = load_image_grayscale(f'{data_folder}{img_name}')
            # org_img = load_image_grayscale(f'{data_folder}{org_name}')
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
                corner_idx3, s_corners3, xy_corners3, (s3, kappa3, x_u3, y_u3, dtheta3, ang_dict3) = detect_corners_by_curvature(
                    x, y,
                    resample_points=params['corner_detection_resample_points'],
                    sg_polyorder=3,
                    window_length_arc=params['curvature_window_length_arc'],
                    integ_window_arc=params['integ_window_arc'],
                    angle_measure='turning',
                    angle_range_deg=params['corner_angle_range_deg'],
                    angle_mode='both',
                    angle_margin_deg=params['angle_margin_deg'],
                    return_all=True
                )
                corners3 = np.array([x_u3, y_u3]).T.reshape(-1, 1, 2)
                simplified_contours3 = simplify_contour_with_corners(corners3, corner_indices=corner_idx3,
                                                linearity_threshold=params['linearity_threshold'],
                                                approx_epsilon_ratio=params['approx_epsilon_ratio'])
                # plot_single_contour(simplified_contours3, i)

            with section("optimization"):
                params_opt = Param(
                    lambda_data=1,
                    lambda_pos=params['opt_lambda_pos'],
                    lambda_angle=params['opt_lambda_angle']
                )
                opt_points, result = optimize_contour(simplified_contours3, ref_image, params_opt)
                opt_points = opt_points[0].reshape(-1, 1, 2)
                new_contours.append(opt_points.astype(np.int32))
                # plot_single_contour(opt_points, i)

        with section("postprocess"):
            aligned_contours = postprocessing(new_contours, ref_image, silhouette,
                                            params['min_edge_length'], params['angle_margin_deg'],
                                            params['edge_cumulative_window_size'], 
                                            params['neighbor_distance'])

        with section("visualize"):
            initial_contours_img = visualize_contours(wall_img, contours)
            new_contours_img = visualize_contours(wall_img, new_contours)
            contours_on_org_img = visualize_contours(wall_img, aligned_contours)
            # display_results(initial_contours_img, new_contours_img)
            # display_results(new_contours_img, contours_on_org_img)
            # display_results(initial_contours_img, contours_on_org_img)
            # plt.savefig(f'{results_folder}{img_name}', dpi=300)
            # 入出力画像を結合
            wall_img_for_stack = wall_img
            if wall_img_for_stack.ndim == 2:
                wall_img_for_stack = np.expand_dims(wall_img_for_stack, axis=-1)
                wall_img_for_stack = np.repeat(wall_img_for_stack, 3, axis=-1)
            combined_img1 = np.hstack((wall_img_for_stack, initial_contours_img))
            combined_img2 = np.hstack((initial_contours_img, contours_on_org_img))   
            combined_img = np.vstack((combined_img1, combined_img2))
            cv2.imwrite(f'{results_folder}{img_name}', combined_img)
            # plt.show()


if __name__ == "__main__":
    data_folder = '../data/'
    main(data_folder)
