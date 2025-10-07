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
from postprocessing import postprocessing
from timing import section


def main() -> None:
    """
    メイン処理関数
    """
    data_folder = 'D:/20250929_layout_fitting3/data/'
    img_name = '001759.png' # 斜めの壁がある図面
    # img_name = '000325.png' # 曲がった壁がある図面
    org_name = 'image_org/'+img_name # 曲がった壁がある図面


    with section("preprocess"):
        wall_img = load_image_grayscale(f'{data_folder}{img_name}')
        org_img = load_image_grayscale(f'{data_folder}{org_name}')
        org_image = org_img.copy()
        org_image = cv2.GaussianBlur(org_image, (9, 9), 0)
        org_image = cv2.normalize(org_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    with section("contour_extraction"):
        contours = extract_room_contours(wall_img)
        initial_contours_img = visualize_contours(wall_img, contours)
        contours = drop_small_contours(contours, min_area=100.0)

    new_contours = []
    for i, cnt in enumerate(contours):
        with section("corner_detection"):
            x = cnt[:, 0, 0]
            y = cnt[:, 0, 1]
            s, kappa, x_u, y_u = curvature_from_closed_contour(
                x, y,
                resample_points=64,
                sg_polyorder=3,
                window_length_arc=int(6),
                return_resampled_coords=True
            )
            corner_idx3, s_corners3, xy_corners3, (s3, kappa3, x_u3, y_u3, dtheta3, ang_dict3) = detect_corners_by_curvature(
                x, y,
                resample_points=256,
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
        with section("optimization"):
            params = Param(
                lambda_data=1,
                lambda_smooth=0.000,
                lambda_angle=10
            )
            opt_points, result = optimize_contour(simplified_contours3, org_image, params)
            opt_points = opt_points[0].reshape(-1, 1, 2)
            new_contours.append(opt_points.astype(np.int32))

    with section("postprocess"):
        # postprocessing: new_contoursのエッジ統計処理
        aligned_contours = postprocessing(new_contours, org_image)

    with section("visualize"):
        contours_on_org_img = visualize_contours(org_img, aligned_contours)
        display_results(initial_contours_img, contours_on_org_img)



if __name__ == "__main__":
    main()