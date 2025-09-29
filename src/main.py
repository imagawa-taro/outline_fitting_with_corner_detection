from typing import List, Tuple, Optional

from visualize import visualize_contours, display_results
from image_utils import load_image_grayscale, extract_room_contours
from contours_utils import drop_small_contours




def main() -> None:
    """
    メイン処理関数
    """
    data_folder = 'D:/20250929_layout_fitting3/data/'
    img_name = '001759.png' # 斜めの壁がある図面
    # img_name = '000325.png' # 曲がった壁がある図面

    # 元画像を加工して作成した壁画像の読み込み
    wall_img = load_image_grayscale(f'{data_folder}{img_name}')

    # 輪郭抽出
    contours = extract_room_contours(wall_img)
    initial_contours_img = visualize_contours(wall_img, contours)
    
    # 面積が小さい輪郭を削除
    contours = drop_small_contours(contours, min_area=100.0)
    
    # 輪郭の可視化
    contours_on_org_img = visualize_contours(wall_img, contours)
    display_results(initial_contours_img, contours_on_org_img)


if __name__ == "__main__":
    main()