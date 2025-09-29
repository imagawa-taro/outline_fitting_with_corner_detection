import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from scipy.spatial import cKDTree

def load_image_grayscale(image_path: str) -> np.ndarray:
    """
    画像をグレースケールで読み込む
    
    Args:
        image_path (str): 画像ファイルのパス
    
    Returns:
        ndarray: グレースケール画像
    
    Raises:
        FileNotFoundError: 画像ファイルが見つからない場合
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"画像ファイルが見つかりません。画像ファイル名を確認してください: {image_path}")
    return img


def extract_room_contours(image: np.ndarray, threshold: int = 225) -> List[np.ndarray]:
    """
    画像から部屋の輪郭を抽出する
    
    Args:
        image (ndarray): グレースケール画像
        threshold (int): 2値化の閾値
    
    Returns:
        list: 部屋の輪郭リスト
    """
    # 2値化（壁を白、それ以外を黒にする）
    _, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)

    # 輪郭抽出（第2引数をRETR_CCOMPとして建物外周と部屋外周の2層で抽出）
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # 階層の末端輪郭のみを残す（部屋外周のみを残す）
    if hierarchy is not None:
        hierarchy = hierarchy[0]  # shape: (num_contours, 4)
        leaf_contours = [cnt for cnt, h in zip(contours, hierarchy) if h[2] == -1]
        return leaf_contours
    else:
        return contours



def visualize_contours(image: np.ndarray, contours: List[np.ndarray]) -> np.ndarray:
    """
    輪郭を可視化する
    
    Args:
        image (ndarray): 元のグレースケ画像
        contours (list): 輪郭リスト
    
    Returns:
        ndarray: 描画結果画像
    """
    # 描画用画像
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # カラーパレット生成
    cmap = plt.get_cmap('Set1')
    PALETTE = [(np.array(cmap(i)[:3]) * 255).astype(int)[::-1] for i in range(cmap.N)]
    # 輪郭の線描画と塗りつぶし
    for i, cnt in enumerate(contours):
        if len(cnt) >= 3:
            line_color = PALETTE[i % len(PALETTE)]
            fill_color = (line_color * 0.5 + 255 * 0.5).astype(int)
            cv2.fillPoly(output, [cnt], fill_color.tolist())
            cv2.drawContours(output, [cnt], -1, line_color.tolist(), 1)
            # 輪郭の左上にインデックスを表示
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])
                cv2.putText(output, str(i), (cX-10, cY+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
    
    return output

def display_results(original_image: np.ndarray, result_image: np.ndarray) -> None:
    """
    結果を並べて表示する
    
    Args:
        original_image (ndarray): 元画像
        result_image (ndarray): 処理結果画像
    """
    # 結果表示（元画像と結果を並べて表示）
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 元画像
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Before')
    axes[0].axis('off')

    # 処理結果
    axes[1].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title('After')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()



def drop_small_contours(contours: List[np.ndarray], min_area: float = 100.0) -> List[np.ndarray]:
    """
    面積が小さい輪郭を削除する
    
    Args:
        contours (list): 元の輪郭リスト
        min_area (float): 保持する輪郭の最小面積（ピクセル）
    
    Returns:
        list: フィルタリングされた輪郭リスト
    """
    filtered_contours = []
    
    for contour in contours:
        # 輪郭の面積を計算
        area = cv2.contourArea(contour)
        
        # 最小面積以上の輪郭のみを保持
        if area >= min_area:
            filtered_contours.append(contour)
    
    return filtered_contours

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