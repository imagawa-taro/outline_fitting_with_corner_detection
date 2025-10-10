import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from scipy.spatial import cKDTree
from timing import section

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

def snap_contours_to_points(contours: List[np.ndarray], points: np.ndarray, max_distance: Optional[float] = None, keep_unsnapped: bool = True) -> List[np.ndarray]:
    """
    輪郭の頂点を最も近い指定された点にスナップする
    
    Args:
        contours (list): 元の輪郭リスト
        points (ndarray): スナップ先の点群 (N, 2) の形状
        max_distance (float, optional): スナップする最大距離（ピクセル）。Noneの場合は制限なし
        keep_unsnapped (bool): スナップできない点を保持するか（True: 保持, False: 削除）
    
    Returns:
        list: スナップされた輪郭リスト
    """
    if points.size == 0:
        return contours
    
    # KDTreeで点のインデックスを構築
    kdtree = cKDTree(points)
    
    # 各輪郭の各頂点をスナップ
    snapped_contours = []
    for cnt in contours:
        snapped_cnt = []
        for point in cnt:
            x, y = point[0]
            # 最も近い点を検索
            dist, closest_idx = kdtree.query([x, y])
            # 最大距離チェック
            if max_distance is None or dist <= max_distance:
                # 点にスナップ
                snapped_point = points[closest_idx]
                snapped_cnt.append([[snapped_point[0], snapped_point[1]]])
            elif keep_unsnapped:
                snapped_cnt.append([[x, y]])    # スナップしない場合は元の点を保持
            # keep_unsnapped=Falseの場合は何もしない（点を削除）
        
        # 空でない輪郭のみを追加
        if len(snapped_cnt) > 0:
            snapped_contours.append(np.array(snapped_cnt, dtype=np.int32))
    
    return snapped_contours

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
            # 頂点プロット
            for point in cnt:
                cv2.circle(output, tuple(point[0]), 2, (255, 0, 0), -1)

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

def detect_lines_and_intersections(contours: List[np.ndarray], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    輪郭からハフ変換で直線を検出し、直線同士の交点を計算する

    Args:
        contours (list): 輪郭リスト
        image_shape (tuple): 画像のサイズ (height, width)

    Returns:
        ndarray: 交点配列 shape=(N,2)
    """
    # 輪郭を描画した画像を作成
    contour_image = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(contour_image, contours, -1, 255, 1)

    # ハフ変換で直線検出(垂直、水平の線に限定)
    lines = cv2.HoughLines(contour_image, 1, np.pi / 2, threshold=10)

    """ 
    # デバッグ用にlinesを画像化して表示
    if lines is not None:
        line_image = np.zeros(image_shape, dtype=np.uint8)
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            # 直線の始点と終点を計算
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(line_image, (x1, y1), (x2, y2), 255, 1)
        plt.imshow(line_image, cmap='gray')
        plt.title('Detected Lines')
        plt.axis('off')
        plt.show()
    """

    # 直線同士の交点を計算
    intersections = []
    if lines is not None:
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                rho1, theta1 = lines[i][0]
                rho2, theta2 = lines[j][0]
                # 直線のパラメータから交点を計算
                A = np.array([[np.cos(theta1), np.sin(theta1)],
                              [np.cos(theta2), np.sin(theta2)]])
                b = np.array([rho1, rho2])
                try:
                    intersection = np.linalg.solve(A, b)
                    intersections.append(intersection)
                except np.linalg.LinAlgError:
                    continue    # 直線が平行な場合は解（交点）が存在しないためスキップ
                
    # 交点を整数座標に変換
    intersections = [tuple(np.round(intersection).astype(float)) for intersection in intersections if np.all(np.isfinite(intersection))]
    intersections = list(set(intersections))  # 重複を削除

    # 画像のサイズに合わせて交点をフィルタリング
    intersections = [pt for pt in intersections if 0 <= pt[0] < image_shape[1] and 0 <= pt[1] < image_shape[0]]

    if intersections:
        intersections = np.array(intersections, dtype=float)
    else:
        intersections = np.empty((0,2), dtype=float)

    return intersections

def remove_redundant_contour_points(contours: List[np.ndarray]) -> List[np.ndarray]:
    """
    輪郭から冗長な頂点を削除する（削除しても面積が変わらない点を削除）
    
    Args:
        contours (list): 元の輪郭リスト
    
    Returns:
        list: 冗長な頂点が削除された輪郭リスト
    """
    cleaned_contours = []
    for cnt in contours:
        org_area = cv2.contourArea(cnt)
        if len(cnt) >= 3 and org_area > 0:  # 3点以下の輪郭、面積が0の輪郭は無視（削除）
            # 頂点を後ろから参照して、削除しても面積が変わらない点を削除
            for i in range(len(cnt)-1, -1, -1):
                cnt_cand = np.delete(cnt, i, axis=0)  # i番目の頂点を削除した候補輪郭
                # 面積が変わらない場合は採用
                if cv2.contourArea(cnt_cand) == org_area:
                    cnt = cnt_cand
            cleaned_contours.append(cnt)
    
    return cleaned_contours

def simplify_contours_with_hough_intersections(contours: List[np.ndarray], image_shape: Tuple[int, int], max_distance: float = 15.0) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    輪郭から抽出した直線の交点を使って輪郭をシンプル化する
    ただし、直線は水平、垂直に限定され、交点にスナップされなかった点は削除されるため、斜めの壁や曲線の壁がある図面では利用できない
    Args:
        contours (list): 元の輪郭リスト
        image_shape (tuple): 画像のサイズ (height, width)
        max_distance (float): スナップする最大距離
    
    Returns:
        tuple: (シンプル化された輪郭リスト, 使用された交点配列)
    """
    # 1. 輪郭から直線を検出し、交点を計算する
    intersections = detect_lines_and_intersections(contours, image_shape)
    
    if intersections.shape[0] == 0:
        return contours, intersections # 交点がない場合は元の輪郭を返す
    
    # 2. 輪郭の頂点を交点にスナップ（交点以外の頂点は削除）
    snapped_contours = snap_contours_to_points(contours, intersections, max_distance, keep_unsnapped=False)
    
    # 3. 輪郭における同一点、同一直線状の点を削除する（削除しても面積が変わらない点を削除）
    cleaned_contours = remove_redundant_contour_points(snapped_contours)

    return cleaned_contours, intersections

def visualize_intersections(image: np.ndarray, intersections: np.ndarray, color: Tuple[int, int, int] = (255, 0, 0), radius: int = 1) -> np.ndarray:
    """
    交点を描画する
    
    Args:
        image (ndarray): 描画対象の画像
        intersections (ndarray): 交点配列 shape=(N,2)
        color (tuple): 描画色 (B, G, R)
        radius (int): 描画する円の半径
    
    Returns:
        ndarray: 交点が描画された画像
    """
    output = image.copy()
    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    
    for intersection in intersections:
        x, y = intersection
        cv2.circle(output, (int(x), int(y)), radius, color, -1)
    
    return output

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
    num_list = [263, 325, 406, 426, 547, 581, 1407, 1423, 1521, 1759]
    data_folder = 'D:/20250929_layout_fitting3/data/'
    for i in range(len(num_list)):
        img_name = f'{num_list[i]:06d}.png'
        # img_name = '001759.png' # 斜めの壁がある図面
        # img_name = '000325.png' # 曲がった壁がある図面

        # 元画像の読み込み
        org_img = load_image_grayscale(f'{data_folder}{img_name}')

        # 元画像を加工して作成した壁画像の読み込み
        wall_img = load_image_grayscale(f'{data_folder}{img_name}')

        # 輪郭抽出
        contours = extract_room_contours(wall_img)
        
        # 面積が小さい輪郭を削除
        contours = drop_small_contours(contours, min_area=100.0)
        
        # 輪郭の可視化
        contours_on_org_img = visualize_contours(org_img, contours)

        ### 輪郭の直線化、頂点削減処理 ###
        # 輪郭をシンプルにする処理
        with section("simplify_contours"):
            simplified_contours, detected_intersections = simplify_contours_with_hough_intersections(contours, wall_img.shape, 10)

        # シンプル化された輪郭を元画像に描画
        simplified_contours_on_org_img = visualize_contours(org_img, simplified_contours)
        
        # 使用された交点も描画
        # simplified_contours_on_org_img = visualize_intersections(simplified_contours_on_org_img, detected_intersections)

        # デバッグ用に輪郭の数と各輪郭の頂点数を表示
        print("Debug: Number of contours:", len(simplified_contours))
        for i, contour in enumerate(simplified_contours):
            print(f"Debug: Number of vertices in contour {i}:", len(contour))

        # シンプル化の結果表示
        display_results(contours_on_org_img, simplified_contours_on_org_img)

        # 画像を保存
        root_name = img_name.split('.')[0]
        output_file = f'{data_folder}/{root_name}_simplified.png'
        init_file = f'{data_folder}/{root_name}_initial.png'
        cv2.imwrite(output_file, simplified_contours_on_org_img)
        cv2.imwrite(init_file, contours_on_org_img)

if __name__ == "__main__":
    main()