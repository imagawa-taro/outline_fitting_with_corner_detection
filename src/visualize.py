import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Noto Sans CJK JP'  # 日本語フォント設定
from typing import List

def visualize_contours(image: np.ndarray, contours: List[np.ndarray]) -> np.ndarray:
    """
    輪郭を可視化する
    Args:
        image (ndarray): 元のグレースケ画像
        contours (list): 輪郭リスト
    Returns:
        ndarray: 描画結果画像
    """
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cmap = plt.get_cmap('Set1')
    PALETTE = [(np.array(cmap(i)[:3]) * 255).astype(int)[::-1] for i in range(cmap.N)]
    for i, cnt in enumerate(contours):
        if len(cnt) >= 3:
            line_color = PALETTE[i % len(PALETTE)]
            fill_color = (line_color * 0.5 + 255 * 0.5).astype(int)
            cv2.fillPoly(output, [cnt], fill_color.tolist())
            cv2.drawContours(output, [cnt], -1, line_color.tolist(), 1)
            # 頂点を表示
            for p in cnt:
                cv2.circle(output, tuple(p[0]), 1, (255, 0, 0), -1)
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Before')
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title('After')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig('contours.png')
    plt.close(fig)  # これを追加


def plot_single_contour(contour: np.ndarray, index: int) -> None:
    """
    単一の輪郭を表示する
    Args:
        contour (ndarray): 輪郭
        index (int): 輪郭のインデックス
    """
    plt.figure()
    #始点と終点は色で区別
    plt.plot(contour[:, 0, 0], contour[:, 0, 1], marker='o')
    plt.plot(contour[0, 0, 0], contour[0, 0, 1], marker='o', color='red', label='Start')
    plt.plot(contour[-1, 0, 0], contour[-1, 0, 1], marker='o', color='blue', label='End')
    #最初の点は最後の点とつなげる
    plt.plot([contour[-1, 0, 0], contour[0, 0, 0]], [contour[-1, 0, 1], contour[0, 0, 1]], color='gray')
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'Contour {index}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()
    plt.grid()
    plt.show()
    # plt.close()