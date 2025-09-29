import cv2
import numpy as np
import matplotlib.pyplot as plt
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
    plt.show()
