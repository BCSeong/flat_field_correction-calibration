# -*- coding: utf-8 -*-
"""
3x3 그리드 ground truth 밝기(컬러 타겟) 이미지를 생성합니다.
좌표: (1,1)=왼쪽위, (1,3)=오른쪽위, (3,1)=왼쪽아래, (3,3)=오른쪽아래.
각 셀에 RGB 색상과 색상코드 텍스트를 그립니다.
"""

import os
import numpy as np
import cv2

# 3x3 색상 (RGB 순서). 행: (1,1),(1,2),(1,3), (2,1),(2,2),(2,3), (3,1),(3,2),(3,3)
# 418은 0~255 범위 밖이므로 255로 클리핑 (오타일 경우 수정)
RGB_PATCHES = [
    [115, 82, 68],    # (1,1)
    [194, 150, 130],  # (1,2)
    [98, 122, 157],   # (1,3)
    [214, 126, 44],   # (2,1)
    [80, 91, 166],    # (2,2)
    [193, 90, 99],    # (2,3)
    [56, 61, 150],    # (3,1)
    [70, 148, 73],    # (3,2) 
    [175, 54, 60],    # (3,3)
]
'''
RGB_PATCHES = [
    [87, 108, 67],    # (1,4)
    [133, 128, 177],  # (1,5)
    [103, 189, 170],   # (1,6)
    [94, 60, 108],   # (2,4)
    [157, 188, 64],    # (2,5)
    [224, 163, 46],    # (2,6)
    [231, 199, 31],    # (3,4)
    [187, 86, 149],    # (3,5) 
    [8, 133, 161],    # (3,6)
]
'''

PATCH_SIZE = 280   # 셀 한 변 픽셀
BORDER = 2        # 셀 사이 테두리
FONT_SCALE = 1
FONT_THICKNESS = 2
TEXT_COLOR_WHITE = (255, 255, 255)
TEXT_COLOR_BLACK = (0, 0, 0)


def _contrast_color(r, g, b):
    """배경에 잘 보이도록 흰색/검정 중 하나 반환 (BGR)."""
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return TEXT_COLOR_WHITE if lum < 50 else TEXT_COLOR_BLACK


def create_ground_truth_image():
    """3x3 컬러 타겟 이미지 생성. 각 셀에 색상코드 텍스트 표시."""
    cell = PATCH_SIZE + BORDER
    w = h = 3 * cell + BORDER
    img = np.ones((h, w, 3), dtype=np.uint8) * 255  # 흰 배경

    for idx, (r, g, b) in enumerate(RGB_PATCHES):
        r, g, b = int(r), int(g), int(min(b, 255))
        row, col = idx // 3, idx % 3
        y0 = BORDER + row * cell
        x0 = BORDER + col * cell
        y1, x1 = y0 + PATCH_SIZE, x0 + PATCH_SIZE

        color_bgr = (b, g, r)
        img[y0:y1, x0:x1] = color_bgr

        # 색상코드 텍스트 (예: 115, 82, 68)
        label = f"{r}, {g}, {b}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
        tx = x0 + (PATCH_SIZE - tw) // 2
        ty = y0 + (PATCH_SIZE + th) // 2
        text_color = _contrast_color(r, g, b)
        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, text_color, FONT_THICKNESS, cv2.LINE_AA)

    return img


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(base_dir, "ground_truth_color_target.png")
    img = create_ground_truth_image()
    cv2.imwrite(out_path, img)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
