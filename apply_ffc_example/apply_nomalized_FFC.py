# -*- coding: utf-8 -*-
"""
Normalized FFC (Flat-Field Correction): raw 이미지를 normalized background map으로 나눈 뒤
Bayer GR 디모자이싱하여 BGR로 저장합니다.
KYCAL 등에서 제공하는 [0,1] 정규화된 배경 맵을 사용합니다.

Usage:
  대화형 (bat에서 인자 없이 실행 시 프롬프트로 입력):
    python apply_nomalized_FFC.py

  Single image:
    python apply_nomalized_FFC.py --input raw.bmp --ffc_map ffc_map.bmp --output out.bmp
  Batch (input/output 폴더, 파일명 stem으로 background 매칭):
    python apply_nomalized_FFC.py --input ./color_target --ffc_map ./ffc_map --output ./result_colorTarget
"""

import os
import sys
import argparse
import numpy as np
import cv2
import tifffile

EPS = 1e-6


def load_raw_grayscale(path):
    """이미지를 단일 채널로 로드. Bayer decode 가 진행되지 않은 grayscale 이미지를 로드."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")    
    if img.ndim == 3:
        raise ValueError(f"Image is not grayscale: {path}")
    if img.dtype != np.uint8:
        raise ValueError(f"Image is not uint8, expected uint8: {path}")        
    return img

def load_ffc_map_grayscale(path):
    """FFC map을 단일 채널로 로드. Bayer decode 가 진행되지 않은 grayscale 이미지를 로드."""
    img = tifffile.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")    
    if img.ndim == 3:
        raise ValueError(f"Image is not grayscale: {path}")
    if img.dtype == np.uint8:
        raise ValueError(f"Image is uint8, expected float: {path}")        
    return img


def process_ffc(raw, raw_bg_normalized):
    """
    raw를 normalized background map으로 픽셀별 나눈 뒤 Bayer GR 디모자이싱하여 BGR 반환.
    raw_bg_normalized: [0, 1] 범위 float.
    """
    raw = np.asarray(raw, dtype=np.float64)
    bg = np.asarray(raw_bg_normalized, dtype=np.float64)
    divisor = np.clip(bg + EPS, 0.0, 1.0)
    ratio = (raw / divisor).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(ratio, cv2.COLOR_BAYER_GR2BGR)


def process_one(input_path, ffc_map_path, output_path):
    """단일 이미지 처리."""
    raw = load_raw_grayscale(input_path)
    ffc_map = load_ffc_map_grayscale(ffc_map_path)
    if raw.shape != ffc_map.shape:
        raise ValueError(
            f"Shape mismatch: input {raw.shape} vs ffc_map {ffc_map.shape}"
        )
    out = process_ffc(raw, ffc_map)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, out)
    print(f"Saved: {output_path}")


def process_batch(input_dir, ffc_map_dir, output_dir):
    """폴더 내 이미지들을 stem으로 ffc_map 매칭하여 처리."""
    ffc_map_paths = {}
    for f in os.listdir(ffc_map_dir):
        if f.startswith("."):
            continue
        stem = os.path.splitext(f)[0]
        ffc_map_paths[stem] = os.path.join(ffc_map_dir, f)

    for root, _, files in os.walk(input_dir):
        for fname in sorted(files):
            if not fname.lower().endswith((".bmp", ".png", ".jpg", ".tif")):
                continue
            stem = os.path.splitext(fname)[0]
            if stem not in ffc_map_paths:
                print(f"[WARN] no ffc_map for {fname}, skip")
                continue
            rel = os.path.relpath(root, input_dir)
            if rel == ".":
                out_sub = output_dir
            else:
                out_sub = os.path.join(output_dir, rel)
            input_path = os.path.join(root, fname)
            ffc_map_path = ffc_map_paths[stem]
            output_path = os.path.join(out_sub, fname)
            raw = load_raw_grayscale(input_path)
            ffc_map = load_ffc_map_grayscale(ffc_map_path)
            if raw.shape != ffc_map.shape:
                print(f"[WARN] shape mismatch {input_path} vs {ffc_map_path}, skip")
                continue
            out = process_ffc(raw, ffc_map)
            os.makedirs(out_sub, exist_ok=True)
            cv2.imwrite(output_path, out)
            print(f"Saved: {output_path}")


def prompt_if_empty(value, prompt_msg, default=None):
    """값이 비어 있으면 사용자에게 입력받아 반환."""
    s = (value or "").strip()
    if s:
        return s
    if default:
        return default.strip()
    try:
        return input(prompt_msg).strip()
    except EOFError:
        return ""


def main():
    parser = argparse.ArgumentParser(
        description="Normalized FFC: raw / ffc_map, then Bayer GR demosaic to BGR."
    )
    parser.add_argument(
        "--input", "-i", default=None,
        help="Input raw image path or directory of raw images",
    )
    parser.add_argument(
        "--ffc_map", "-f", default=None,
        help="Normalized FFC (Flat-Field Correction) map path or directory (0~1 float)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output BGR image path or output directory",
    )
    args = parser.parse_args()

    # 인자가 없으면 대화형으로 입력받기 (bat에서 실행 시 편리)
    inp = args.input and args.input.strip()
    bg = args.ffc_map and args.ffc_map.strip()
    out = args.output and args.output.strip()
    if not inp or not bg or not out:
        print("=== Normalized FFC: Interactive input ===\n")
        if not inp:
            inp = prompt_if_empty(inp, "Input (raw image or folder path): ")
        if not bg:
            bg = prompt_if_empty(bg, "FFC map (file or folder path): ")
        if not out:
            out = prompt_if_empty(out, "Output (output path or folder): ")
    if not inp or not bg or not out:
        print("Error: input, ffc_map, and output cannot be empty.", file=sys.stderr)
        sys.exit(1)

    inp = os.path.abspath(inp)
    bg = os.path.abspath(bg)
    out = os.path.abspath(out)

    if not os.path.exists(inp):
        print(f"Error: input does not exist: {inp}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(bg):
        print(f"Error: ffc_map does not exist: {bg}", file=sys.stderr)
        sys.exit(1)

    if os.path.isfile(inp):
        if not os.path.isfile(bg):
            print("Error: when input is a file, ffc_map must be a file.", file=sys.stderr)
            sys.exit(1)
        process_one(inp, bg, out)
    else:
        if not os.path.isdir(bg):
            print("Error: when input is a directory, ffc_map must be a directory.", file=sys.stderr)
            sys.exit(1)
        process_batch(inp, bg, out)

    print("Done.")


if __name__ == "__main__":
    main()
