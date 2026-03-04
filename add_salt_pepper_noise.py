"""
example_lota_SeqID19_unit1 폴더의 이미지들에 동일한 salt-and-pepper noise를 추가하여 새 폴더에 저장

조건:
- 모든 이미지에 동일한 노이즈 패턴 (같은 위치, 같은 0/255 값)
- 노이즈 값: 0 또는 255만 사용
- 노이즈 비율: 전체 픽셀의 0.1%
"""
import argparse
import glob
import os
import numpy as np
import cv2


def create_salt_pepper_noise_mask(h, w, noise_ratio=0.001, general_ratio=0.0, seed=42):
    """
    Salt-and-pepper 노이즈 마스크 생성
    - 노이즈 위치는 동일하고, 모든 이미지에 동일 패턴 적용
    - 극단 노이즈: 0(pepper) 또는 255(salt)
    - 일반 노이즈: [0, 255] 구간의 랜덤 값 (general_ratio 비율)

    Args:
        h: 이미지 높이
        w: 이미지 너비
        noise_ratio: 전체 픽셀 대비 노이즈 비율 (기본 0.1% = 0.001)
        general_ratio: 노이즈 픽셀 중 일반(랜덤 강도) 비율 (0~1, 기본 0)
        seed: 재현성을 위한 난수 시드

    Returns:
        noise_mask: (h, w) shape, -1(원본 유지), 0~255(해당 값으로 대체)
    """
    rng = np.random.default_rng(seed)
    total_pixels = h * w
    num_noise = max(1, int(total_pixels * noise_ratio))

    # 노이즈가 적용될 픽셀 위치 (랜덤 선택)
    flat_indices = rng.choice(total_pixels, size=num_noise, replace=False)

    # 일반 노이즈 개수: [0, 255] 균등 랜덤
    num_general = int(num_noise * general_ratio)
    num_salt_pepper = num_noise - num_general

    # 극단 노이즈: 0 또는 255 (50:50)
    salt_pepper_values = rng.choice([0, 255], size=num_salt_pepper)
    # 일반 노이즈: [0, 255] 균등 분포
    general_values = rng.integers(0, 256, size=num_general, dtype=np.int32)

    noise_values = np.concatenate([salt_pepper_values, general_values])
    rng.shuffle(noise_values)

    # 마스크 생성: -1=원본 유지, 0~255=노이즈
    noise_mask = np.full((h, w), -1, dtype=np.int32)
    rows = flat_indices // w
    cols = flat_indices % w
    noise_mask[rows, cols] = noise_values

    return noise_mask


def apply_noise_to_image(img, noise_mask):
    """
    uint8 이미지에 노이즈 마스크 적용

    Args:
        img: uint8 이미지 (grayscale 또는 color)
        noise_mask: (h, w) 마스크, -1=유지, 0~255=해당 값으로 대체

    Returns:
        noised_img: 노이즈가 적용된 uint8 이미지
    """
    noised_img = img.copy()
    apply_mask = noise_mask >= 0  # 0 또는 255인 위치만 True
    if img.ndim == 2:
        noised_img[apply_mask] = noise_mask[apply_mask]
    else:
        # color 이미지: 각 채널에 동일 값 적용
        for c in range(img.shape[2]):
            noised_img[..., c][apply_mask] = noise_mask[apply_mask]
    return noised_img


def add_salt_pepper_noise(input_dir, output_dir, noise_ratio=0.001, general_ratio=0.0, pattern="*.bmp", seed=42):
    """
    입력 폴더의 모든 이미지에 동일한 salt-and-pepper 노이즈를 추가하여 출력 폴더에 저장

    Args:
        input_dir: 입력 이미지 폴더
        output_dir: 출력 이미지 폴더
        noise_ratio: 전체 픽셀 대비 노이즈 비율 (0.001 = 0.1%)
        general_ratio: 노이즈 픽셀 중 일반(랜덤 강도) 비율 (0~1)
        pattern: 파일 패턴
        seed: 재현성용 난수 시드
    """
    os.makedirs(output_dir, exist_ok=True)

    image_path = os.path.join(input_dir, pattern)
    image_files = sorted(glob.glob(image_path))

    if len(image_files) == 0:
        print(f"Error: No images found in {input_dir}")
        return

    # 첫 번째 이미지로 크기 확인 및 노이즈 마스크 생성
    first_img = cv2.imread(image_files[0])
    if first_img is None:
        print(f"Error: Failed to load {image_files[0]}")
        return

    h, w = first_img.shape[0], first_img.shape[1]
    noise_mask = create_salt_pepper_noise_mask(h, w, noise_ratio=noise_ratio, general_ratio=general_ratio, seed=seed)

    num_noise_pixels = np.sum(noise_mask >= 0)
    total_pixels = h * w
    actual_ratio = num_noise_pixels / total_pixels * 100
    num_salt = np.sum(noise_mask == 255)
    num_pepper = np.sum(noise_mask == 0)
    num_general = np.sum((noise_mask > 0) & (noise_mask < 255))
    print(f"Image size: {h}x{w} = {total_pixels:,} pixels")
    print(f"Noise pixels: {num_noise_pixels:,} ({actual_ratio:.3f}%)")
    print(f"Salt(255): {num_salt:,}, Pepper(0): {num_pepper:,}, General([1,254]): {num_general:,}")
    print("-" * 50)

    for img_path in image_files:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Failed to load {img_path}")
            continue

        # grayscale로 읽었는지 확인 (color이면 동일 로직 적용)
        if img.ndim == 3:
            # BGR 이미지: 각 채널에 동일한 노이즈 적용
            noised_img = apply_noise_to_image(img, noise_mask)
        else:
            noised_img = apply_noise_to_image(img, noise_mask)

        # uint8 보장
        noised_img = noised_img.astype(np.uint8)

        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, noised_img)
        print(f"Saved: {filename}")

    print("-" * 50)
    print(f"Done: {len(image_files)} images -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="이미지들에 동일한 salt-and-pepper 노이즈(0/255) 추가"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        nargs="?",
        default="example_lota_SeqID19_unit1",
        help="입력 이미지 폴더 (기본: example_lota_SeqID19_unit1)",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        nargs="?",
        default="example_lota_SeqID19_unit1_noisy",
        help="출력 이미지 폴더 (기본: example_lota_SeqID19_unit1_noisy)",
    )
    parser.add_argument(
        "--noise_ratio",
        type=float,
        default=0.001,
        help="전체 픽셀 대비 노이즈 비율 (0.001 = 0.1%%, 기본: 0.001)",
    )
    parser.add_argument(
        "--general_ratio",
        type=float,
        default=0.0,
        help="노이즈 픽셀 중 일반(랜덤 [0,255]) 비율 0~1 (기본: 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="재현성용 난수 시드 (기본: 42)",
    )

    args = parser.parse_args()

    print(f"Input dir: {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Noise ratio: {args.noise_ratio * 100}%")
    print(f"General ratio: {args.general_ratio * 100}%")
    print(f"Seed: {args.seed}")
    print("-" * 50)

    add_salt_pepper_noise(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        noise_ratio=args.noise_ratio,
        general_ratio=args.general_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    # usage: python add_salt_pepper_noise.py [input_dir] [output_dir] [--noise_ratio 0.001] [--seed 42]
    # example: 
    # python add_salt_pepper_noise.py ./example_data/lota_SeqID19 ./example_data/lota_SeqID19_noisy
    main()
