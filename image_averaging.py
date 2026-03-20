import argparse
import json
import tifffile
import glob
import os
import numpy as np
import cv2


def load_images(image_dir, pattern="*.bmp"):
    """
    Step1: 이미지 로드 (glob)
    
    Args:
        image_dir: 이미지가 있는 디렉토리 경로
        pattern: 파일 패턴 (기본값: "*.bmp")
    
    Returns:
        정렬된 이미지 파일 경로 리스트
    """
    image_path = os.path.join(image_dir, pattern)
    image_files = glob.glob(image_path)
    image_files.sort()  # 파일명 순서대로 정렬
    
    print(f"Number of images loaded: {len(image_files)}")
    return image_files

def load_dark_image(dark_image_dir):
    """
    Step1: dark 이미지 로드 (uint8 배열 반환).
    """
    dark_image_files = load_images(dark_image_dir)
    if len(dark_image_files) == 0:
        raise ValueError("Error: No dark image found.")
    path = dark_image_files[0]
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Error: Failed to load dark image: {path}")
    return img


def average_images_by_cycle(image_files, cycle_length):
    """
    Step2: 주어진 cycle_length 별로 이미지 average 하여 반환
    
    Args:
        image_files: 이미지 파일 경로 리스트
        cycle_length: 사이클 길이
    
    Returns:
        averaged 이미지 리스트 (float32, 개수 = cycle_length)
    """
    if len(image_files) == 0:
        raise ValueError("No image files.")
    
    # 첫 번째 이미지를 로드하여 크기 확인
    first_img = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
    if first_img is None:
        raise ValueError(f"Failed to load image: {image_files[0]}")
    
    h, w = first_img.shape
    num_groups = cycle_length
    averaged_images_float = []
    
    # 각 그룹별로 이미지 평균 계산
    for group_idx in range(num_groups):
        # 해당 그룹에 속하는 이미지 인덱스들 수집
        group_indices = []
        img_idx = group_idx
        while img_idx < len(image_files):
            group_indices.append(img_idx)
            img_idx += cycle_length
        
        if len(group_indices) == 0:
            continue
        
        # 그룹 내 이미지들을 로드하여 누적
        sum_img_float = np.zeros((h, w), dtype=np.float32)
        count = 0
        
        for idx in group_indices:
            img = cv2.imread(image_files[idx], cv2.IMREAD_GRAYSCALE)
            if img is not None:
                sum_img_float += img.astype(np.float32)
                count += 1
        
        if count > 0:
            avg_img_float = sum_img_float / count  # float32로 평균 계산
            averaged_images_float.append(avg_img_float)
            print(f"Group {group_idx}: averaged {len(group_indices)} images")
    
    
    
    print(f"Created {len(averaged_images_float)} averaged images in total")
    return averaged_images_float

def _split_bayer(img):
    """
    Bayer 이미지를 4개 서브 채널로 분할 (각 2x2 위치).
    2x2 Bayer block: (0,0), (0,1), (1,0), (1,1) 위치의 픽셀을 각각 추출.
    """
    ch0 = img[0::2, 0::2].copy()
    ch1 = img[0::2, 1::2].copy()
    ch2 = img[1::2, 0::2].copy()
    ch3 = img[1::2, 1::2].copy()
    return ch0, ch1, ch2, ch3


def _merge_bayer(ch0, ch1, ch2, ch3):
    """4개 서브 채널을 Bayer (H x W) 형태로 재결합."""
    h, w = ch0.shape[0] * 2, ch0.shape[1] * 2
    out = np.zeros((h, w), dtype=ch0.dtype)
    out[0::2, 0::2] = ch0
    out[0::2, 1::2] = ch1
    out[1::2, 0::2] = ch2
    out[1::2, 1::2] = ch3
    return out


def _denoise_single_channel(sub_img, kernel_size):
    """
    단일 채널(동일 색 픽셀만 포함)에 median filter 적용.
    채널 내부에서는 median filter 사용이 안전함 (다른 색 픽셀과 혼합되지 않음).
    Masking 없이 전체에 적용하여 경계 불연속을 피하고 stereo matching/photometry에 유리.
    """
    return cv2.medianBlur(sub_img, kernel_size)


def remove_noise(avg_img_float, kernel_size=5):
    """
    Bayer raw 이미지의 salt-and-pepper 노이즈만 선택적으로 제거.

    avg_img_float 은 Bayer pattern RGB 카메라의 de-bayer 되지 않은 이미지.
    출력도 de-bayer 되지 않은 Bayer raw로 유지.

    ** NOTE **
    - Raw Bayer에 median/smoothing을 직접 적용하면 R, G, B 픽셀이 섞여 색상 왜곡 발생.
    - Bayer를 4개 서브 채널로 분할 후, 각 채널(동일 색만 포함)에 대해 노이즈 제거 수행.
    - 각 서브 채널 내부에서는 median filter 사용이 안전 (동일 색 픽셀만 비교).

    Args:
        avg_img_float: averaging 된 Bayer raw 이미지 (float32)
        kernel_size: median filter 창 크기 (홀수, 서브 채널 크기에 맞게 설정)

    Returns:
        노이즈 제거된 Bayer raw 이미지 (float32)
    """
    ch0, ch1, ch2, ch3 = _split_bayer(avg_img_float)
    ch0 = _denoise_single_channel(ch0, kernel_size)
    ch1 = _denoise_single_channel(ch1, kernel_size)
    ch2 = _denoise_single_channel(ch2, kernel_size)
    ch3 = _denoise_single_channel(ch3, kernel_size)
    return _merge_bayer(ch0, ch1, ch2, ch3)

def post_process_averaged_images(averaged_images_float, noise_removal=True, kernel_size=5, normalize=True):
    """
    Step3: averaged 이미지에 대해 노이즈 제거 및 (옵션) 정규화(최대값 1) 적용.

    Args:
        averaged_images_float: averaged 이미지 리스트 (float32)
        noise_removal: Bayer 채널별 median 노이즈 제거 여부
        kernel_size: median filter 창 크기 (noise_removal=True 일 때)
        normalize: True면 최대값 1로 정규화, False면 정규화 생략

    Returns:
        (normalized_images, raw_images): normalized=[0,1] float32, raw=정규화 전 float32
    """
    normalized = []
    raw_images = []
    for idx, avg_img in enumerate(averaged_images_float):
        if noise_removal:
            avg_img = remove_noise(avg_img, kernel_size=kernel_size)
        raw_images.append(avg_img.copy())
        if normalize:
            max_val = np.max(avg_img)
            if max_val > 0:
                avg_img = avg_img / max_val
        normalized.append(avg_img)
    return normalized, raw_images


def save_averaged_images(averaged_images_float, output_dir, base_name="averaged", ext=".tif"):
    """
    Step4: (이미 post-process 된) float32 이미지들을 지정 확장자로 저장.

    Args:
        averaged_images_float: 이미지 리스트 (float32, 정규화 시 [0,1], 미정규화 시 [0,~255])
        output_dir: 출력 디렉토리
        base_name: 출력 파일명 기본 이름
        ext: 확장자 ".bmp" | ".tif" | ".png"
            - .bmp: uint8 변환 후 저장
            - .tif: float32 그대로 저장
            - .png: float32를 16bit(0-65535)로 변환 후 저장
    """
    ext = ext.lower() if ext.startswith(".") else "." + ext.lower()
    os.makedirs(output_dir, exist_ok=True)
    for idx, img in enumerate(averaged_images_float):
        img = np.asarray(img, dtype=np.float32)
        output_filename = f"{base_name}_{idx:04d}{ext}"
        output_path = os.path.join(output_dir, output_filename)
        if ext == ".bmp":
            img_uint8 = np.clip(img, 0, 255).round().astype(np.uint8)
            cv2.imwrite(output_path, img_uint8)
        elif ext == ".tif":
            tifffile.imwrite(output_path, img)
        elif ext == ".png":            
            img_uint8 = np.clip(img, 0, 255).round().astype(np.uint8)
            cv2.imwrite(output_path, img_16)
        elif ext == ".png_16bit":            
            img_16 = (np.clip(img, 0, 1) * 65535).round().astype(np.uint16)            
            cv2.imwrite(output_path, img_16)
        else:
            raise ValueError(f"Unsupported extension: {ext} (.bmp, .tif, .png)")
        print(f"Saved: {output_path}")
    print(f"Saved {len(averaged_images_float)} images in total")

def save_dark_image(dark_image_uint8, output_dir, base_name="dark"):
    """
    Step4: dark 이미지 저장
    """
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{base_name}.bmp"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, dark_image_uint8)
    print(f"Saved: {output_path}")

def _draw_debug_figure(entries, axhline_values, value_ylims, supertitle, output_path, reference_line_at_1=False, reference_line_values=None):
    """
    레이아웃: N행 x 3열. 각 행 = 한 이미지.
    axhline_values: 각 행별 (cls_min, cls_max) 또는 (vmin, vmax) - axhline에 사용 (reference_line_at_1=True면 무시)
    value_ylims: 각 행별 (y_lo, y_hi) - y축 범위
    reference_line_at_1: True면 min/max 빨간선 없이 파란 점선만 그림 (normalized figure용)
    reference_line_values: reference_line_at_1=True일 때 각 행별 blue dotted line y값 리스트 (없으면 1.0 사용)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(entries)
    fig, axs = plt.subplots(n_rows, 3, figsize=(12, 2 * n_rows))
    fig.suptitle(supertitle, fontsize=14, fontweight="bold")

    if n_rows == 1:
        axs = axs.reshape(1, -1)

    hlw = 3

    for row, img in enumerate(entries):
        h, w = img.shape[0], img.shape[1]
        y_lo, y_hi = value_ylims[row]
        cls_min, cls_max = axhline_values[row]
        is_bottom = row == n_rows - 1
        ax_h, ax_v, ax_im = axs[row, 0], axs[row, 1], axs[row, 2]

        ax_h.plot(np.arange(w), img[h // 2, :], color="black")
        if reference_line_at_1:
            y_ref = reference_line_values[row] if reference_line_values is not None else 1.0
            ax_h.axhline(y_ref, color="blue", linestyle=":", linewidth=hlw)
        else:
            ax_h.axhline(cls_min, color="red", linewidth=hlw)
            ax_h.axhline(cls_max, color="red", linewidth=hlw)
        ax_h.set_ylim(y_lo, y_hi)
        ax_h.set_ylabel("value")
        if row == 0:
            ax_h.set_title("Horizontal profile")
        ax_h.grid(axis="y")
        if not is_bottom:
            ax_h.set_xticks([])
            ax_h.set_xlabel("")
        else:
            ax_h.set_xlabel("x")

        ax_v.plot(np.arange(h), img[:, w // 2], color="black")
        if reference_line_at_1:
            y_ref = reference_line_values[row] if reference_line_values is not None else 1.0
            ax_v.axhline(y_ref, color="blue", linestyle=":", linewidth=hlw)
        else:
            ax_v.axhline(cls_min, color="red", linewidth=hlw)
            ax_v.axhline(cls_max, color="red", linewidth=hlw)
        ax_v.set_ylim(y_lo, y_hi)
        ax_v.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax_v.set_yticklabels([])
        ax_v.set_ylabel("")
        if row == 0:
            ax_v.set_title("Vertical profile")
        ax_v.grid(axis="y")
        if not is_bottom:
            ax_v.set_xticks([])
            ax_v.set_xlabel("")
        else:
            ax_v.set_xlabel("y")

        vmin_im, vmax_im = (0.0, 1.0) if y_hi <= 1.1 else (y_lo, y_hi)
        ax_im.imshow(img, cmap="gray", vmin=vmin_im, vmax=vmax_im)
        ax_im.set_xticks([])
        ax_im.set_yticks([])
        ax_im.set_title(f"img_{row + 1}")

    fig.tight_layout(rect=[0, 0, 1, 0.92], pad=0.3)
    fig.subplots_adjust(wspace=0.2, hspace=0.25)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def save_debug_images(
    averaged_images_float,
    raw_images_float,
    dark_image_uint8,
    output_dir,
    nClsDirectionalMin,
    nClsDirectionalMax,
    nClsdarkMin,
    nClsdarkMax,
):
    """
    Step5: debug 이미지 저장 (matplotlib 사용).
    두 figure: normalized ffc map preview, raw ffc map preview.
    normalized figure: axhline = vmin, vmax (데이터 min/max)
    raw figure: axhline = cls_min, cls_max (recipe.json 값)
    """
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    # Normalized entries: float [0,1]
    norm_entries = []
    norm_axhline = []
    norm_ylims = []
    for img in averaged_images_float:
        img_f = np.clip(img.astype(np.float64), 0.0, 1.0)
        norm_entries.append(img_f)
        norm_axhline.append((float(np.min(img_f)), float(np.max(img_f))))
        norm_ylims.append((-0.1, 1.1))
    img_f = dark_image_uint8.astype(np.float64)
    norm_entries.append(img_f)
    norm_axhline.append((float(np.min(img_f)), float(np.max(img_f))))
    norm_ylims.append((nClsdarkMin - 10, nClsdarkMax + 10))

    norm_reference = [1.0] * len(averaged_images_float) + [float(np.max(dark_image_uint8))]
    _draw_debug_figure(
        norm_entries, norm_axhline, norm_ylims,
        supertitle="normalized ffc map preview",
        output_path=os.path.join(debug_dir, "report_normalized.png"),
        reference_line_at_1=True,
        reference_line_values=norm_reference,
    )

    # Raw entries: 정규화 전 (float [0,~255])
    raw_entries = [np.asarray(x, dtype=np.float64) for x in raw_images_float]
    raw_axhline = [(nClsDirectionalMin, nClsDirectionalMax)] * len(raw_images_float)
    raw_ylims = [(-10.0, 265.0)] * len(raw_images_float)
    raw_entries.append(dark_image_uint8.astype(np.float64))
    raw_axhline.append((nClsdarkMin, nClsdarkMax))
    raw_ylims.append((nClsdarkMin - 10, nClsdarkMax + 10))

    _draw_debug_figure(
        raw_entries, raw_axhline, raw_ylims,
        supertitle="raw ffc map preview",
        output_path=os.path.join(debug_dir, "report_raw.png"),
    )


def good_bad_judgment(
    raw_images_float,
    dark_image_uint8,
    nClsDirectionalMin,
    nClsDirectionalMax,
    nClsdarkMin,
    nClsdarkMax,
):
    """
    normalization 이전 FFC map이 cls 범위 내에 데이터가 존재하는지 확인.
    각 FFC 이미지의 good/bad, min, max, mean 출력.

    Returns:
        dict: directionalMin(array), directionalMax(array), directionalMean(array),
              darkMin(float), darkMax(float), darkMean(float), directional_good(array bool), dark_good(bool)
    """
    directional_min = np.array([float(np.min(img)) for img in raw_images_float])
    directional_max = np.array([float(np.max(img)) for img in raw_images_float])
    directional_mean = np.array([float(np.mean(img)) for img in raw_images_float])

    dark_min = float(np.min(dark_image_uint8))
    dark_max = float(np.max(dark_image_uint8))
    dark_mean = float(np.mean(dark_image_uint8))

    directional_good = (directional_min >= nClsDirectionalMin) & (directional_max <= nClsDirectionalMax)
    dark_good = (dark_min >= nClsdarkMin) & (dark_max <= nClsdarkMax)

    print("-" * 50)
    print("FFC Good/Bad Judgment (cls range)")
    print("-" * 50)
    for i in range(len(raw_images_float)):
        status = "good" if directional_good[i] else "bad"
        print(f"  img_{i + 1}: {status}  min={directional_min[i]:.2f} max={directional_max[i]:.2f} mean={directional_mean[i]:.2f}")
    status = "good" if dark_good else "bad"
    print(f"  dark: {status}  min={dark_min:.2f} max={dark_max:.2f} mean={dark_mean:.2f}")
    print("-" * 50)

    return {
        "directionalMin": directional_min,
        "directionalMax": directional_max,
        "directionalMean": directional_mean,
        "darkMin": dark_min,
        "darkMax": dark_max,
        "darkMean": dark_mean,
        "directional_good": directional_good,
        "dark_good": dark_good,
    }


def load_config(config_path, default=None):
    """JSON 설정 파일 로드. 없으면 default 반환."""
    default = default if default is not None else {}
    if not os.path.isfile(config_path):
        return default
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Average images by cycle_length and save")
    # Required positionals (optional in CLI so we can prompt interactively when missing)
    parser.add_argument("input_dir", type=str, nargs="?", default=None, help="Input image directory")
    parser.add_argument("dark_image_dir", type=str, nargs="?", default=None, help="Dark image directory (single DARK image)")
    parser.add_argument("cycle_length", type=str, nargs="?", default=None, help="Cycle length")
    parser.add_argument("output_dir", type=str, nargs="?", default=None, help="Output directory")
    # optional arguments
    parser.add_argument("--noise_removal", type=str, default=None, help="Noise removal (true/false). If not set, use vision_params.json")
    parser.add_argument("--kernel_size", type=int, default=None, help="Noise removal kernel size. If not set, use vision_params.json")
    parser.add_argument("--normalize", type=str, default=None, help="Normalize to max=1 (true/false). If not set, use vision_params.json or true")

    args = parser.parse_args()

    # Interactive prompt for missing required args (English for CMD compatibility)

    if args.input_dir is None:
        args.input_dir = input("input_dir (image directory): ").strip()
    if args.dark_image_dir is None:
        args.dark_image_dir = input("dark_image_dir (dark image directory): ").strip()
    if args.cycle_length is None:
        args.cycle_length = input("cycle_length (cycle length): ").strip()
    if args.output_dir is None:
        args.output_dir = input("output_dir (output directory): ").strip()

    try:
        args.cycle_length = int(args.cycle_length)
    except (TypeError, ValueError):
        parser.error("cycle_length must be an integer")

    cwd = os.path.dirname(os.path.abspath(__file__)) or "."
    recipe = load_config(os.path.join(cwd, "recipe.json"))
    vision_params = load_config(os.path.join(cwd, "vision_params.json"))

    nClsDirectionalMin = recipe.get("nClsDirectionalMin", 0)
    nClsDirectionalMax = recipe.get("nClsDirectionalMax", 255)
    nClsdarkMin = recipe.get("nClsdarkMin", 0)
    nClsdarkMax = recipe.get("nClsdarkMax", 255)

    if args.noise_removal is not None:
        noise_removal = str(args.noise_removal).lower() in ("true", "1", "yes")
    else:
        noise_removal = bool(vision_params.get("noise_removal", True))

    if args.kernel_size is not None:
        kernel_size = args.kernel_size
    else:
        kernel_size = int(vision_params.get("kernel_size", 5))

    if args.normalize is not None:
        normalize = str(args.normalize).lower() in ("true", "1", "yes")
    else:
        normalize = bool(vision_params.get("normalize", True))

    print(f"Input dir: {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Cycle length: {args.cycle_length}")
    print(f"Noise removal: {noise_removal} (from vision_params.json or --noise_removal)")
    print(f"Kernel size: {kernel_size} (from vision_params.json or --kernel_size)")
    print(f"Normalize: {normalize} (from vision_params.json or --normalize)")
    print("-" * 50)

    # Step1: ffc 이미지 로드, dark 이미지 로드
    image_files = load_images(args.input_dir)
    dark_image_uint8 = load_dark_image(args.dark_image_dir)

    if len(image_files) == 0:
        print("Error: No image files found.")
        return

    # Step2: cycle_length별로 averaging
    averaged_images_float = average_images_by_cycle(image_files, args.cycle_length)

    # Step3: 노이즈 제거 + (옵션) 최대값 1 정규화
    averaged_images_float, raw_images_float = post_process_averaged_images(
        averaged_images_float, noise_removal=noise_removal, kernel_size=kernel_size, normalize=normalize
    )

    # Step4: 이미지 저장 (정규화 시 tif, 미정규화 시 bmp / 정규화 시 정규화 전 결과도 tif로 저장)
    save_averaged_images(averaged_images_float, args.output_dir, base_name="averaged_norm", ext=".tif" if normalize else ".bmp")
    if normalize:
        save_averaged_images(raw_images_float, args.output_dir, base_name="averaged_wo_norm", ext=".png")
    save_dark_image(dark_image_uint8, args.output_dir, base_name="dark")

    # Step5: debug 이미지 저장 (normalized + raw figure)
    save_debug_images(
        averaged_images_float,
        raw_images_float,
        dark_image_uint8,
        args.output_dir,
        nClsDirectionalMin=nClsDirectionalMin,
        nClsDirectionalMax=nClsDirectionalMax,
        nClsdarkMin=nClsdarkMin,
        nClsdarkMax=nClsdarkMax,
    )

    # Step6: good/bad 판정 (모든 처리 완료 후 출력)
    good_bad_judgment(
        raw_images_float,
        dark_image_uint8,
        nClsDirectionalMin=nClsDirectionalMin,
        nClsDirectionalMax=nClsDirectionalMax,
        nClsdarkMin=nClsdarkMin,
        nClsdarkMax=nClsdarkMax,
    )


if __name__ == "__main__":
    # usage: python image_averaging.py <input_dir> <dark_image_dir> <cycle_length> <output_dir> [--noise_removal true|false] [--kernel_size 5] [--normalize true|false]
    # example: normalized output:
    # python image_averaging.py ./example_data/lota_SeqID19 ./example_data/lota_dark 8 ./output_with_normalize --noise_removal true --kernel_size 5 --normalize true
    # example: unnormalized output:
    # python image_averaging.py ./example_data/lota_SeqID19 ./example_data/lota_dark 8 ./output_without_normalize --noise_removal true --kernel_size 5 --normalize false
    main()

