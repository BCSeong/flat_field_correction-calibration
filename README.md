# Flat Field Correction / Calibration 도구


---

## 요구 사항

- Python 3.7+
- 아래 패키지 (설치: `pip install -r requirements.txt`)

| 패키지 | 용도 |
|--------|------|
| numpy | 배열 연산 |
| opencv-python | 이미지 읽기/쓰기, 그레이스케일, 미디언 필터 |
| tifffile | TIFF 저장 |
| matplotlib | 디버그 프로파일/이미지 시각화 |

---

## 환경 설정과 실행 (Windows / 다른 PC)

다른 PC에서도 이 폴더만 받으면 실행할 수 있도록, 폴더 안에 가상환경(`.venv`)을 만드는 것을 권장합니다.

**1. 가상환경 만들기 (최초 1회)** — Python 3.7 이상 필요 (검증 환경: Python 3.9.12)

```bat
cd flat_field_correction-calibration
py -3.9 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**2. 실행**

- `run_ffc_calibration.bat` 를 더블클릭하거나, 명령창에서 인자를 붙여 실행합니다: `run_ffc_calibration.bat [입력폴더] [dark폴더] [cycle_length] [출력폴더]`
- 인자 없이 실행하면 필요한 값을 대화형으로 묻습니다.
- 끝나면 결과 코드(Exit code)를 표시하고 창이 닫히지 않고 대기합니다.

**`run_ffc_calibration.bat` 가 사용하는 Python (찾는 순서)**

1. 이 폴더의 `.venv` (위 1번에서 만든 환경)
2. 상위 폴더의 `.venv39` (lota_tools 공용 환경, 기존 방식)
3. PATH의 `py -3` 또는 `python`

셋 다 없으면 가상환경 만드는 방법을 출력하고 종료합니다.

**Linux / macOS** 는 bat 대신 직접 실행합니다.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python image_averaging.py [입력폴더] [dark폴더] [cycle_length] [출력폴더]
```

## 예제 데이터

예제 입력·출력은 용량이 커서 git에 포함하지 않습니다(최신 1세트만 공유). 공유 폴더에서 받아 이 폴더에 둡니다.

| 폴더 | 내용 |
|---|---|
| `example_data/lota_SeqID19/` | 입력 예제 (SeqID19, 8 조명 × 21 cycle) |
| `example_data/lota_dark/` | dark 이미지 |
| `output_lota_SeqID19_wo_normalization/` | 위 입력의 출력 예제 (`averaged_norm_*.tif`, `averaged_wo_norm_*.png`, `dark.bmp`, `debug/`) |

실행 예: `run_ffc_calibration.bat example_data\lota_SeqID19 example_data\lota_dark 8 output_lota_SeqID19_wo_normalization`

- 공유 폴더: `<공유 폴더 링크 — 추가 예정>`

## 1. add_salt_pepper_noise.py
** noise 가 있는 경우에도 FFC 로직이 잘 작동하는지 검토하기 위해 pseudo data 를 생성합니다 **
입력 폴더의 **모든 이미지에 동일한** salt-and-pepper 노이즈(0 또는 255)를 붙여 새 폴더에 저장합니다.

- 노이즈 위치·값이 이미지마다 동일 (같은 시드 사용)
- 비율 기본 0.1% (전체 픽셀 대비)
- 그레이스케일/컬러 모두 지원 (컬러는 각 채널에 동일 마스크 적용)

### 사용법

```bash
python add_salt_pepper_noise.py [input_dir] [output_dir] [--noise_ratio 0.001] [--seed 42]
```

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `input_dir` | `example_lota_SeqID19_unit1` | 입력 이미지 폴더 |
| `output_dir` | `example_lota_SeqID19_unit1_noisy` | 출력 이미지 폴더 |
| `--noise_ratio` | `0.001` | 노이즈 비율 (0.001 = 0.1%) |
| `--seed` | `42` | 재현용 난수 시드 |

### 예시

```bash
python add_salt_pepper_noise.py ./input_images ./noisy_images
python add_salt_pepper_noise.py ./input_images ./noisy_images --noise_ratio 0.002 --seed 123
```

---

## 2. image_averaging.py

이미지를 **cycle_length** 단위로 묶어 평균한 뒤, Bayer raw 기준으로 노이즈 제거·정규화하고 TIFF로 저장합니다. Dark 이미지도 함께 복사합니다.

- 입력: FFC용 이미지 폴더 + dark 이미지 1개만 있는 폴더
- 사이클별 평균 → Bayer 4채널 분리 후 채널별 미디언 필터 → 최대값 1 정규화 → TIFF 저장
- 디버그용으로 가로/세로 프로파일 + 이미지 레이아웃 PNG 생성

### 사용법

```bash
python image_averaging.py input_dir dark_image_dir [output_dir] --cycle_length N [--noise_removal true|false] [--kernel_size 5]
```

| 인자 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `input_dir` | ✓ | - | FFC 입력 이미지 디렉토리 |
| `dark_image_dir` | ✓ | - | dark 이미지 1개만 있는 디렉토리 |
| `output_dir` | - | `input_dir_averaged` | 출력 디렉토리 |
| `--cycle_length` | ✓ | - | 사이클 길이 (그룹당 이미지 수) |
| `--noise_removal` | - | vision_params.json | 노이즈 제거 여부 (`true` / `false`) |
| `--kernel_size` | - | vision_params.json | 미디언 필터 커널 크기 |

### 설정 파일 (image_averaging.py 경로 기준)

- **recipe.json**: cls 범위 (nClsDirectionalMin, nClsDirectionalMax, nClsdarkMin, nClsdarkMax)
- **vision_params.json**: noise_removal, kernel_size (미지정 시 사용)

처리 완료 후 good/bad 판정 출력 (directionalMin, directionalMax, directionalMean, darkMin, darkMax, darkMean 포함).

### 예시

```bash
python image_averaging.py ./ffc_images ./dark_folder --cycle_length 4
python image_averaging.py ./ffc_images ./dark_folder ./out_avg --cycle_length 8 --noise_removal true --kernel_size 5
```

### 출력

- `output_dir/averaged_0000.tif`, `averaged_0001.tif`, ... (정규화된 float32 TIFF)
- `output_dir/dark.bmp` (dark 이미지 복사)
- `output_dir/debug/report_normalized.png` (정규화 FFC map preview)
- `output_dir/debug/report_raw.png` (정규화 전 raw FFC map preview)

---

## 3. apply_ffc_example

생성된 FFC map을 raw 이미지에 적용하는 예제입니다. Raw를 normalized background map으로 나눈 뒤 카메라 Bayer 패턴으로 디모자이싱하여 BGR로 저장합니다.

### 사용법

```bash
cd apply_ffc_example
# 단일 이미지
python apply_nomalized_FFC.py --input raw.bmp --ffc_map ffc_map.tif --output out.bmp
# 배치 (폴더 단위, 파일명 stem으로 매칭)
python apply_nomalized_FFC.py --input ./raw_images --ffc_map ./ffc_map --output ./result
```

| 인자 | 설명 |
|------|------|
| `--input` / `-i` | Raw 이미지 경로 또는 디렉토리 |
| `--ffc_map` / `-f` | 정규화된 FFC map 경로 또는 디렉토리 ([0,1] float) |
| `--output` / `-o` | 출력 BGR 이미지 경로 또는 디렉토리 |

---

## 설치 및 실행 요약

```bash
pip install -r requirements.txt
# Salt-pepper 노이즈 추가
python add_salt_pepper_noise.py [입력폴더] [출력폴더]
# 이미지 averaging (FFC map 생성)
python image_averaging.py [입력폴더] [dark폴더] --cycle_length [숫자]
# FFC map 적용
cd apply_ffc_example && python apply_nomalized_FFC.py -i [raw] -f [ffc_map] -o [출력]
```
---

## launch.json으로 예제 실행하기

이미 준비된 `launch.json`을 활용해 VSCode에서 바로 예제를 실행해 볼 수 있습니다.  
상단의 **실행 및 디버그** 탭에서 원하는 설정을 선택해 실행하세요.
