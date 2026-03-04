# Normalized FFC 적용 예제

Raw 이미지를 정규화된 FFC(Flat-Field Correction) 맵으로 나눈 뒤 Bayer GR 디모자이싱하여 BGR로 저장합니다.

## 설치

```bash
python -m venv .venv_py39
.venv_py39\Scripts\activate
pip install -r requirements.txt
```

## 실행

- **대화형**: bat에서 인자 없이 실행하면 입력/FFC맵/출력 경로를 순서대로 입력합니다.
  ```bash
  python apply_nomalized_FFC.py
  ```
- **단일 이미지**: `--input`, `--ffc_map`, `--output` 에 각각 파일 경로 지정.
- **배치**: 세 인자 모두 폴더 경로로 지정 (파일명 stem으로 FFC 맵 매칭).

Windows에서는 `run_apply_normalized_FFC.bat` 를 실행한 뒤 대화형으로 경로를 입력하면 됩니다.
