@echo off
chcp 65001 >nul
setlocal

REM ========== Move to script directory ==========
cd /d "%~dp0"

REM ========== Activate .venv39 in parent folder (apply_ffc_example) ==========
set "VENV_ACTIVATE=%~dp0..\..\.venv39\Scripts\activate.bat"
if exist "%VENV_ACTIVATE%" (
    call "%VENV_ACTIVATE%"
) else (
    echo [Error] .venv39 not found. Create it in parent folder: python -m venv .venv39
    pause
    exit /b 1
)

REM ========== 실행 (인자 없이 실행 시 대화형으로 입력) ==========
python apply_nomalized_FFC.py %*

REM 인자를 직접 넘기려면 예:
REM python apply_nomalized_FFC.py --input ./color_target --ffc_map ./ffc_map --output ./result_colorTarget

pause
