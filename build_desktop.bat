@echo off
setlocal

REM ========================================
REM   LiteLLM Studio - Build Desktop
REM ========================================
echo.
echo ========================================
echo    LiteLLM Studio - Build Desktop
echo ========================================
echo.

REM ---- Paths ----
set "PROJ_DIR=%~dp0"
set "VENV_DIR=%PROJ_DIR%env"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

REM ---- Garantir venv (usa o do projeto se já existir) ----
if not exist "%PYTHON_EXE%" (
  echo [1/5] Creating virtual environment at "%VENV_DIR%"...
  py -3.12 -m venv "%VENV_DIR%" || goto :fail
) else (
  echo [1/5] Using existing virtual environment at "%VENV_DIR%"
)

REM ---- Upgrade pip toolchain ----
echo [2/5] Upgrading pip/setuptools/wheel...
"%PYTHON_EXE%" -m pip install -U pip setuptools wheel || goto :fail

REM ---- Instalar deps de build (projeto + pyinstaller) ----
echo [3/5] Installing build dependencies...
"%PYTHON_EXE%" -m pip install -e "%PROJ_DIR%.[dev]" || "%PYTHON_EXE%" -m pip install -e "%PROJ_DIR%" || goto :fail
"%PYTHON_EXE%" -m pip install -U pyinstaller || goto :fail

REM ---- Limpar builds anteriores ----
echo [4/5] Cleaning previous builds...
if exist "%PROJ_DIR%dist"  rmdir /s /q "%PROJ_DIR%dist"
if exist "%PROJ_DIR%build" rmdir /s /q "%PROJ_DIR%build"

REM ---- Build com spec ----
echo.
echo [5/5] Creating executable...
echo This may take a few minutes...
echo.
cd /d "%PROJ_DIR%"
"%PYTHON_EXE%" -m PyInstaller "%PROJ_DIR%build.spec" --clean --noconfirm || goto :fail

REM Verify if the build was successful
if exist "dist\LiteLLM Studio.exe" (
    echo.
    echo ========================================
    echo BUILD COMPLETED SUCCESSFULLY!
    echo ========================================
    echo.
    echo Executable created: dist\LiteLLM-Studio.exe
    echo File size: dir "dist\LiteLLM Studio.exe" | findstr "Studio"
    echo.
    echo To test: dist\LiteLLM Studio.exe
    echo.
) else (
    echo.
    echo ========================================
    echo BUILD FAILED!
    echo ========================================
    echo.
    echo Check the logs for error details.
    echo.
)
