@echo off
chcp 65001
cd /D "%~dp0"
set PATH=%PATH%;%SystemRoot%\system32

:: Main Menu
:menu
cls
color b
echo.
echo ####################################
echo #      NeuroVT - Control Menu      #
echo ####################################
echo.
echo ====================================
echo = 1 - Full installation            =
echo = 2 - Run without installation     =
echo = -------------------------------- =
echo = 0 - Exit                         =
echo ====================================
echo.
set /p choice="Select action [1-0]: "

if "%choice%"=="1" goto install
if "%choice%"=="2" goto start
if "%choice%"=="0" exit /b
goto menu

:: Full Installation
:install
cls
echo Starting automatic installation...
echo This may take several minutes depending on your internet connection...
echo.

echo Starting installation process...

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo Error creating virtual environment
        pause
        goto menu
    )
) else (
    echo Virtual environment already exists, skipping creation...
)

echo Activating virtual environment...
call venv\Scripts\activate
if %ERRORLEVEL% NEQ 0 (
    echo Error activating virtual environment
    pause
    goto menu
)

echo Updating pip...
python.exe -m pip install --upgrade pip
if %ERRORLEVEL% NEQ 0 (
    echo Error updating pip
    pause
    goto menu
)

echo Installing main dependencies...
pip install torchaudio
pip install flask
pip install torch
pip install omegaconf
pip install torchcodec
pip install numpy
pip install soundfile
pip install openai
pip install vosk
pip install pyaudio
if %ERRORLEVEL% NEQ 0 (
    echo Error installing main dependencies
    pause
    goto menu
)
timeout /t 3 >nul
cls

color a
echo.
echo ##################################################################
echo #              Installation completed successfully!              #
echo ##################################################################
echo =                                                                =
echo =       You can now run the application from the main menu       =
echo =                                                                =
echo ==================================================================
echo.
timeout /t 3 >nul
cls
call venv\Scripts\activate
color b
python main.py
pause
goto menu

:: Run without installation
:start
cls
echo Starting main.py...

SET has_error=0

if exist "venv\" (
    echo Activating virtual environment...
    call venv\Scripts\activate
    if %ERRORLEVEL% NEQ 0 (
        echo Warning: Failed to activate virtual environment
        SET has_error=1
    )
) else (
    echo Warning: Virtual environment not found
    SET has_error=1
)

color b
python main.py
IF %ERRORLEVEL% NEQ 0 (
    SET has_error=1
)

pause
goto menu