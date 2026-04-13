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
echo = 3 - Update all dependencies      =
echo = -------------------------------- =
echo = 0 - Exit                         =
echo ====================================
echo.
set /p choice="Select action [1-3 or 0]: "

if "%choice%"=="1" goto install
if "%choice%"=="2" goto start
if "%choice%"=="3" goto update
if "%choice%"=="0" exit /b
goto menu

:: Install Module Dependencies
:install_module_deps
set "module_path=%~1"
if exist "%module_path%\requirements.txt" (
    echo Installing dependencies for module: %~nx1
    pip install -r "%module_path%\requirements.txt"
    if %ERRORLEVEL% NEQ 0 (
        echo Warning: Failed to install dependencies for module %~nx1
        exit /b 1
    )
    echo ✓ Module %~nx1 dependencies installed
)
exit /b 0

:: Install All Dependencies
:install_all_deps
echo Installing main dependencies from root requirements.txt...
if exist "requirements.txt" (
    pip install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo Error installing main dependencies
        exit /b 1
    )
    echo ✓ Main dependencies installed
) else (
    echo Warning: root requirements.txt not found
)

echo.
echo Checking for module dependencies...

if exist "modules\" (
    for /d %%i in ("modules\*") do (
        call :install_module_deps "%%i"
    )
) else (
    echo No modules folder found, skipping module dependencies
)
exit /b 0

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

call :install_all_deps
if %ERRORLEVEL% NEQ 0 (
    echo Error installing dependencies
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

:: Update Dependencies
:update
cls
echo Updating all dependencies...
echo.

if not exist "venv\" (
    echo Virtual environment not found. Please run full installation first.
    pause
    goto menu
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

echo Updating main dependencies...
if exist "requirements.txt" (
    pip install --upgrade -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo Error updating main dependencies
    ) else (
        echo ✓ Main dependencies updated
    )
)

if exist "modules\" (
    echo.
    echo Updating module dependencies...
    for /d %%i in ("modules\*") do (
        if exist "%%i\requirements.txt" (
            echo Updating dependencies for module: %%~nxi
            pip install --upgrade -r "%%i\requirements.txt"
            if %ERRORLEVEL% EQU 0 (
                echo ✓ Module %%~nxi dependencies updated
            )
        )
    )
)

echo.
color a
echo Dependencies update completed!
color b
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