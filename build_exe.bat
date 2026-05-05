@echo off
echo ========================================
echo   Building Blob Tracker 4K .exe
echo ========================================
echo.

REM Nettoyer les anciens builds
if exist "build" rmdir /s /q build
if exist "dist" rmdir /s /q dist
if exist "*.spec" del /q *.spec

echo Building executable...
pyinstaller --onefile --console --name "BlobTracker4K" blob_tracker_menu.py

echo.
echo Copying required files...
copy config.txt dist\ 2>nul

echo.
echo ========================================
echo   Build complete!
echo ========================================
echo.
echo Executable: dist\BlobTracker4K.exe
echo.
pause