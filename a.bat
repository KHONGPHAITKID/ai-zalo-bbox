@echo off
echo Searching and removing all folders named "extract_images"...
for /d /r %%i in (*) do (
    if /i "%%~nxi"=="extract_images" (
        echo Deleting folder: %%i
        rd /s /q "%%i"
    )
)
echo Done!
pause
