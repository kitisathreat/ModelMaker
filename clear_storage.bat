@echo off
echo Clearing all Excel files from Storage folder...
echo.

REM Check if Storage folder exists
if not exist "Storage" (
    echo Storage folder not found. Creating it...
    mkdir Storage
    echo Storage folder created.
    goto :end
)

REM Change to the Storage directory
cd Storage

REM Count Excel files before deletion
set /a count=0
for %%f in (*.xlsx *.xls *.xlsm *.xlsb) do set /a count+=1

if %count%==0 (
    echo No Excel files found in Storage folder.
) else (
    echo Found %count% Excel file(s) to delete:
    echo.
    
    REM List files that will be deleted
    for %%f in (*.xlsx *.xls *.xlsm *.xlsb) do (
        echo   - %%f
    )
    echo.
    
    REM Ask for confirmation
    set /p confirm="Are you sure you want to delete these files? (y/N): "
    if /i "%confirm%"=="y" (
        echo.
        echo Deleting Excel files...
        
        REM Delete all Excel files
        del /q *.xlsx 2>nul
        del /q *.xls 2>nul
        del /q *.xlsm 2>nul
        del /q *.xlsb 2>nul
        
        echo.
        echo All Excel files have been deleted from the Storage folder.
    ) else (
        echo.
        echo Operation cancelled. No files were deleted.
    )
)

:end
echo.
echo Press any key to exit...
pause >nul 