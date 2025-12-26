@echo off
chcp 65001 >nul
cls
echo.
echo ========================================
echo   VỰA VUI VẺ - GROUP 5 SETUP SCRIPT
echo ========================================
echo.

REM Check Node.js
echo [1/5] Kiểm tra Node.js...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  ERROR: Node.js chưa được cài đặt!
    echo  Vui lòng tải tại: https://nodejs.org/
    pause
    exit /b 1
)
echo  OK: Node.js đã được cài đặt
node --version
echo.

REM Install dependencies
echo [2/5] Cài đặt dependencies...
call npm install
if %errorlevel% neq 0 (
    echo  Lỗi khi cài đặt dependencies!
    pause
    exit /b 1
)
echo  Đã cài đặt dependencies
echo.

REM Install JSON Server globally
echo [3/5] Cài đặt JSON Server...
call npm install -g json-server
if %errorlevel% neq 0 (
    echo   Không thể cài global, thử cài local...
    call npm install json-server --save-dev
)
echo  Đã cài đặt JSON Server
echo.

REM Sync products to database
echo [4/5] Đồng bộ sản phẩm vào database...
call npm run sync
if %errorlevel% neq 0 (
    echo   Lỗi khi sync products
    echo  Bạn có thể chạy lại sau: npm run sync
)
echo.

REM Done
echo [5/5] Setup hoàn tất!
echo.
echo ========================================
echo   CÁCH CHẠY PROJECT:
echo ========================================
echo.
echo    Chạy tự động (khuyến nghị):
echo      npm start
echo.
echo   🔧 Hoặc chạy thủ công:
echo      Terminal 1: npm run backend
echo      Terminal 2: npm run frontend
echo.
echo ========================================
echo   TRUY CẬP:
echo ========================================
echo.
echo    Frontend:  http://localhost:8000
echo    Recipes:   http://localhost:8000/html/recipes.html
echo    Admin:     http://localhost:8000/backoffice
echo     API:       http://localhost:3000
echo.
echo ========================================
echo.
pause
