@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title Vựa Vui Vẻ - Full Stack Launcher
color 0A

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║           VỰA VUI VẺ - KHỞI ĐỘNG DỰ ÁN                       ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM Kiểm tra Node.js
where node >nul 2>nul
if !errorlevel! neq 0 (
    echo [X] Lỗi: Node.js chưa được cài đặt!
    echo     Vui lòng tải Node.js tại: https://nodejs.org/
    pause
    exit /b 1
)

echo [OK] Node.js: 
node --version
echo.

REM Kiểm tra Python
set SKIP_ML=1
where python >nul 2>nul
if !errorlevel! equ 0 (
    echo [OK] Python: 
    python --version
    set SKIP_ML=0
) else (
    echo [!] Cảnh báo: Python chưa được cài đặt!
    echo     ML Recommender sẽ không khả dụng
)
echo.

REM Dừng các process Node.js cũ nếu có
echo [*] Dừng các process cũ...
taskkill /F /IM node.exe >nul 2>nul
timeout /t 2 /nobreak >nul

REM Kiểm tra dependencies
echo.
echo [*] Kiểm tra dependencies...
if not exist "node_modules" (
    echo     Thiếu dependencies cho dự án chính
    echo [*] Đang cài đặt...
    call npm install
    if !errorlevel! neq 0 (
        echo [X] Lỗi khi cài đặt dependencies!
        pause
        exit /b 1
    )
)

if not exist "vnpay_nodejs\node_modules" (
    echo     Thiếu dependencies cho VNPay
    echo [*] Đang cài đặt...
    cd vnpay_nodejs
    call npm install
    cd ..
    if !errorlevel! neq 0 (
        echo [X] Lỗi khi cài đặt VNPay dependencies!
        pause
        exit /b 1
    )
)

REM Kiểm tra ML API dependencies
if !SKIP_ML! equ 0 (
    if not exist "VuaVuiVe_Recommender\models\nmf_model.pkl" (
        echo [!] Cảnh báo: ML model chưa được train!
        echo     Hãy chạy notebooks trong VuaVuiVe_Recommender/notebooks/
    )
    
    python -c "import flask" >nul 2>nul
    if !errorlevel! neq 0 (
        echo [*] Đang cài đặt Python dependencies cho ML API...
        cd VuaVuiVe_Recommender
        pip install -r requirements.txt
        cd ..
        if !errorlevel! neq 0 (
            echo [X] Lỗi khi cài đặt ML dependencies!
            set SKIP_ML=1
        )
    )
)

echo [OK] Dependencies đã sẵn sàng
echo.

REM Khởi động các services
echo ╔══════════════════════════════════════════════════════════════╗
echo ║           ĐANG KHỞI ĐỘNG CÁC SERVICES                        ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

echo [1/4] Khởi động VNPay Backend ^(Port 8888^)...
start "VNPay Backend - Port 8888" cmd /k "cd /d %~dp0vnpay_nodejs && color 0E && npm start"
timeout /t 3 /nobreak >nul

echo [2/4] Khởi động API Backend ^(Port 3000^)...
start "API Backend - Port 3000" cmd /k "cd /d %~dp0 && color 0B && npm run backend"
timeout /t 3 /nobreak >nul

echo [3/4] Khởi động Frontend ^(Port 8000^)...
start "Frontend - Port 8000" cmd /k "cd /d %~dp0 && color 0D && npm run frontend"
timeout /t 3 /nobreak >nul

if !SKIP_ML! equ 0 (
    echo [4/4] Khởi động ML Recommender API ^(Port 5001^)...
    start "ML Recommender API - Port 5001" cmd /k "cd /d %~dp0VuaVuiVe_Recommender && color 0C && python src/api.py"
    timeout /t 3 /nobreak >nul
) else (
    echo [4/4] ML API bỏ qua ^(Python không khả dụng^)
)

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                    CÁC LIÊN KẾT QUAN TRỌNG                   ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo  Trang chủ:            http://localhost:8000/Home/Hôm/index.html
echo  Gợi ý ML:             http://localhost:8000/html/recommended.html
echo  Giỏ hàng:             http://localhost:8000/html/cart.html
echo  Tài khoản:            http://localhost:8000/html/account.html
echo  Backoffice:           http://localhost:8000/backoffice/
echo.
echo  API Endpoints:
echo    ├─  Products API:    http://localhost:3000/products
echo    ├─  Users API:       http://localhost:3000/users
echo    ├─  Orders API:      http://localhost:3000/orders
echo    ├─  VNPay API:       http://localhost:8888/order/create_payment_url
if !SKIP_ML! equ 0 (
echo    └─  ML Recommender:  http://localhost:5001/health
) else (
echo    └─  ML Recommender:  [Khong kha dung]
)
echo.

REM Đợi services khởi động
echo [*] Đang đợi services khởi động...
timeout /t 5 /nobreak >nul

REM Kiểm tra trạng thái từng service
echo.
echo [*] Kiểm tra trạng thái services:

REM Check VNPay (8888)
netstat -ano | findstr ":8888" | findstr "LISTENING" >nul
if !errorlevel! equ 0 (
    echo    ├─ [OK] VNPay Backend ^(8888^)
) else (
    echo    ├─ [!] VNPay Backend chua khoi dong
)

REM Check API Backend (3000)
netstat -ano | findstr ":3000" | findstr "LISTENING" >nul
if !errorlevel! equ 0 (
    echo    ├─ [OK] API Backend ^(3000^)
) else (
    echo    ├─ [!] API Backend chua khoi dong
)

REM Check Frontend (8000) - quan trọng nhất
echo    ├─ Đợi Frontend Server ^(8000^)...
set FRONTEND_READY=0
for /L %%i in (1,1,20) do (
    netstat -ano | findstr ":8000" | findstr "LISTENING" >nul
    if !errorlevel! equ 0 (
        set FRONTEND_READY=1
        goto frontend_check_done
    )
    timeout /t 1 /nobreak >nul
)
:frontend_check_done

if !FRONTEND_READY! equ 1 (
    echo    ├─ [OK] Frontend Server ^(8000^) - Sẵn sàng!
) else (
    echo    ├─ [!] Frontend Server chua khoi dong
)

REM Check ML API (5001)
if !SKIP_ML! equ 0 (
    netstat -ano | findstr ":5001" | findstr "LISTENING" >nul
    if !errorlevel! equ 0 (
        echo    └─ [OK] ML Recommender ^(5001^)
    ) else (
        echo    └─ [!] ML Recommender chua khoi dong
    )
) else (
    echo    └─ [SKIP] ML Recommender
)

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                    MỞ BROWSER                                ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM Mở browser nếu frontend sẵn sàng
if !FRONTEND_READY! equ 1 (
    echo [OK] Đang mở browser...
    timeout /t 2 /nobreak >nul
    echo     ├─ Trang chủ...
    start "" "http://localhost:8000/Home/Hôm/index.html"
    timeout /t 1 /nobreak >nul
    echo     └─ Backoffice...
    start "" "http://localhost:8000/backoffice/"
    echo.
    echo [OK] Browser đã được mở thành công!
) else (
    echo [X] Frontend chua san sang - Browser khong tu dong mo
    echo     Vui long doi them 10-20s roi mo thu cong:
    echo     • Trang chu: http://localhost:8000/Home/Hôm/index.html
    echo     • Backoffice: http://localhost:8000/backoffice/
)

echo.
if !SKIP_ML! equ 1 (
    echo [!] Lưu ý: ML Recommender khong chay!
    echo     Để bật: Cài Python va chạy: pip install -r VuaVuiVe_Recommender/requirements.txt
    echo.
)

echo ╔══════════════════════════════════════════════════════════════╗
echo ║           ĐÃ KHỞI ĐỘNG XONG - CHÚC BẠN CODE VUI VẺ!          ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo  Mẹo:
echo    • Để dừng: Đóng các cửa sổ terminal hoặc Ctrl+C
echo    • Để khởi động lại: Chạy lại file này
echo    • Xem log: Kiểm tra các cửa sổ terminal màu
echo.
echo  Cửa sổ này có thể đóng. Services sẽ tiếp tục chạy.
echo.
pause
