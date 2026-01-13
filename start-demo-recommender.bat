@echo off
REM Script khoi dong nhanh demo Recommender System

echo ========================================
echo   DEMO RECOMMENDER SYSTEM - KHOI DONG
echo ========================================
echo.

echo [1/2] Khoi dong API Server...
cd VuaVuiVe_Recommender
start "Recommendation API Server" cmd /k "python src/api.py"

echo.
echo [2/2] Cho API khoi dong...
timeout /t 5 /nobreak > nul

echo.
echo ========================================
echo   THANH CONG!
echo ========================================
echo.
echo API Server: http://localhost:5001
echo Demo Web:   VuaVuiVe_Recommender\demo.html
echo.
echo Huong dan:
echo 1. Mo file demo.html bang trinh duyet web
echo 2. Nhap User ID (1-206209) de lay goi y
echo 3. Hoac them san pham vao gio hang de test
echo.
echo Nhan bat ky phim nao de mo demo.html...
pause > nul

start "" "VuaVuiVe_Recommender\demo.html"

echo.
echo Demo da duoc mo! Chuc ban thanh cong!
echo.
pause
