@echo off
chcp 65001 >nul
title Dừng Vựa Vui Vẻ
color 0C

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║              DỪNG TẤT CẢ SERVICES                            ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

echo  Đang dừng tất cả Node.js processes...
taskkill /F /IM node.exe >nul 2>nul

if %errorlevel% equ 0 (
    echo  Đã dừng thành công!
) else (
    echo ℹ  Không có process nào đang chạy
)

echo.
echo  Kiểm tra ports...
netstat -ano | findstr ":8888 :3000 :8000" | findstr "LISTENING" >nul
if %errorlevel% equ 0 (
    echo   Một số services vẫn đang chạy:
    netstat -ano | findstr ":8888 :3000 :8000" | findstr "LISTENING"
) else (
    echo  Tất cả ports đã được giải phóng
)

echo.
echo  Hoàn tất!
echo.
pause
