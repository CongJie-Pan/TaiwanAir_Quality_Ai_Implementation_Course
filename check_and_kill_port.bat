@echo off
chcp 65001 >nul
echo ========================================
echo 檢查端口 8501 占用情況
echo ========================================
echo.

echo [步驟 1] 檢查端口 8501...
netstat -ano | findstr :8501

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ✅ 端口 8501 未被占用，可以正常使用！
    echo.
    pause
    exit /b 0
)

echo.
echo ⚠️ 發現端口 8501 被占用！
echo.

for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8501') do (
    set PID=%%a
    goto :found
)

:found
echo [步驟 2] 占用端口的進程 PID: %PID%
echo.

echo [步驟 3] 查詢進程信息...
tasklist | findstr %PID%
echo.

echo ========================================
echo 是否要終止此進程？
echo ========================================
echo.
echo 注意：請確認上方顯示的程序名稱
echo      如果是重要程序（如系統服務），請勿終止！
echo.
set /p CONFIRM=確定要終止進程 %PID% 嗎？(Y/N):

if /i "%CONFIRM%" NEQ "Y" (
    echo.
    echo ❌ 已取消操作
    echo.
    echo 💡 建議：使用其他端口運行 Streamlit
    echo    streamlit run src/main/python/app.py --server.port 8502
    echo.
    pause
    exit /b 1
)

echo.
echo [步驟 4] 正在終止進程 %PID%...
taskkill /PID %PID% /F

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ 成功終止進程！
    echo.
    echo 現在可以運行 Streamlit：
    echo    streamlit run src/main/python/app.py
    echo.
) else (
    echo.
    echo ❌ 終止進程失敗！可能需要管理員權限
    echo.
    echo 💡 解決方案：
    echo    1. 以管理員身份重新運行此腳本
    echo    2. 或使用其他端口：streamlit run src/main/python/app.py --server.port 8502
    echo.
)

pause
