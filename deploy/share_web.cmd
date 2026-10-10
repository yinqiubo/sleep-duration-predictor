@echo off
REM ---------------------------------------------------------------------------
REM One-click: publish the prototype on a TEMPORARY public https URL.
REM   Uses Cloudflare "quick tunnel": no account, no sign-up.
REM   The URL looks like https://<random>.trycloudflare.com and dies when this
REM   window is closed. Anyone with the URL can open it - see deploy\README.md.
REM
REM All messages here are ASCII-only on purpose (Windows cmd parses .cmd bytes
REM with the active console codepage). Chinese docs: deploy\README.md
REM ---------------------------------------------------------------------------

setlocal
cd /d "%~dp0.."
set CLOUDFLARED=%~dp0cloudflared.exe

echo ============================================================
echo  Publish prototype on a temporary public URL
echo ============================================================
echo.

echo [1/4] checking cloudflared.exe ...
if not exist "%CLOUDFLARED%" (
    echo       not found. Downloading from GitHub ^(~35 MB, one time only^) ...
    powershell -NoProfile -Command "$ProgressPreference='SilentlyContinue'; Invoke-WebRequest -Uri 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' -OutFile '%CLOUDFLARED%'"
)
if not exist "%CLOUDFLARED%" (
    echo [ERROR] download failed. Download it manually from
    echo         https://github.com/cloudflare/cloudflared/releases
    echo         save it as deploy\cloudflared.exe and run this file again.
    pause
    exit /b 1
)

echo [2/4] starting the Streamlit prototype on 127.0.0.1:8501 ...
start "sleep-prototype" /min python -m scripts.run_app --local-only

echo [3/4] waiting until port 8501 accepts connections ...
powershell -NoProfile -Command "for ($i=0; $i -lt 45; $i++) { if ((Test-NetConnection -ComputerName 127.0.0.1 -Port 8501 -InformationLevel Quiet -WarningAction SilentlyContinue)) { exit 0 }; Start-Sleep -Seconds 2 }; exit 1"
if errorlevel 1 echo       [WARN] port 8501 is not open yet, trying the tunnel anyway ...

echo [4/4] creating the tunnel ...
echo.
echo    ^>^> When you see the line "https://xxxx.trycloudflare.com", that is your
echo       public web address. Open it in any browser (phone or PC, any network).
echo    ^>^> Keep this window open while people use it.
echo    ^>^> Press Ctrl+C here to stop the tunnel; then run deploy\stop_web.cmd
echo       to also shut down the local app.
echo.

"%CLOUDFLARED%" tunnel --url http://localhost:8501 --protocol http2 --logfile "%~dp0_tunnel.log"

echo.
echo tunnel closed.
pause
