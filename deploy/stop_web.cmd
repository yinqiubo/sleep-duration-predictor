@echo off
REM ---------------------------------------------------------------------------
REM Stop whatever deploy\share_web.cmd started: the Cloudflare tunnel process
REM and the Streamlit app listening on port 8501.
REM ASCII-only on purpose (see share_web.cmd).
REM ---------------------------------------------------------------------------

echo stopping the app listening on 8501 ...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }"

echo stopping cloudflared ...
taskkill /f /im cloudflared.exe >nul 2>nul

echo.
powershell -NoProfile -Command "$l = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue; $t = Get-Process cloudflared -ErrorAction SilentlyContinue; if ($l) { Write-Output 'WARN: something still listens on 8501' } else { Write-Output 'port 8501 is free' }; if ($t) { Write-Output 'WARN: cloudflared still running' } else { Write-Output 'cloudflared is stopped' }"
echo done.
pause
