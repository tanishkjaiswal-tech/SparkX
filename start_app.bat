@echo off
REM GeoSR application launcher.
REM Starts the FastAPI backend (with the trained checkpoint, if present) and the
REM Next.js frontend in separate console windows. The checkpoint env var is inherited
REM by the child process, so real-SR inference is used when GEOSR_CHECKPOINT exists.
setlocal
set "REPO=%~dp0"
set "CKPT=%REPO%model\checkpoints\geosr_v2\GeoSR_v2_epoch34_best.pt"
if exist "%CKPT%" (
  set "GEOSR_CHECKPOINT=%CKPT%"
  echo [GeoSR] Checkpoint found: %CKPT%
  echo [GeoSR] Backend starting in REAL-SM mode on :8000 ...
) else (
  set "GEOSR_CHECKPOINT="
  echo [GeoSR] No checkpoint found - backend starting in baseline (demo) mode.
)

start "GeoSR-Backend" /d "%REPO%backend" cmd /c "python -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
start "GeoSR-Frontend" /d "%REPO%frontend" cmd /c "npm run dev -- -p 3000"

echo.
echo [GeoSR] Servers are starting in separate console windows.
echo   Backend  : http://127.0.0.1:8000   (Swagger UI: http://127.0.0.1:8000/docs)
echo   Frontend : http://localhost:3000
echo.
echo   If ports 3000/8000 are already in use, run stop_app.bat first.
endlocal
