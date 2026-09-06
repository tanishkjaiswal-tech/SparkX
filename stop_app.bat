@echo off
REM GeoSR application stopper.
REM Kills the processes LISTENing on ports 3000 (frontend) and 8000 (backend) by PID.
REM Targeted by port - does NOT kill unrelated node/python processes.
setlocal enabledelayedexpansion
echo [GeoSR] Stopping frontend (:3000) and backend (:8000) processes ...
for %%P in (3000 8000) do (
  for /f "tokens=5" %%I in ('netstat -aon ^| findstr /R "^  *TCP.*:%%P "') do (
    if not "%%I"=="" (
      echo   Killing PID %%I on port %%P
      taskkill /f /pid %%I 2>nul
    )
  )
)
echo [GeoSR] Done.
endlocal
