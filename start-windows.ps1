$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path backend/.venv/Scripts/python.exe)) { throw 'Run setup-windows.ps1 first.' }
if (-not (Test-Path frontend/dist/index.html)) { throw 'Build frontend first: cd frontend; npm.cmd run build' }
Set-Location backend
Write-Host 'Open http://127.0.0.1:8000 in your browser. Press Ctrl+C to stop.'
& .venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
