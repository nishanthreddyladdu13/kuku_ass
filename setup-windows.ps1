$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw 'Install Python 3.12 from python.org, including the Python launcher.' }
if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) { throw 'Install Node.js 22.12 or newer, then reopen PowerShell.' }
& py -3.12 -m venv backend/.venv
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required. Install it and rerun setup.' }
& backend/.venv/Scripts/python.exe -m pip install -r backend/requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
if (-not (Test-Path backend/.env)) { Copy-Item backend/.env.example backend/.env }
Push-Location frontend
try {
 & npm.cmd ci
 if ($LASTEXITCODE -ne 0) { throw 'npm ci failed.' }
 & npm.cmd run build
 if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
} finally { Pop-Location }
Write-Host 'Setup complete. Add your API key and available model ID to backend/.env for new text generation.'
Write-Host 'Run: powershell -ExecutionPolicy Bypass -File .\start-windows.ps1'
