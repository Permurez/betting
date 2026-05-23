# Uruchom dashboard Streamlit (QuantBet)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\venv\Scripts\python.exe")) {
    Write-Host "Brak venv. Uruchom najpierw: .\setup.ps1" -ForegroundColor Yellow
    exit 1
}

& .\venv\Scripts\python.exe -m streamlit run src\app.py
