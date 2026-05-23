# QuantBet – jednorazowa instalacja (Windows PowerShell)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Brak python w PATH. Zainstaluj Python 3.10+ z python.org" -ForegroundColor Red
    exit 1
}

# Preferuj Python 3.11 jesli dostepny (stabilniejsze wheel'e); inaczej domyslny python
$py = "python"
if (Get-Command py -ErrorAction SilentlyContinue) {
    $list = py -0p 2>$null | Out-String
    if ($list -match "3\.11") { $py = "py -3.11" }
}

Invoke-Expression "$py -m venv venv"
& .\venv\Scripts\python.exe -m pip install --upgrade pip
& .\venv\Scripts\python.exe -m pip install -r requirements.txt
& .\venv\Scripts\python.exe src\main.py

Write-Host ""
Write-Host "Gotowe. Uruchom dashboard:" -ForegroundColor Green
Write-Host "  .\run.ps1"
Write-Host "  (lub: .\venv\Scripts\python.exe -m streamlit run src\app.py)"
