$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Venv = Join-Path $Root ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path $Python)) {
    python -m venv $Venv
}

$env:PIP_CACHE_DIR = Join-Path $Root ".cache\pip"
$env:TMP = Join-Path $Root ".tmp"
$env:TEMP = $env:TMP
$env:MPLCONFIGDIR = Join-Path $Root ".cache\matplotlib"
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null

& $Python -m pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0+cu128"
& $Python -m pip install -e ".[api,cross-encoder,dev,openai]"

Write-Host "Project environment ready: $Python"
