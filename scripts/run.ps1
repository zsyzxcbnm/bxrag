param(
    [Parameter(Position = 0)]
    [ValidateSet("up", "down", "install", "init", "ingest", "eval", "chunking", "index-bench", "ablation", "judge", "zh-charts", "all", "test", "stats", "reset")]
    [string]$Command = "all"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

switch ($Command) {
    "up" { docker compose up -d db }
    "down" { docker compose down }
    "install" { & $Python -m pip install -e ".[api,cross-encoder,dev,openai]" }
    "init" { & $Python -m src.cli init-db }
    "ingest" { & $Python -m src.cli ingest }
    "eval" { & $Python -m src.cli eval }
    "chunking" { & $Python -m src.cli compare-chunking }
    "index-bench" { & $Python -m src.cli benchmark-index }
    "ablation" { & $Python -m src.cli ablation-metadata }
    "judge" { & $Python -m src.cli judge }
    "zh-charts" { & $Python scripts/generate_zh_charts.py }
    "all" { & $Python -m src.cli all }
    "test" { & $Python -m pytest }
    "stats" { & $Python -m src.cli stats }
    "reset" { & $Python -m src.cli reset-db --yes }
}

