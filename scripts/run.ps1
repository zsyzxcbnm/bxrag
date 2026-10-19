param(
    [Parameter(Position = 0)]
    [ValidateSet("up", "down", "install", "init", "ingest", "eval", "chunking", "index-bench", "zh-charts", "all", "test", "stats", "reset")]
    [string]$Command = "all"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

switch ($Command) {
    "up" { docker compose up -d db }
    "down" { docker compose down }
    "install" { python -m pip install -e ".[dev]" }
    "init" { python -m src.cli init-db }
    "ingest" { python -m src.cli ingest }
    "eval" { python -m src.cli eval }
    "chunking" { python -m src.cli compare-chunking }
    "index-bench" { python -m src.cli benchmark-index }
    "zh-charts" { python scripts/generate_zh_charts.py }
    "all" { python -m src.cli all }
    "test" { pytest }
    "stats" { python -m src.cli stats }
    "reset" { python -m src.cli reset-db --yes }
}

