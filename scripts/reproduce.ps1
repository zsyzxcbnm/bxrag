param(
    [ValidateSet("offline", "bge-gpu")]
    [string]$Profile = "offline"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    & (Join-Path $Root "scripts\setup.ps1")
}

$env:MPLCONFIGDIR = Join-Path $Root ".cache\matplotlib"
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null
$env:DENSE_WEIGHT = "0.5"
$env:SPARSE_WEIGHT = "0.5"
$env:RERANKER_PROVIDER = "heuristic"

if ($Profile -eq "offline") {
    $env:BACKEND = "memory"
    $env:EMBEDDING_PROVIDER = "local"
    $env:EMBEDDING_MODEL = "local-tfidf-svd"
    $env:EMBEDDING_DIM = "128"
    $env:REPORTS_DIR = "reports/offline"
    & $Python -m src.cli all --backend memory --refit
    & $Python scripts/generate_zh_charts.py
    Write-Host "Offline reproduction complete: reports/offline"
    exit 0
}

docker compose up -d db
$env:BACKEND = "postgres"
$env:POSTGRES_DSN = "postgresql://apex:apex@localhost:5432/apex"
$env:EMBEDDING_PROVIDER = "sentence-transformers"
$env:EMBEDDING_MODEL = "data\models\bge-large-en-v1.5"
$env:EMBEDDING_DIM = "1024"
$env:EMBEDDING_DEVICE = "cuda"
$env:RERANKER_MODEL = "data\models\bge-reranker-large"
$env:RERANKER_DEVICE = "cuda"
$env:RERANKER_FP16 = "true"
$env:JUDGE_PROVIDER = "local"
$env:JUDGE_MODEL = "data\models\Qwen2.5-0.5B-Instruct"
$env:JUDGE_DEVICE = "cuda"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:REPORTS_DIR = "reports"

$ModelArgs = @()
if (-not (Test-Path (Join-Path $Root "data\models\bge-large-en-v1.5"))) {
    $ModelArgs += "bge-embedding"
}
if (-not (Test-Path (Join-Path $Root "data\models\bge-reranker-large"))) {
    $ModelArgs += "bge-reranker"
}
if ($ModelArgs.Count -gt 0) {
    Write-Host "Downloading missing models: $($ModelArgs -join ', ')"
    & $Python scripts\download_models.py @ModelArgs
}

$RequiredModels = @(
    (Join-Path $Root $env:EMBEDDING_MODEL),
    (Join-Path $Root $env:RERANKER_MODEL)
)
foreach ($Model in $RequiredModels) {
    if (-not (Test-Path $Model)) {
        throw "Missing model after download: $Model"
    }
}

& $Python -m src.cli reset-db --yes
& $Python -m src.cli all
& $Python -m src.cli benchmark-index
& $Python -m src.cli ablation-metadata
Copy-Item "reports\results.json" "reports\results_bge_large_pgvector.json" -Force
Copy-Item "reports\results.csv" "reports\results_bge_large_pgvector.csv" -Force
Copy-Item "reports\results.md" "reports\results_bge_large_pgvector.md" -Force
& $Python scripts/generate_zh_charts.py
Write-Host "BGE GPU reproduction complete: reports"
