$root = "C:\Users\DHyo\RAG\RAG-simple"
$envFile = Join-Path $root "frontend\.env.local"

if (-not (Test-Path $envFile)) {
    throw "Khong tim thay frontend\.env.local"
}

$tokenLine = Get-Content $envFile |
    Where-Object { $_ -match '^\s*UNIGUIDE_API_TOKEN\s*=' } |
    Select-Object -Last 1

if (-not $tokenLine) {
    throw "Khong tim thay UNIGUIDE_API_TOKEN"
}

$env:UNIGUIDE_API_TOKEN = (
    ($tokenLine -split '=', 2)[1]
).Trim().Trim('"').Trim("'")

if ([string]::IsNullOrWhiteSpace($env:UNIGUIDE_API_TOKEN)) {
    throw "API token dang rong"
}

$env:UNIGUIDE_RETRIEVAL_MODE = "hybrid"

Set-Location (Join-Path $root "backend")

Write-Host "Starting UniGuide AI Backend..."
Write-Host "Retrieval: Hybrid"
Write-Host "Address: http://127.0.0.1:8000"

& .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
