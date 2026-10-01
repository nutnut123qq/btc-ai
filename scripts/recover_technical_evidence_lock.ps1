param(
    [string]$OutputDirectory = "docs/research/evidence/technical-descriptive",
    [double]$MaxLockAgeHours = 12
)

$ErrorActionPreference = "Stop"
$AiRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $AiRoot "venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "AI virtual-environment Python was not found at $Python"
}

Push-Location $AiRoot
try {
    & $Python "run_technical_evidence_pipeline.py" --output-dir $OutputDirectory --recover-stale-lock --max-lock-age-hours $MaxLockAgeHours
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
