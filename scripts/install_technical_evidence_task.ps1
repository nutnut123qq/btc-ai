[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$TaskName = "BTCTechnicalDescriptiveEvidence",
    [string]$DailyAt = "02:20",
    [string]$OutputDirectory = "docs/research/evidence/technical-descriptive",
    [switch]$ApplyRetention,
    [ValidateRange(2, 30)][int]$KeepSuccessfulRuns = 2,
    [switch]$AuditOnly
)

$ErrorActionPreference = "Stop"
$AiRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$Runner = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "run_technical_evidence_pipeline.ps1"))
if (-not (Test-Path -LiteralPath $Runner)) { throw "Runner not found: $Runner" }
$Python = Join-Path $AiRoot "venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw "AI virtual-environment Python not found: $Python" }
$ResolvedOutput = if ([IO.Path]::IsPathRooted($OutputDirectory)) {
    [IO.Path]::GetFullPath($OutputDirectory)
} else {
    [IO.Path]::GetFullPath((Join-Path $AiRoot $OutputDirectory))
}

Push-Location $AiRoot
try {
    & $Python -c "from technical_event_modules import verify_golden_fixture; verify_golden_fixture()"
    if ($LASTEXITCODE -ne 0) { throw "Golden technical-event semantic preflight failed" }
}
finally { Pop-Location }

$At = [DateTime]::ParseExact($DailyAt, "HH:mm", [Globalization.CultureInfo]::InvariantCulture)
$PowerShell = Join-Path $env:SystemRoot "System32/WindowsPowerShell/v1.0/powershell.exe"
$Arguments = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$Runner`" -OutputDirectory `"$ResolvedOutput`""
if ($ApplyRetention) {
    $Arguments += " -ApplyRetention -KeepSuccessfulRuns $KeepSuccessfulRuns"
}
$ExpectedWorkingDirectory = $AiRoot

if ($AuditOnly) {
    $Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
    $Actions = @($Task.Actions)
    if ($Actions.Count -ne 1 `
        -or $Actions[0].Execute -ne $PowerShell `
        -or $Actions[0].Arguments -ne $Arguments `
        -or $Actions[0].WorkingDirectory -ne $ExpectedWorkingDirectory) {
        throw "Scheduled Task action differs from the reviewed technical-evidence wrapper"
    }
    Write-Host "Technical evidence task action verified: $TaskName"
    return
}

$Action = New-ScheduledTaskAction -Execute $PowerShell -Argument $Arguments -WorkingDirectory $ExpectedWorkingDirectory
$Trigger = New-ScheduledTaskTrigger -Daily -At $At
$Settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 4)
if ($PSCmdlet.ShouldProcess($TaskName, "install finite technical-evidence task")) {
    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings `
        -Description "Finite BTCUSDT 1h/4h/1d descriptive technical-evidence publisher" -Force | Out-Null
}

Write-Host "Technical evidence task configured: $TaskName"
Write-Host "Action: $PowerShell $Arguments"
Write-Host "Working directory: $ExpectedWorkingDirectory"
Write-Host "Daily at: $DailyAt"
