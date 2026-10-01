[CmdletBinding()]
param(
    [string]$OutputDirectory = "docs/research/evidence/technical-descriptive",
    [ValidateRange(3, 90)][int]$LogRetentionCount = 14,
    [switch]$ApplyRetention,
    [ValidateRange(2, 30)][int]$KeepSuccessfulRuns = 2
)

$ErrorActionPreference = "Stop"
$AiRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$Python = Join-Path $AiRoot "venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "AI virtual-environment Python was not found at $Python"
}
$ResolvedOutput = if ([IO.Path]::IsPathRooted($OutputDirectory)) {
    [IO.Path]::GetFullPath($OutputDirectory)
} else {
    [IO.Path]::GetFullPath((Join-Path $AiRoot $OutputDirectory))
}
$OpsDirectory = Join-Path $AiRoot ".ops/technical-evidence"
$LogDirectory = Join-Path $OpsDirectory "logs"
$StatusPath = Join-Path $OpsDirectory "scheduler-status.json"
New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null

$Started = [DateTimeOffset]::UtcNow
$RunId = $Started.ToString("yyyyMMddTHHmmssZ") + "_" + [Guid]::NewGuid().ToString("N").Substring(0, 8)
$LogPath = Join-Path $LogDirectory "technical-evidence_$RunId.log"
$ExitCode = 1
$FailureType = $null

Push-Location $AiRoot
$StdErrPath = "$LogPath.stderr"
$PlanJsonPath = "$LogPath.plan.tmp"
try {
    try {
        # Native stderr must not touch the PowerShell error stream: under
        # ErrorActionPreference=Stop even a redirected warning becomes a
        # terminating RemoteException and masks the real exit code.
        # Start-Process redirection bypasses the stream machinery entirely.
        $proc = Start-Process -FilePath $Python -WorkingDirectory $AiRoot -NoNewWindow -Wait -PassThru `
            -ArgumentList @("run_technical_evidence_pipeline.py", "--output-dir", $ResolvedOutput) `
            -RedirectStandardOutput $LogPath -RedirectStandardError $StdErrPath
        $ExitCode = $proc.ExitCode
        if ($ExitCode -eq 2) { $FailureType = "PipelineLockedError" }
        elseif ($ExitCode -ne 0) { $FailureType = "PipelineNonZeroExit" }
        if ((Test-Path -LiteralPath $StdErrPath) -and (Get-Item -LiteralPath $StdErrPath).Length -gt 0) {
            "--- stderr ---" | Out-File -LiteralPath $LogPath -Encoding utf8 -Append
            Get-Content -LiteralPath $StdErrPath | Out-File -LiteralPath $LogPath -Encoding utf8 -Append
        }
        if ($ExitCode -eq 0 -and $ApplyRetention) {
            $proc = Start-Process -FilePath $Python -WorkingDirectory $AiRoot -NoNewWindow -Wait -PassThru `
                -ArgumentList @("technical_evidence_retention.py", "--output-dir", $ResolvedOutput, "--keep-successful-runs", "$KeepSuccessfulRuns") `
                -RedirectStandardOutput $PlanJsonPath -RedirectStandardError $StdErrPath
            if ($proc.ExitCode -ne 0) { throw "Retention dry-run failed" }
            $PlanText = Get-Content -LiteralPath $PlanJsonPath -Raw
            $PlanText | Out-File -LiteralPath $LogPath -Encoding utf8 -Append
            $Plan = $PlanText | ConvertFrom-Json
            if ($null -eq $Plan.blockedReason -and $Plan.candidateCount -gt 0) {
                $proc = Start-Process -FilePath $Python -WorkingDirectory $AiRoot -NoNewWindow -Wait -PassThru `
                    -ArgumentList @("technical_evidence_retention.py", "--output-dir", $ResolvedOutput, "--keep-successful-runs", "$KeepSuccessfulRuns", "--apply", "--confirm-plan-sha256", $Plan.planSha256) `
                    -RedirectStandardOutput $PlanJsonPath -RedirectStandardError $StdErrPath
                if ($proc.ExitCode -ne 0) { throw "Retention apply failed" }
                Get-Content -LiteralPath $PlanJsonPath | Out-File -LiteralPath $LogPath -Encoding utf8 -Append
            }
        }
    }
    catch {
        $ExitCode = 1
        $FailureType = $_.Exception.GetType().Name
        "runner failure type: $FailureType" | Out-File -LiteralPath $LogPath -Encoding utf8 -Append
    }
}
finally {
    Remove-Item -LiteralPath $StdErrPath -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $PlanJsonPath -Force -ErrorAction SilentlyContinue
    Pop-Location
    $Completed = [DateTimeOffset]::UtcNow
    $Status = [ordered]@{
        schema = "btc-technical-evidence-scheduler-status/v1"
        runId = $RunId
        succeeded = ($ExitCode -eq 0)
        exitCode = $ExitCode
        failureType = $FailureType
        startedAtUtc = $Started.ToString("o")
        completedAtUtc = $Completed.ToString("o")
        durationSeconds = [Math]::Round(($Completed - $Started).TotalSeconds, 3)
        logFileName = [IO.Path]::GetFileName($LogPath)
        outputDirectory = $ResolvedOutput
        retentionEnabled = [bool]$ApplyRetention
        keepSuccessfulRuns = $KeepSuccessfulRuns
    }
    $TemporaryStatus = "$StatusPath.$RunId.tmp"
    $Status | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $TemporaryStatus -Encoding UTF8
    Move-Item -LiteralPath $TemporaryStatus -Destination $StatusPath -Force

    $Logs = @(Get-ChildItem -LiteralPath $LogDirectory -File | Where-Object {
        $_.Name -match '^technical-evidence_\d{8}T\d{6}Z_[a-f0-9]{8}\.log$'
    } | Sort-Object LastWriteTimeUtc -Descending)
    foreach ($Expired in @($Logs | Select-Object -Skip $LogRetentionCount)) {
        if ($Expired.DirectoryName -ne $LogDirectory) { throw "Refusing log path outside scheduler log directory" }
        Remove-Item -LiteralPath $Expired.FullName -Force
    }
}

exit $ExitCode
