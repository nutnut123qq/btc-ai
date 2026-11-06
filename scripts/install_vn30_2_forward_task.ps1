# Registers (or audits) the Windows task BTCVn30TwoForward: reruns of "vn30_2_forward.py record" at 19:30 and
# 08:30, Mon-Fri, so a decision date whose backend ingest was late at 16:20 (BTCVn30Screen, exit 4 data-not-ready)
# is still recorded before the next session opens (09:00 ICT). Same principal style as BTCVn30Screen (current user,
# interactive, limited). Run it from the checkout the task must use (normally <workspace>\btc-ai), after the code is
# merged there. -WhatIf shows what would be registered; -AuditOnly checks an existing task and changes nothing.
[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$TaskName = "BTCVn30TwoForward",
    [string[]]$DailyAt = @("19:30", "08:30"),
    [switch]$AuditOnly
)

$ErrorActionPreference = "Stop"
$AiRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$Runner = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot "run_vn30_2_forward.ps1"))
if (-not (Test-Path -LiteralPath $Runner)) { throw "Runner not found: $Runner" }
$Python = Join-Path $AiRoot "venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw "AI virtual-environment Python not found: $Python" }

Push-Location $AiRoot
try {
    & $Python -c "import vn30_2_forward"
    if ($LASTEXITCODE -ne 0) { throw "vn30_2_forward.py does not import in $AiRoot" }
}
finally { Pop-Location }

$PowerShell = Join-Path $env:SystemRoot "System32/WindowsPowerShell/v1.0/powershell.exe"
$Arguments = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$Runner`""
$Days = @("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")

if ($AuditOnly) {
    $Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
    $Actions = @($Task.Actions)
    if ($Actions.Count -ne 1 -or $Actions[0].Execute -ne $PowerShell -or $Actions[0].Arguments -ne $Arguments `
        -or $Actions[0].WorkingDirectory -ne $AiRoot) {
        throw "Task $TaskName action differs from the expected runner $Runner"
    }
    $Times = @($Task.Triggers | ForEach-Object { ([DateTime]$_.StartBoundary).ToString("HH:mm") }) | Sort-Object
    $Want = @($DailyAt | ForEach-Object { [DateTime]::ParseExact($_, "HH:mm", [Globalization.CultureInfo]::InvariantCulture).ToString("HH:mm") }) | Sort-Object
    if (($Times -join ",") -ne ($Want -join ",")) { throw "Task $TaskName triggers $($Times -join ',') differ from $($Want -join ',')" }
    "Task $TaskName OK: $Runner at $($Times -join ', ') Mon-Fri"
    exit 0
}

$Triggers = @($DailyAt | ForEach-Object {
    $At = [DateTime]::ParseExact($_, "HH:mm", [Globalization.CultureInfo]::InvariantCulture)
    New-ScheduledTaskTrigger -Weekly -DaysOfWeek $Days -At $At
})
$Action = New-ScheduledTaskAction -Execute $PowerShell -Argument $Arguments -WorkingDirectory $AiRoot
$Settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 30) -StartWhenAvailable `
    -MultipleInstances IgnoreNew
$Principal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive -RunLevel Limited
if ($PSCmdlet.ShouldProcess($TaskName, "Register task running $Runner at $($DailyAt -join ', ') Mon-Fri")) {
    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Triggers -Settings $Settings `
        -Principal $Principal -Description "VN30-2 forward paper recorder retry (record only, GET-only backend reads)" `
        -Force | Out-Null
    "Registered $TaskName"
}
