# VN30-SCREEN daily run (Windows task BTCVn30Screen, Mon-Fri 16:20, after the 15:15 equity ingest).
# vn30_screen.py reads the backend over HTTP only, so no secret store is loaded here (ops/common.ps1 would decrypt it).
[CmdletBinding()]
param([ValidateRange(5, 365)][int]$LogRetentionCount = 60)

$ErrorActionPreference = "Stop"
$AiRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$Python = Join-Path $AiRoot "venv/Scripts/python.exe"
$LogDirectory = Join-Path $AiRoot ".ops/vn30-screen"
New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
$LogPath = Join-Path $LogDirectory ("vn30-screen_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

# Start-Process keeps native stderr off the PowerShell error stream and gives the real exit code (#33).
$proc = Start-Process -FilePath $Python -WorkingDirectory $AiRoot -NoNewWindow -Wait -PassThru `
    -ArgumentList @("vn30_screen.py") -RedirectStandardOutput $LogPath -RedirectStandardError "$LogPath.stderr"
"exit=$($proc.ExitCode) finished=$((Get-Date).ToString('o'))" | Out-File -LiteralPath $LogPath -Encoding utf8 -Append
$exitCode = $proc.ExitCode

# Roadmap 0.7: append this session's top 10 to the immutable forward ledger (<workspace>/data/forward), only after a
# successful screen. A ledger failure exits 7 so the task shows it; the screen output itself is already written.
if ($exitCode -eq 0) {
    $fwd = Start-Process -FilePath $Python -WorkingDirectory $AiRoot -NoNewWindow -Wait -PassThru `
        -ArgumentList @("vn30_screen_forward.py", "record") -RedirectStandardOutput "$LogPath.forward" -RedirectStandardError "$LogPath.forward.stderr"
    "forward-record exit=$($fwd.ExitCode)" | Out-File -LiteralPath $LogPath -Encoding utf8 -Append
    if ($fwd.ExitCode -ne 0) { $exitCode = 7 }
}

# Roadmap 2.6/3.3: VN30-2 forward paper record (<workspace>/data/forward/vn30-2.jsonl). It reads the frozen snapshot and
# the backend, not the screen output, so it runs whatever the screen did; it writes only on a protocol decision date
# (every 20 sessions) and is a no-op otherwise. A failure exits 8 only when nothing failed before, so an earlier
# screen (non-zero) or forward-record (7) exit code is never hidden. Exit 4 (data-not-ready: the ingest of a decision
# date is late) is not a failure of this task when the retry task BTCVn30TwoForward (19:30, 08:30) is registered:
# that task records the date inside its window and shows its own result.
$v2 = Start-Process -FilePath $Python -WorkingDirectory $AiRoot -NoNewWindow -Wait -PassThru `
    -ArgumentList @("vn30_2_forward.py", "record") -RedirectStandardOutput "$LogPath.vn30-2" -RedirectStandardError "$LogPath.vn30-2.stderr"
"vn30-2-forward exit=$($v2.ExitCode)" | Out-File -LiteralPath $LogPath -Encoding utf8 -Append
$v2Code = $v2.ExitCode
if ($v2Code -eq 4 -and $null -ne (Get-ScheduledTask -TaskName "BTCVn30TwoForward" -ErrorAction SilentlyContinue)) {
    "vn30-2-forward data-not-ready: left to the retry task BTCVn30TwoForward" | Out-File -LiteralPath $LogPath -Encoding utf8 -Append
    $v2Code = 0
}
if ($v2Code -ne 0 -and $exitCode -eq 0) { $exitCode = 8 }

Get-ChildItem -LiteralPath $LogDirectory -Filter "vn30-screen_*.log" | Sort-Object Name -Descending |
    Select-Object -Skip $LogRetentionCount | ForEach-Object {
        Remove-Item -LiteralPath $_.FullName, "$($_.FullName).stderr", "$($_.FullName).forward", "$($_.FullName).forward.stderr", "$($_.FullName).vn30-2", "$($_.FullName).vn30-2.stderr" -Force -ErrorAction SilentlyContinue
    }
exit $exitCode
