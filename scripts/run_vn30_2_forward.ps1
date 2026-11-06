# VN30-2 forward paper recorder retry (Windows task BTCVn30TwoForward, Mon-Fri 19:30 and 08:30; installed by
# scripts/install_vn30_2_forward_task.ps1). BTCVn30Screen already runs "vn30_2_forward.py record" at 16:20; when the
# backend ingest is late on a decision date that run exits 4 (data-not-ready), and these reruns record it inside
# the window that ends at 09:00 ICT of the next session. Repeated runs are idempotent (status "unchanged" or
# "not-a-decision-date"). It reads the backend over HTTP only (GET), so no secret store is loaded here.
# Exit code = the recorder's: 0 ok, 1 error, 3 missed-deadline, 4 data-not-ready.
[CmdletBinding()]
param([ValidateRange(5, 365)][int]$LogRetentionCount = 60)

$ErrorActionPreference = "Stop"
$AiRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$Python = Join-Path $AiRoot "venv/Scripts/python.exe"
$LogDirectory = Join-Path $AiRoot ".ops/vn30-2-forward"
New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
$LogPath = Join-Path $LogDirectory ("vn30-2-forward_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

# Start-Process keeps native stderr off the PowerShell error stream and gives the real exit code.
$proc = Start-Process -FilePath $Python -WorkingDirectory $AiRoot -NoNewWindow -Wait -PassThru `
    -ArgumentList @("vn30_2_forward.py", "record") -RedirectStandardOutput $LogPath -RedirectStandardError "$LogPath.stderr"
$exitCode = $proc.ExitCode
"exit=$exitCode finished=$((Get-Date).ToString('o'))" | Out-File -LiteralPath $LogPath -Encoding utf8 -Append

Get-ChildItem -LiteralPath $LogDirectory -Filter "vn30-2-forward_*.log" | Sort-Object Name -Descending |
    Select-Object -Skip $LogRetentionCount | ForEach-Object {
        Remove-Item -LiteralPath $_.FullName, "$($_.FullName).stderr" -Force -ErrorAction SilentlyContinue
    }
exit $exitCode
