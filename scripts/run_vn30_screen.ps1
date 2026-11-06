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

Get-ChildItem -LiteralPath $LogDirectory -Filter "vn30-screen_*.log" | Sort-Object Name -Descending |
    Select-Object -Skip $LogRetentionCount | ForEach-Object {
        Remove-Item -LiteralPath $_.FullName, "$($_.FullName).stderr", "$($_.FullName).forward", "$($_.FullName).forward.stderr" -Force -ErrorAction SilentlyContinue
    }
exit $exitCode
