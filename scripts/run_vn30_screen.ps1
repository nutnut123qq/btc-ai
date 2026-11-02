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

Get-ChildItem -LiteralPath $LogDirectory -Filter "vn30-screen_*.log" | Sort-Object Name -Descending |
    Select-Object -Skip $LogRetentionCount | ForEach-Object {
        Remove-Item -LiteralPath $_.FullName, "$($_.FullName).stderr" -Force -ErrorAction SilentlyContinue
    }
exit $proc.ExitCode
