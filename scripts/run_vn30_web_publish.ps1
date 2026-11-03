# VN30-WEB nightly publish (Windows task BTCVn30WebPublish, Mon-Fri 19:00, after the 30-ticker analysis run).
# export_vn30_web.py reads the backend over HTTP (GET only) into web/data; when data/ changed, commit and push the web
# repo. No ops/common.ps1 here (it decrypts the secret store, #44): the only secret is EQX_ADMIN_KEY from the User
# environment, passed to the exporter through the process environment and never written out.
[CmdletBinding()]
param([ValidateRange(5, 365)][int]$LogRetentionCount = 60)

$ErrorActionPreference = "Stop"
$AiRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$WebRoot = [IO.Path]::GetFullPath((Join-Path $AiRoot "../web"))
$Python = Join-Path $AiRoot "venv/Scripts/python.exe"
$LogDirectory = Join-Path $AiRoot ".ops/vn30-web"
New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
$LogPath = Join-Path $LogDirectory ("vn30-web_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

function Write-Log([string]$Text) { $Text | Out-File -LiteralPath $LogPath -Encoding utf8 -Append }

# Start-Process keeps native stderr off the PowerShell error stream and gives the real exit code (#33).
function Invoke-Logged([string]$FilePath, [string[]]$Arguments, [string]$WorkingDirectory) {
    $part = "$LogPath.part"
    $proc = Start-Process -FilePath $FilePath -ArgumentList $Arguments -WorkingDirectory $WorkingDirectory `
        -NoNewWindow -Wait -PassThru -RedirectStandardOutput $part -RedirectStandardError "$part.err"
    foreach ($f in @($part, "$part.err")) {
        if (Test-Path -LiteralPath $f) {
            Get-Content -LiteralPath $f -Encoding utf8 | Out-File -LiteralPath $LogPath -Encoding utf8 -Append
            Remove-Item -LiteralPath $f -Force
        }
    }
    Write-Log "$([IO.Path]::GetFileName($FilePath)) $($Arguments -join ' ') => exit=$($proc.ExitCode)"
    return $proc.ExitCode
}

function Remove-OldLogs {
    Get-ChildItem -LiteralPath $LogDirectory -Filter "vn30-web_*.log" | Sort-Object Name -Descending |
        Select-Object -Skip $LogRetentionCount | ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }
}

function Exit-With([int]$Code) {
    Write-Log "exit=$Code finished=$((Get-Date).ToString('o'))"
    Remove-OldLogs
    exit $Code
}

$env:EQX_ADMIN_KEY = [Environment]::GetEnvironmentVariable("EQX_ADMIN_KEY", "User")
if ([string]::IsNullOrEmpty($env:EQX_ADMIN_KEY)) { Write-Log "EQX_ADMIN_KEY not set for the user: notes will be unavailable" }

$code = Invoke-Logged $Python @("export_vn30_web.py", "--out-dir", "`"$(Join-Path $WebRoot 'data')`"") $AiRoot
Remove-Item -LiteralPath Env:EQX_ADMIN_KEY -ErrorAction SilentlyContinue  # git does not need it
if ($code -ne 0) { Exit-With $code }

$code = Invoke-Logged "git" @("add", "data") $WebRoot
if ($code -ne 0) { Exit-With $code }
$code = Invoke-Logged "git" @("diff", "--cached", "--quiet", "--", "data") $WebRoot
if ($code -eq 0) { Write-Log "data unchanged: nothing to commit"; Exit-With 0 }
if ($code -ne 1) { Exit-With $code }

$session = (Get-Content -LiteralPath (Join-Path $WebRoot "data/meta.json") -Raw -Encoding utf8 | ConvertFrom-Json).session
$messageFile = "$LogPath.msg"
# the e-circumflex of "phien" comes from its code point: Windows PowerShell 5.1 reads a BOM-less script as ANSI
[IO.File]::WriteAllText($messageFile, "data: phi$([char]0x00EA)n $session`n", (New-Object Text.UTF8Encoding $false))
# commit with a pathspec commits only data/, whatever else is staged in the web repo
$code = Invoke-Logged "git" @("commit", "-F", "`"$messageFile`"", "--", "data") $WebRoot
Remove-Item -LiteralPath $messageFile -Force
if ($code -ne 0) { Exit-With $code }
$code = Invoke-Logged "git" @("push", "origin", "main") $WebRoot
Exit-With $code
