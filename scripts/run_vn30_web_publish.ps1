# VN30-WEB nightly publish: export the public site's data into the web repo, commit data/ and push main.
#
# Windows task BTCVn30WebPublish (Mon-Fri 19:00, after the 30-ticker analysis run; principal InteractiveToken;
# definition in evidence/vn30-web-20261103/task-BTCVn30WebPublish.xml) runs it with no arguments:
#   powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File <this file>
# Every parameter default is what that task needs; the other values exist for tests (see Parameters).
#
# A run, in order:
#   1. Takes the named mutex -LockName; a run started while another one holds it exits 5.
#   2. Reads EQX_ADMIN_KEY from the User environment; when it is empty, exits 4 before exporting (the notes part
#      cannot be read without it).
#   3. Checks the web repo before exporting; any failure exits 3 with nothing exported or committed:
#      - -WebRoot is the top of a git work tree;
#      - HEAD is the branch main, main has a commit, and no merge, cherry-pick, revert or rebase is in progress;
#      - user.email is set in the repo's own config and git's author and committer identities use it: the identity
#        shown on the public repo is a deliberate choice, never inherited from the global config or the environment;
#      - git fetch origin main works, and local main is neither behind nor diverged from origin/main;
#      - commits ahead of origin/main (left by a push that failed) are not merges, carry that identity and touch
#        data/ only.
#   4. Exports: python export_vn30_web.py --out-dir <web>/data, with the key in the exporter's environment only and
#      PYTHONIOENCODING=utf-8 so that its Vietnamese messages reach the log intact.
#   5. Stages data/*.json (never a leftover .tmp) and, when that changed, commits those paths alone, whatever else is
#      staged, as "data: phien <session of data/meta.json>" (with e-circumflex).
#   6. Pushes HEAD:refs/heads/main whenever main is ahead of origin/main, so that a later run retries a failed push
#      even when its own data did not change.
#
# Exit codes:
#   0      done: pushed, or nothing to publish
#   1      export: backend error (exit code of export_vn30_web.py)
#   2      export: VN30-SCREEN table missing or invalid (exit code of export_vn30_web.py)
#   3      refused by a check of step 3: nothing exported, nothing committed
#   4      EQX_ADMIN_KEY is not set for the user: nothing exported
#   5      another run holds the lock
#   6      unexpected error of this script, or an export exit code other than 0, 1 and 2 (details in the log)
#   other  a git command of steps 5-6 failed: git's own exit code, plus 10 when that code is 1-9 so that it never
#          reads as one of the codes above (a rejected push, git 1, exits 11; a fatal error, git 128, exits 128)
#
# Log: ai/.ops/vn30-web/vn30-web_<yyyyMMdd_HHmmss>_<pid>.log (UTF-8), one per run. Each command is logged before it
# starts and its output after it ends; the last line is "exit=<code> finished=<time>" unless the process itself is
# killed. The newest -LogRetentionCount logs are kept, and .part/.part.err/.msg files left by a killed run are removed.
# Secrets: no ops/common.ps1 here (it decrypts the secret store, #44). The only secret, EQX_ADMIN_KEY, is put in the
# environment of the exporter alone, removed before any git command and never written out.
# SSH: unless GIT_SSH_COMMAND, GIT_SSH or core.sshCommand is already set (then it is left as it is), git runs
# "ssh -o BatchMode=yes -o ConnectTimeout=30", so that a host-key or passphrase question fails at once instead of
# waiting for the task's time limit.
#
# Parameters:
#   -LogRetentionCount <n>   logs kept (default 60)
#   -WebRoot <dir>           web repo (default <ai>/../web)
#   -Python <exe>            interpreter of the exporter (default <ai>/venv/Scripts/python.exe)
#   -LockName <name>         named mutex (default Global\BTCVn30WebPublish)
#   -TestUseProcessAdminKey  TESTS ONLY: take EQX_ADMIN_KEY from this process's environment instead of the User
#                            scope; the real task never passes it
[CmdletBinding()]
param(
    [ValidateRange(5, 365)][int]$LogRetentionCount = 60,
    [string]$WebRoot = "",
    [string]$Python = "",
    [string]$LockName = "Global\BTCVn30WebPublish",
    [switch]$TestUseProcessAdminKey
)

$ErrorActionPreference = "Stop"
$AiRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$WebRoot = [IO.Path]::GetFullPath($(if ($WebRoot) { $WebRoot } else { Join-Path $AiRoot "../web" }))
if (-not $Python) { $Python = [IO.Path]::GetFullPath((Join-Path $AiRoot "venv/Scripts/python.exe")) }
$LogDirectory = Join-Path $AiRoot ".ops/vn30-web"
New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
$LogPath = Join-Path $LogDirectory ("vn30-web_{0}_{1}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"), $PID)
$Utf8 = New-Object Text.UTF8Encoding $false
$DataSpec = "data/*.json"  # a git pathspec: * also matches /, so data/symbols/*.json too, but never a leftover .tmp
$Tracking = "refs/remotes/origin/main"
$SshCommand = "ssh -o BatchMode=yes -o ConnectTimeout=30"

function Write-Log([string]$Text) {
    try { [IO.File]::AppendAllText($LogPath, $Text + "`r`n", $Utf8) } catch { }  # logging never ends a run
}

# Start-Process joins -ArgumentList with spaces as is: quote by the Windows rules (CommandLineToArgvW).
function ConvertTo-NativeArgument([string]$Value) {
    if ($Value -and $Value -notmatch '[\s"]') { return $Value }
    return '"' + (($Value -replace '(\\*)"', '$1$1\"') -replace '(\\+)$', '$1$1') + '"'
}

function Read-Captured([string]$Path) {
    if (Test-Path -LiteralPath $Path) {
        Get-Content -LiteralPath $Path -Encoding UTF8
        Remove-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue  # a later run removes what is left
    }
}

# Start-Process keeps native stderr off the PowerShell error stream and gives the real exit code (#33). WaitForExit
# rather than -Wait: in Windows PowerShell 5.1, -Wait polls once a second, so every git call took a second.
function Invoke-Native([string]$FilePath, [string[]]$Arguments, [string]$WorkingDirectory) {
    $part = "$LogPath.part"
    $started = Get-Date
    Write-Log ("[{0}] > {1}" -f $started.ToString("HH:mm:ss"), ((@([IO.Path]::GetFileName($FilePath)) + $Arguments) -join " "))
    $proc = Start-Process -FilePath $FilePath -ArgumentList @($Arguments | ForEach-Object { ConvertTo-NativeArgument $_ }) `
        -WorkingDirectory $WorkingDirectory -NoNewWindow -PassThru -RedirectStandardOutput $part `
        -RedirectStandardError "$part.err"
    $null = $proc.Handle  # Windows PowerShell 5.1: without a cached handle, ExitCode can stay empty
    $proc.WaitForExit()
    $stdout = @(Read-Captured $part)
    foreach ($line in $stdout + @(Read-Captured "$part.err")) { Write-Log "  $line" }
    Write-Log "  => exit=$($proc.ExitCode) ($([math]::Round(((Get-Date) - $started).TotalSeconds, 1)) s)"
    New-Object PSObject -Property @{ Code = $proc.ExitCode; Out = $stdout }
}

function Invoke-Git([string[]]$Arguments) { Invoke-Native "git" $Arguments $WebRoot }

# The number printed by git rev-list --count, or -1 when git failed.
function Get-GitCount([string[]]$Arguments) {
    $r = Invoke-Git $Arguments
    $count = 0
    if ($r.Code -eq 0 -and $r.Out.Count -eq 1 -and [int]::TryParse($r.Out[0], [ref]$count)) { return $count }
    return -1
}

# A failed git command's exit code; 1-9 are lifted by 10 so that they never read as one of this script's own codes.
function Get-GitExitCode([int]$Code) {
    if ($Code -ge 1 -and $Code -le 9) { return $Code + 10 }
    return $Code
}

# git must never wait for an answer: BatchMode makes ssh fail on a host-key or passphrase question. A transport the
# user has chosen (GIT_SSH_COMMAND, GIT_SSH or core.sshCommand) is left as it is.
function Set-GitSsh {
    if ($env:GIT_SSH_COMMAND -or $env:GIT_SSH) { Write-Log "ssh: GIT_SSH_COMMAND or GIT_SSH set by the user, left as it is"; return }
    if ((Invoke-Git @("config", "--get", "core.sshCommand")).Code -eq 0) { Write-Log "ssh: core.sshCommand set, left as it is"; return }
    $env:GIT_SSH_COMMAND = $SshCommand
    Write-Log "ssh: GIT_SSH_COMMAND=$SshCommand"
}

# Why the web repo is not safe to export into, commit in and push from, or $null when it is.
function Get-WebRepoProblem {
    if (-not (Test-Path -LiteralPath $WebRoot -PathType Container)) { return "web folder not found: $WebRoot" }
    $r = Invoke-Git @("rev-parse", "--show-prefix", "--absolute-git-dir")
    if ($r.Code -ne 0 -or $r.Out.Count -ne 2) { return "not a git work tree: $WebRoot" }
    if ($r.Out[0]) { return "$WebRoot is a folder inside the work tree of a repo, not the top of the web repo" }
    $gitDir = $r.Out[1]
    $r = Invoke-Git @("symbolic-ref", "--quiet", "--short", "HEAD")
    if ($r.Code -ne 0) { return "HEAD is detached, not on branch main" }
    if ($r.Out[0] -ne "main") { return "on branch '$($r.Out[0])', not main" }
    if ((Invoke-Git @("rev-parse", "--verify", "--quiet", "HEAD")).Code -ne 0) {
        return "branch main has no commit yet: make the first commit by hand"
    }
    foreach ($state in "MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "rebase-merge", "rebase-apply") {
        if (Test-Path -LiteralPath (Join-Path $gitDir $state)) {
            return "a merge, cherry-pick, revert or rebase is in progress ($state)"
        }
    }
    $r = Invoke-Git @("config", "--local", "--get", "user.email")
    $email = if ($r.Code -eq 0 -and $r.Out.Count -ge 1) { $r.Out[0].Trim() } else { "" }
    if (-not $email) {
        return "the web repo has no user.email of its own (git config user.email <address> in it): the identity shown on the public repo is chosen for it, never inherited from the global config"
    }
    foreach ($ident in "GIT_AUTHOR_IDENT", "GIT_COMMITTER_IDENT") {
        $r = Invoke-Git @("var", $ident)
        if ($r.Code -ne 0 -or $r.Out.Count -lt 1 -or -not $r.Out[0].Contains("<$email>")) {
            return "$ident is not the repo's user.email: an environment variable or a command-line setting overrides it"
        }
    }
    Set-GitSsh
    $r = Invoke-Git @("fetch", "--no-tags", "origin", "+refs/heads/main:$Tracking")
    if ($r.Code -ne 0) { return "git fetch origin main failed (exit $($r.Code))" }
    $behind = Get-GitCount @("rev-list", "--count", "HEAD..$Tracking")
    $ahead = Get-GitCount @("rev-list", "--count", "$Tracking..HEAD")
    if ($behind -lt 0 -or $ahead -lt 0) { return "cannot compare main with origin/main" }
    if ($behind -gt 0 -and $ahead -gt 0) { return "main has diverged from origin/main ($ahead ahead, $behind behind)" }
    if ($behind -gt 0) { return "main is $behind commit(s) behind origin/main" }
    if ($ahead -eq 0) { return $null }
    # Commits left by a push that failed go out with the next push: only data commits of this identity may.
    if ((Get-GitCount @("rev-list", "--count", "--min-parents=2", "$Tracking..HEAD")) -ne 0) {
        return "a merge commit is among the $ahead commit(s) ahead of origin/main"
    }
    $r = Invoke-Git @("log", "--no-renames", "--name-only", "--format=", "$Tracking..HEAD")
    $outside = @($r.Out | Where-Object { $_ -and -not $_.StartsWith("data/", [StringComparison]::Ordinal) })
    if ($r.Code -ne 0 -or $outside.Count -gt 0) {
        return "the $ahead commit(s) ahead of origin/main touch files outside data/: $($outside -join ', ')"
    }
    $r = Invoke-Git @("log", "--format=%ae%n%ce", "$Tracking..HEAD")
    if ($r.Code -ne 0 -or @($r.Out | Where-Object { $_ -and $_ -cne $email }).Count -gt 0) {
        return "the $ahead commit(s) ahead of origin/main carry another identity than the repo's user.email"
    }
    return $null
}

# Steps 2-6; returns the exit code.
function Invoke-Publish {
    $key = if ($TestUseProcessAdminKey) { $env:EQX_ADMIN_KEY } else { [Environment]::GetEnvironmentVariable("EQX_ADMIN_KEY", "User") }
    Remove-Item -LiteralPath Env:EQX_ADMIN_KEY -ErrorAction SilentlyContinue  # the exporter alone gets it, in step 4
    if ([string]::IsNullOrWhiteSpace($key)) {
        Write-Log "refused: EQX_ADMIN_KEY is not set for the user, so the notes part cannot be read; nothing exported"
        return 4
    }
    $problem = Get-WebRepoProblem
    if ($problem) { Write-Log "refused: $problem; nothing exported or committed"; return 3 }

    $exporter = Join-Path $AiRoot "export_vn30_web.py"
    if (-not (Test-Path -LiteralPath $exporter -PathType Leaf)) { throw "exporter not found: $exporter" }
    $env:PYTHONIOENCODING = "utf-8"
    $env:EQX_ADMIN_KEY = $key
    try { $r = Invoke-Native $Python @("export_vn30_web.py", "--out-dir", (Join-Path $WebRoot "data")) $AiRoot }
    finally { Remove-Item -LiteralPath Env:EQX_ADMIN_KEY -ErrorAction SilentlyContinue }
    if ($r.Code -eq 1 -or $r.Code -eq 2) { return $r.Code }
    if ($r.Code -ne 0) { Write-Log "the exporter ended with exit code $($r.Code), which it never returns by itself"; return 6 }

    $r = Invoke-Git @("add", "--all", "--", $DataSpec)
    if ($r.Code -ne 0) { return (Get-GitExitCode $r.Code) }
    $r = Invoke-Git @("diff", "--cached", "--quiet", "--", $DataSpec)
    if ($r.Code -eq 1) {
        $session = (Get-Content -LiteralPath (Join-Path $WebRoot "data/meta.json") -Raw -Encoding UTF8 | ConvertFrom-Json).session
        if ("$session" -notmatch '^[0-9]{4}-[0-9]{2}-[0-9]{2}\z') { throw "data/meta.json has no session date" }
        $messageFile = "$LogPath.msg"
        # the e-circumflex of "phien" comes from its code point: Windows PowerShell 5.1 reads a BOM-less script as ANSI
        [IO.File]::WriteAllText($messageFile, "data: phi$([char]0x00EA)n $session`n", $Utf8)
        # a commit with a pathspec takes those paths only, whatever else is staged in the web repo
        $r = Invoke-Git @("commit", "--file", $messageFile, "--", $DataSpec)
        Remove-Item -LiteralPath $messageFile -Force
        if ($r.Code -ne 0) { return (Get-GitExitCode $r.Code) }
    } elseif ($r.Code -eq 0) {
        Write-Log "data unchanged: nothing to commit"
    } else {
        return (Get-GitExitCode $r.Code)
    }

    $ahead = Get-GitCount @("rev-list", "--count", "$Tracking..HEAD")
    if ($ahead -lt 0) { throw "cannot count the commits ahead of origin/main" }
    if ($ahead -eq 0) { Write-Log "nothing to push: main is at origin/main"; return 0 }
    $r = Invoke-Git @("push", "--no-follow-tags", "origin", "HEAD:refs/heads/main")
    if ($r.Code -ne 0) { return (Get-GitExitCode $r.Code) }
    Write-Log "pushed $ahead commit(s) to origin/main"
    return 0
}

# .part/.part.err/.msg files of a run that was killed (time limit, power loss) before its finally block ran.
function Remove-Leftovers {
    foreach ($file in @(Get-ChildItem -LiteralPath $LogDirectory -File)) {
        if ($file.Name -notlike "vn30-web_*.log.part*" -and $file.Name -notlike "vn30-web_*.log.msg") { continue }
        try { Remove-Item -LiteralPath $file.FullName -Force; Write-Log "removed leftover $($file.Name)" }
        catch { Write-Log "could not remove leftover $($file.Name): $($_.Exception.Message)" }
    }
}

function Remove-OldLogs {
    try {
        @(Get-ChildItem -LiteralPath $LogDirectory -File | Where-Object { $_.Name -like "vn30-web_*.log" } |
            Sort-Object Name -Descending | Select-Object -Skip $LogRetentionCount) |
            ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }
    } catch { }  # housekeeping never changes the exit code
}

$exitCode = 6
$mutex = $null
$hasLock = $false
try {
    Write-Log "started $((Get-Date).ToString('o')) pid=$PID web=$WebRoot python=$Python"
    try { $mutex = [Threading.Mutex]::new($false, $LockName) }
    catch [System.UnauthorizedAccessException] { $mutex = $null }  # it exists, made by another user or elevation level
    if ($mutex) {
        try { $hasLock = $mutex.WaitOne(0) }
        catch [System.Threading.AbandonedMutexException] { $hasLock = $true; Write-Log "the lock's last holder ended without releasing it" }
    }
    if ($hasLock) {
        Remove-Leftovers
        $exitCode = Invoke-Publish
        if ($exitCode -isnot [int]) { throw "internal error: the publish steps returned '$exitCode'" }
    } elseif ($mutex) {
        Write-Log "another run holds the lock $LockName"
        $exitCode = 5
    } else {
        Write-Log "the lock $LockName exists but this process may not open it: a run of another user or elevation level holds it"
        $exitCode = 5
    }
} catch {
    Write-Log "ERROR: $($_.Exception.GetType().Name): $($_.Exception.Message) (script line $($_.InvocationInfo.ScriptLineNumber))"
    $exitCode = 6
} finally {
    foreach ($file in "$LogPath.part", "$LogPath.part.err", "$LogPath.msg") {
        if (Test-Path -LiteralPath $file) { Remove-Item -LiteralPath $file -Force -ErrorAction SilentlyContinue }
    }
    Remove-Item -LiteralPath Env:EQX_ADMIN_KEY -ErrorAction SilentlyContinue
    if ($hasLock) { Remove-OldLogs }
    Write-Log "exit=$exitCode finished=$((Get-Date).ToString('o'))"
    if ($mutex) {
        if ($hasLock) { try { $mutex.ReleaseMutex() } catch { } }
        $mutex.Dispose()
    }
}
exit $exitCode
