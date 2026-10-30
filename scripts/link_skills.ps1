<#
.SYNOPSIS
  Makes the skills in ai\skills discoverable by agent sessions opened in the btc workspace (SKILLS-1).

.DESCRIPTION
  For every ai\skills\<name>\SKILL.md whose front-matter description does not start with DEPRECATED,
  creates the junctions <Root>\.claude\skills\<name> (Claude Code) and <Root>\.agents\skills\<name>
  (Codex) pointing to ai\skills\<name>. Junctions in those folders that point into ai\skills but whose
  skill is deprecated or gone are removed (the link only, never the target).
  Never deletes a real folder or a link that points outside ai\skills: either one under a skill's
  name is reported and the script exits 1. Idempotent; junctions need no admin rights.
  Windows PowerShell 5.1 or PowerShell 7. Keep this file ASCII (5.1 reads BOM-less files as ANSI).

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File D:\code\btc\ai\scripts\link_skills.ps1
#>
param([string]$Root = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)))

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path -LiteralPath $Root).Path
$skillsDir = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\skills')).Path.TrimEnd('\')

function Get-FullPath([string]$Path) { [IO.Path]::GetFullPath($Path).TrimEnd('\') }

$active = @{}
foreach ($dir in Get-ChildItem -LiteralPath $skillsDir -Directory) {
    $skillFile = Join-Path $dir.FullName 'SKILL.md'
    if (-not (Test-Path -LiteralPath $skillFile -PathType Leaf)) { continue }
    $text = [IO.File]::ReadAllText($skillFile)
    $front = if ($text -match '\A---\r?\n([\s\S]*?)\r?\n---') { $Matches[1] } else { '' }
    if ($front -match '(?m)^description:[ \t]*DEPRECATED') { continue }
    $active[$dir.Name] = Get-FullPath $dir.FullName
}

$failed = $false
foreach ($sub in '.claude\skills', '.agents\skills') {
    $linkDir = Join-Path $Root $sub
    New-Item -ItemType Directory -Force -Path $linkDir | Out-Null

    foreach ($item in @(Get-ChildItem -LiteralPath $linkDir -Force)) {
        $raw = [string]@($item.Target)[0]
        if (-not ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -or -not $raw) { continue }
        $target = Get-FullPath $raw
        $intoSkills = $target.StartsWith($skillsDir + '\', [StringComparison]::OrdinalIgnoreCase)
        if ($intoSkills -and -not $active.ContainsKey($item.Name)) {
            [IO.Directory]::Delete($item.FullName)  # removes the junction itself, not the target
            Write-Output "removed $sub\$($item.Name) (skill deprecated or gone)"
        }
    }

    foreach ($name in @($active.Keys | Sort-Object)) {
        $link = Join-Path $linkDir $name
        $source = $active[$name]
        $existing = Get-Item -LiteralPath $link -Force -ErrorAction SilentlyContinue
        if ($existing) {
            if (-not ($existing.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
                Write-Output "CONFLICT $sub\$name is a real folder or file; left untouched"
                $failed = $true
                continue
            }
            $raw = [string]@($existing.Target)[0]
            $target = if ($raw) { Get-FullPath $raw } else { '(unknown target)' }
            if ([string]::Equals($target, $source, [StringComparison]::OrdinalIgnoreCase)) {
                Write-Output "ok $sub\$name"
                continue
            }
            if (-not $target.StartsWith($skillsDir + '\', [StringComparison]::OrdinalIgnoreCase)) {
                Write-Output "CONFLICT $sub\$name links outside ai\skills ($target); left untouched"
                $failed = $true
                continue
            }
            [IO.Directory]::Delete($existing.FullName)
            Write-Output "relinked $sub\$name (was -> $target)"
        }
        New-Item -ItemType Junction -Path $link -Target $source | Out-Null
        Write-Output "linked $sub\$name -> $source"
    }
}
if ($failed) { exit 1 }
