"""SKILLS-1: the skills in ai/skills are found by agent sessions opened in the btc workspace.

Sessions read a skill through a junction (``.claude/skills/<name>``, ``.agents/skills/<name>``), so
every active skill must be valid and self-contained; ``investment-research`` is retired and never
linked; ``scripts/link_skills.ps1`` links exactly the active skills. The script is only run on a
temp folder here (Windows only), never on the real workspace. No DB, no network.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

AI_ROOT = Path(__file__).resolve().parent.parent.parent
SKILLS = AI_ROOT / "skills"
LINK_SCRIPT = AI_ROOT / "scripts" / "link_skills.ps1"
ACTIVE = {"equity-technical", "equity-position"}
LINK_DIRS = (Path(".claude") / "skills", Path(".agents") / "skills")

needs_windows = pytest.mark.skipif(
    sys.platform != "win32" or shutil.which("powershell") is None,
    reason="junctions and link_skills.ps1 are Windows-only")


def _read(name: str) -> tuple[dict, str]:
    text = (SKILLS / name / "SKILL.md").read_text(encoding="utf-8")
    m = re.match(r"---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n", text, flags=re.S)
    assert m, f"{name}/SKILL.md must start with a --- front matter block"
    return dict(re.findall(r"^(name|description):[ \t]*([^\r\n]+)", m.group(1), flags=re.M)), text


def _skill_names() -> set[str]:
    return {d.name for d in SKILLS.iterdir() if (d / "SKILL.md").is_file()}


def test_active_skills_are_the_equity_pair():
    active = {n for n in _skill_names() if not _read(n)[0].get("description", "").startswith("DEPRECATED")}
    assert active == ACTIVE


@pytest.mark.parametrize("name", sorted(ACTIVE))
def test_active_skill_front_matter_is_loadable(name):
    fields, _ = _read(name)
    assert fields.get("name", "").strip() == name
    assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) and len(name) <= 64
    assert 0 < len(fields["description"]) <= 1024


@pytest.mark.parametrize("name", sorted(ACTIVE))
def test_active_skill_is_self_contained(name):
    # Read through a junction, a relative path resolves inside the junction folder: no "../" at all.
    _, text = _read(name)
    assert not re.search(r"\.\.[/\\]", text), re.findall(r"\S*\.\.[/\\]\S*", text)
    assert not re.search(r"investment-research[/\\]", text)
    assert "use investment-research" not in text and "point to `investment-research`" not in text
    assert "This file is self-contained" in text
    assert "link_skills.ps1" in text


def test_investment_research_is_retired():
    fields, text = _read("investment-research")
    description = fields["description"]
    assert description.startswith("DEPRECATED (retired 2026-10-30, SKILLS-1)")
    assert "do not record notes" in description
    assert "equity-technical" in description and "equity-position" in description
    assert "**Retired 2026-10-30 (SKILLS-1). Do not follow this file and do not record notes with it.**" in text


def test_note_body_rules_match_the_backend_validator():
    # equity-technical carries the equity-research-note-v1 rules itself now (SKILLS-1.1).
    _, text = _read("equity-technical")
    flat = " ".join(text.split())
    for rule in ("at most 4000 characters", "at most 64 KiB", "`evidence[]` (at most 100)",
                 "`counterEvidence[]` (at most 100)", "`scenarios[]` (at most 20)", "any other key is rejected",
                 "`insufficient-input` requires `actionView: \"none\"`", "`INVALID_NOTE`"):
        assert rule in flat, rule
    validator = AI_ROOT.parent / "backend" / "Services" / "EquityResearchNoteService.cs"
    if not validator.is_file():
        pytest.skip("backend repo not next to ai/ (separate-repo CI)")
    source = validator.read_text(encoding="utf-8")
    assert "MaxBodyBytes = 64 * 1024" in source
    assert '"thesis") is not { Length: <= 4000 }' in source
    assert 'Array(b, "evidence", 100' in source and 'Array(b, "counterEvidence", 100' in source
    assert 'Array(b, "scenarios", 20' in source


def _run_link_script(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(LINK_SCRIPT), "-Root", str(root)],
        capture_output=True, text=True, timeout=120)


def _junction(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)


@pytest.fixture
def link_root(tmp_path):
    yield tmp_path
    # Remove the links themselves first, so no cleanup can ever walk into ai/skills.
    for sub in LINK_DIRS:
        folder = tmp_path / sub
        for entry in (folder.iterdir() if folder.is_dir() else ()):
            if entry.is_junction():
                os.rmdir(entry)


@needs_windows
def test_link_script_links_exactly_the_active_skills(link_root):
    first = _run_link_script(link_root)
    assert first.returncode == 0, first.stdout + first.stderr
    for sub in LINK_DIRS:
        folder = link_root / sub
        assert {p.name for p in folder.iterdir()} == ACTIVE
        for name in ACTIVE:
            link = folder / name
            assert link.is_junction(), link
            assert os.path.samefile(link, SKILLS / name)
            assert (link / "SKILL.md").read_bytes() == (SKILLS / name / "SKILL.md").read_bytes()
    second = _run_link_script(link_root)
    assert second.returncode == 0, second.stdout + second.stderr
    lines = [line for line in second.stdout.splitlines() if line.strip()]
    assert len(lines) == len(ACTIVE) * len(LINK_DIRS) and all(line.startswith("ok ") for line in lines), lines


@needs_windows
def test_link_script_removes_retired_links_and_never_touches_foreign_folders(link_root):
    claude, agents = link_root / LINK_DIRS[0], link_root / LINK_DIRS[1]
    _junction(claude / "investment-research", SKILLS / "investment-research")
    real = agents / "equity-technical"
    real.mkdir(parents=True)
    (real / "mine.txt").write_text("keep", encoding="utf-8")
    elsewhere = link_root / "elsewhere"
    elsewhere.mkdir()
    _junction(claude / "equity-position", elsewhere)
    # A user's own skill linked from elsewhere, under a name no ai/skills skill has: never removed.
    _junction(claude / "my-own-skill", elsewhere)
    # A link into ai/skills that points at the wrong skill: relinked to the right one.
    _junction(agents / "equity-position", SKILLS / "equity-technical")

    result = _run_link_script(link_root)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "removed .claude\\skills\\investment-research" in result.stdout
    assert not os.path.lexists(claude / "investment-research")
    assert (SKILLS / "investment-research" / "SKILL.md").is_file()  # only the link went
    assert "CONFLICT .agents\\skills\\equity-technical" in result.stdout
    assert not real.is_junction() and (real / "mine.txt").read_text(encoding="utf-8") == "keep"
    assert "CONFLICT .claude\\skills\\equity-position" in result.stdout
    assert os.path.samefile(claude / "equity-position", elsewhere)
    assert os.path.samefile(claude / "equity-technical", SKILLS / "equity-technical")
    assert os.path.samefile(claude / "my-own-skill", elsewhere)
    assert "my-own-skill" not in result.stdout
    assert "relinked .agents\\skills\\equity-position" in result.stdout
    assert os.path.samefile(agents / "equity-position", SKILLS / "equity-position")
