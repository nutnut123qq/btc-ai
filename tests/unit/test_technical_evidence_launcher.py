"""Launcher and task installer of the descriptive evidence pipeline (PowerShell 5.1).

Both scripts only run on the Windows host, so these tests skip elsewhere.  The
launcher's -DryRun prints its plan before the ops secret store is loaded, and the
installer runs under -WhatIf, so nothing here reads secrets, touches the database,
registers a task or publishes evidence.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

AI_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = AI_ROOT / "scripts" / "run_technical_evidence_pipeline.ps1"
INSTALLER = AI_ROOT / "scripts" / "install_technical_evidence_task.ps1"
OUTPUT = (AI_ROOT / "docs" / "research" / "evidence" / "technical-descriptive").resolve()

pytestmark = pytest.mark.skipif(
    sys.platform != "win32" or shutil.which("powershell") is None,
    reason="the evidence launcher is a Windows PowerShell 5.1 script",
)


def _powershell(script: Path, *arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script), *arguments],
        capture_output=True,
        text=True,
        timeout=180,
    )


def _plan(*arguments: str) -> dict:
    result = _powershell(LAUNCHER, *arguments, "-DryRun")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_launcher_default_stays_btcusdt_with_the_legacy_status_file():
    plan = _plan()
    assert plan["symbol"] == "BTCUSDT"
    assert plan["pipelineArguments"] == [
        "run_technical_evidence_pipeline.py", "--output-dir", str(OUTPUT), "--symbol", "BTCUSDT",
    ]
    assert plan["retentionEnabled"] is False
    assert Path(plan["statusPath"]).name == "scheduler-status.json"


def test_launcher_passes_the_symbol_to_pipeline_and_retention_with_its_own_status():
    status = AI_ROOT / ".ops" / "technical-evidence" / "scheduler-status-MBB.json"
    before = status.stat().st_mtime_ns if status.exists() else None
    plan = _plan("-Symbol", "mbb", "-ApplyRetention", "-KeepSuccessfulRuns", "3")
    assert plan["symbol"] == "MBB"
    assert plan["pipelineArguments"] == [
        "run_technical_evidence_pipeline.py", "--output-dir", str(OUTPUT), "--symbol", "MBB",
    ]
    assert plan["retentionEnabled"] is True
    assert plan["retentionArguments"] == [
        "technical_evidence_retention.py", "--output-dir", str(OUTPUT), "--symbol", "MBB",
        "--keep-successful-runs", "3",
    ]
    assert Path(plan["statusPath"]) == status
    # A dry run writes nothing, not even the status file.
    assert (status.stat().st_mtime_ns if status.exists() else None) == before
    assert Path(_plan("-Symbol", "BID")["statusPath"]).name == "scheduler-status-BID.json"


def test_launcher_rejects_a_symbol_that_could_smuggle_arguments():
    result = _powershell(LAUNCHER, "-Symbol", "MBB --cutoff-ms", "-DryRun")
    assert result.returncode != 0
    assert "pipelineArguments" not in result.stdout


def test_installer_names_and_runs_a_per_symbol_task_and_keeps_the_btc_action():
    equity = _powershell(INSTALLER, "-Symbol", "MBB", "-DailyAt", "16:30", "-ApplyRetention", "-WhatIf")
    assert equity.returncode == 0, equity.stderr
    assert "Technical evidence task configured: BTCTechnicalEvidence-MBB" in equity.stdout
    action = next(line for line in equity.stdout.splitlines() if line.startswith("Action: "))
    assert action.endswith(f'-File "{LAUNCHER}" -OutputDirectory "{OUTPUT}" -Symbol MBB -ApplyRetention -KeepSuccessfulRuns 2')

    btc = _powershell(INSTALLER, "-WhatIf")
    assert btc.returncode == 0, btc.stderr
    assert "Technical evidence task configured: BTCTechnicalDescriptiveEvidence" in btc.stdout
    action = next(line for line in btc.stdout.splitlines() if line.startswith("Action: "))
    assert action.endswith(f'-File "{LAUNCHER}" -OutputDirectory "{OUTPUT}"')
