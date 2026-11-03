"""VN30-WEB nightly publish wrapper (scripts/run_vn30_web_publish.ps1) under Windows PowerShell 5.1.

Each test builds a sandbox in tmp_path, in a folder whose name has a space and parentheses: a copy of the wrapper in
<sandbox>/ai/scripts, a fake <sandbox>/ai/export_vn30_web.py that follows fake_export.json next to it and never reads
the environment, a web repo on branch main whose origin is a local bare repo, git hooks that record what git commands
see, and a global git config with another identity. The wrapper runs with -TestUseProcessAdminKey (it never reads the
User environment), a lock name of its own and an environment built here: none of this machine's EQX_ADMIN_KEY, GIT_*
or PYTHONIOENCODING/PYTHONUTF8 variables, a fake key, GIT_CONFIG_GLOBAL on the sandbox config and no system config.
Nothing here touches the real web repo, GitHub or the backend.
"""
import contextlib
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest

AI_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = AI_ROOT / "scripts" / "run_vn30_web_publish.ps1"
POWERSHELL = shutil.which("powershell.exe")
GIT = shutil.which("git")

pytestmark = pytest.mark.skipif(
    os.name != "nt" or POWERSHELL is None or GIT is None,
    reason="the publish wrapper is a Windows PowerShell 5.1 script driving git",
)

OLD_SESSION = "2026-11-03"
SESSION = "2026-11-04"
REPO_EMAIL = "12345678+tester@users.noreply.github.com"
GLOBAL_EMAIL = "global.identity@example.com"
FAKE_KEY = "fake-admin-key-for-tests"
DEFAULT_SSH = "ssh -o BatchMode=yes -o ConnectTimeout=30"
MESSAGE = f"data: phi\u00ean {SESSION}\n"
STDERR_VI = "lỗi backend: giữ nguyên file phiên trước"  # outside cp1258 single bytes: garbled unless UTF-8
DROPPED_ENV = {"EQX_ADMIN_KEY", "PYTHONIOENCODING", "PYTHONUTF8", "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "EMAIL"}

FAKE_EXPORT = '''"""Fake export_vn30_web.py of the publish wrapper tests: follows fake_export.json next to this file and reads
no environment variable."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
scenario = json.loads((HERE / "fake_export.json").read_text(encoding="utf-8"))
out_dir = Path(sys.argv[sys.argv.index("--out-dir") + 1])
with open(HERE / "export-runs.jsonl", "a", encoding="utf-8") as runs:
    print(json.dumps({"argv": sys.argv[1:], "cwd": str(Path.cwd())}), file=runs)
for rel, text in scenario["write"].items():
    target = out_dir / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(text.encode("utf-8"))
if scenario["stdout"]:
    print(scenario["stdout"])
if scenario["stderr"]:
    print(scenario["stderr"], file=sys.stderr)
sys.exit(scenario["exit"])
'''

# Hooks run from the top of the work tree (web/), so ../hook-records.txt is <sandbox>/hook-records.txt. They record
# whether the admin key is set (never its value) and the ssh command git would use.
PRE_COMMIT = """#!/bin/sh
if [ -n "${EQX_ADMIN_KEY+x}" ]; then key=present; else key=absent; fi
echo "pre-commit key=$key" >> ../hook-records.txt
"""
PRE_PUSH = """#!/bin/sh
if [ -n "${EQX_ADMIN_KEY+x}" ]; then key=present; else key=absent; fi
echo "pre-push key=$key ssh=${GIT_SSH_COMMAND-<unset>}" >> ../hook-records.txt
cat > /dev/null
"""
# The bare remote's hooks run in remote.git/: a reject-push file there turns every push down.
PRE_RECEIVE = """#!/bin/sh
cat > /dev/null
if [ -f reject-push ]; then echo "push rejected by the test remote" >&2; exit 1; fi
exit 0
"""


@dataclass
class Run:
    code: int
    log: str


class Sandbox:
    def __init__(self, root: Path):
        self.root = root
        self.ai = root / "ai"
        self.web = root / "web"
        self.remote = root / "remote.git"
        self.logs = self.ai / ".ops" / "vn30-web"
        self.records = root / "hook-records.txt"
        self.global_config = root / "global.gitconfig"
        self.lock = "Global\\BTCVn30WebPublish-test-" + uuid.uuid4().hex
        self.env = {k: v for k, v in os.environ.items()
                    if not k.upper().startswith("GIT_") and k.upper() not in DROPPED_ENV}
        self.env.update(GIT_CONFIG_GLOBAL=str(self.global_config), GIT_CONFIG_NOSYSTEM="1",
                        GIT_CEILING_DIRECTORIES=str(root.parent))

    def git(self, cwd: Path, *args: str) -> str:
        r = subprocess.run([GIT, *args], cwd=cwd, env=self.env, capture_output=True, timeout=60)
        assert r.returncode == 0, (args, r.stderr.decode("utf-8", "replace"))
        return r.stdout.decode("utf-8")

    @staticmethod
    def data(session: str, close: int = 20000) -> dict[str, str]:
        return {
            "data/meta.json": json.dumps({"schema": "vn30-web-data/v2", "session": session, "symbols": ["MBB"]}) + "\n",
            "data/screen.json": json.dumps({"session": session, "rows": []}) + "\n",
            "data/symbols/MBB.json": json.dumps({"symbol": "MBB", "candles": [[session, 1, 2, 0.5, close, 10]]}) + "\n",
        }

    def write_web(self, files: dict[str, str]) -> None:
        for rel, text in files.items():
            path = self.web / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="\n")

    def commit_web(self, files: dict[str, str], message: str, *config: str) -> None:
        self.write_web(files)
        self.git(self.web, "add", "--all")
        self.git(self.web, *config, "commit", "--quiet", "-m", message)

    def build(self) -> "Sandbox":
        (self.ai / "scripts").mkdir(parents=True)
        shutil.copyfile(WRAPPER, self.ai / "scripts" / WRAPPER.name)
        (self.ai / "export_vn30_web.py").write_text(FAKE_EXPORT, encoding="utf-8", newline="\n")
        self.scenario()
        self.global_config.write_text(f"[user]\n\tname = Global Identity\n\temail = {GLOBAL_EMAIL}\n",
                                      encoding="utf-8", newline="\n")
        self.git(self.root, "init", "--quiet", "--bare", "--initial-branch=main", str(self.remote))
        self.git(self.root, "init", "--quiet", "--initial-branch=main", str(self.web))
        self.git(self.web, "config", "user.name", "Repo Identity")
        self.git(self.web, "config", "user.email", REPO_EMAIL)
        self.commit_web({".gitattributes": "* text=auto eol=lf\n", "app/page.tsx": "export default 1;\n",
                         **self.data(OLD_SESSION)}, "site: first commit")
        self.git(self.web, "remote", "add", "origin", str(self.remote))
        self.git(self.web, "push", "--quiet", "--set-upstream", "origin", "main")
        for hooks, name, text in ((self.web / ".git" / "hooks", "pre-commit", PRE_COMMIT),
                                  (self.web / ".git" / "hooks", "pre-push", PRE_PUSH),
                                  (self.remote / "hooks", "pre-receive", PRE_RECEIVE)):
            (hooks / name).write_text(text, encoding="utf-8", newline="\n")
        return self

    def scenario(self, *, exit: int = 0, write: dict[str, str] | None = None, stdout: str = "", stderr: str = "") -> None:
        """What the next exports do; `write` maps paths of the web repo (data/...) to their new text."""
        assert all(rel.startswith("data/") for rel in write or {})
        out_dir_paths = {rel[len("data/"):]: text for rel, text in (write or {}).items()}  # the fake writes into --out-dir
        (self.ai / "fake_export.json").write_text(
            json.dumps({"exit": exit, "write": out_dir_paths, "stdout": stdout, "stderr": stderr}, ensure_ascii=False),
            encoding="utf-8")

    def exports(self) -> list[dict]:
        path = self.ai / "export-runs.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

    def head(self) -> str:
        return self.git(self.web, "rev-parse", "HEAD").strip()

    def remote_head(self) -> str:
        return self.git(self.remote, "rev-parse", "refs/heads/main").strip()

    def staged(self) -> list[str]:
        return self.git(self.web, "diff", "--cached", "--name-only").split()

    def run(self, *args: str, key: str | None = FAKE_KEY, env: dict[str, str] | None = None,
            python: str | None = sys.executable) -> Run:
        run_env = dict(self.env)
        if key is not None:
            run_env["EQX_ADMIN_KEY"] = key
        run_env.update(env or {})
        command = [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
                   str(self.ai / "scripts" / WRAPPER.name), "-LockName", self.lock, "-TestUseProcessAdminKey", *args]
        if python is not None:
            command += ["-Python", python]
        before = set(self.logs.glob("vn30-web_*.log"))
        proc = subprocess.run(command, cwd=self.root, env=run_env, capture_output=True, timeout=240)
        new = sorted(set(self.logs.glob("vn30-web_*.log")) - before)
        assert len(new) == 1, (new, proc.returncode, proc.stderr.decode("utf-8", "replace"))
        log = new[0].read_text(encoding="utf-8-sig")
        lines = log.splitlines()
        assert lines[0].startswith("started "), log
        # every log ends with the exit code the task sees
        assert re.fullmatch(rf"exit={proc.returncode} finished=\S+", lines[-1]), (log, proc.stderr)
        return Run(proc.returncode, log)


@pytest.fixture
def sb(tmp_path: Path) -> Sandbox:
    return Sandbox(tmp_path / "sand box (x)").build()


@contextlib.contextmanager
def _mutex(name: str, deny_all: bool):
    """Holds the named mutex like a run in progress, or (deny_all) creates it with an empty DACL, as a run of another
    user or elevation level would leave it: then no other process may open it."""
    from ctypes import wintypes  # Windows only, and the module is skipped elsewhere

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.ReleaseMutex.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    class SecurityAttributes(ctypes.Structure):
        _fields_ = [("nLength", wintypes.DWORD), ("lpSecurityDescriptor", ctypes.c_void_p),
                    ("bInheritHandle", wintypes.BOOL)]

    descriptor, acl, attributes = ctypes.create_string_buffer(64), ctypes.create_string_buffer(64), None
    if deny_all:
        assert advapi32.InitializeSecurityDescriptor(descriptor, 1)
        assert advapi32.InitializeAcl(acl, 64, 2)
        assert advapi32.SetSecurityDescriptorDacl(descriptor, True, acl, False)
        attributes = ctypes.byref(SecurityAttributes(ctypes.sizeof(SecurityAttributes),
                                                     ctypes.cast(descriptor, ctypes.c_void_p), False))
    handle = kernel32.CreateMutexW(attributes, not deny_all, name)
    assert handle, ctypes.get_last_error()
    try:
        yield
    finally:
        if not deny_all:
            kernel32.ReleaseMutex(handle)
        kernel32.CloseHandle(handle)


def test_the_header_documents_every_exit_code_and_the_task_defaults():
    text = WRAPPER.read_bytes().decode("ascii")  # Windows PowerShell 5.1 reads a BOM-less script as ANSI
    header = text.split("[CmdletBinding()]")[0]
    for code in ("0", "1", "2", "3", "4", "5", "6", "other"):
        assert re.search(rf"^#\s+{code}\s+\S", header, re.MULTILINE), code
    assert "powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File" in header
    assert '[string]$LockName = "Global\\BTCVn30WebPublish"' in text
    assert "common.ps1" not in text.replace("no ops/common.ps1 here", "")  # never dot-sourced (#44)


@pytest.mark.parametrize("key", [None, "   "])
def test_a_missing_admin_key_exits_4_before_anything_is_exported(sb, key):
    sb.scenario(write=sb.data(SESSION))
    head, remote = sb.head(), sb.remote_head()
    run = sb.run(key=key)
    assert run.code == 4, run.log
    assert "refused: EQX_ADMIN_KEY is not set for the user" in run.log
    assert sb.exports() == []
    assert (sb.head(), sb.remote_head(), sb.staged()) == (head, remote, [])


def _unborn(sb: Sandbox) -> None:  # origin has main, the local branch main has no commit yet
    sb.git(sb.web, "update-ref", "-d", "refs/heads/main")


def _feature_branch(sb: Sandbox) -> None:
    sb.git(sb.web, "switch", "--quiet", "-c", "feature/site-tweak")


def _detached(sb: Sandbox) -> None:
    sb.git(sb.web, "switch", "--quiet", "--detach")


def _merge_in_progress(sb: Sandbox) -> None:
    (sb.web / ".git" / "MERGE_HEAD").write_text(sb.head() + "\n", encoding="utf-8")


def _no_repo_email(sb: Sandbox) -> None:  # the global config still has an identity: it must not be used
    sb.git(sb.web, "config", "--unset", "user.email")


def _fetch_fails(sb: Sandbox) -> None:  # origin out of reach (network down, repo moved)
    sb.git(sb.web, "remote", "set-url", "origin", str(sb.root / "missing.git"))


def _behind(sb: Sandbox) -> None:  # an edit on GitHub: only a fetch by the wrapper can see it
    other = sb.root / "other"
    sb.git(sb.root, "clone", "--quiet", str(sb.remote), str(other))
    (other / "README.md").write_text("edited on GitHub\n", encoding="utf-8")
    sb.git(other, "add", "README.md")
    sb.git(other, "commit", "--quiet", "-m", "readme")
    sb.git(other, "push", "--quiet", "origin", "main")


def _diverged(sb: Sandbox) -> None:
    _behind(sb)
    sb.commit_web({"data/symbols/ACB.json": "{}\n"}, "data: by hand")


def _ahead_outside_data(sb: Sandbox) -> None:  # a site change committed, waiting for its review before any push
    sb.commit_web({"app/page.tsx": "export default 2;\n"}, "site: WIP")


def _ahead_merge(sb: Sandbox) -> None:  # data-only commits, but joined by a merge commit
    sb.git(sb.web, "switch", "--quiet", "-c", "side")
    sb.commit_web({"data/symbols/VCB.json": "{}\n"}, "data: side")
    sb.git(sb.web, "switch", "--quiet", "main")
    sb.commit_web({"data/symbols/ACB.json": "{}\n"}, "data: main")
    sb.git(sb.web, "merge", "--quiet", "--no-ff", "--no-edit", "side")


def _ahead_other_identity(sb: Sandbox) -> None:  # a data-only commit made with the personal identity
    sb.commit_web({"data/symbols/ACB.json": "{}\n"}, "data: by hand", "-c", f"user.email={GLOBAL_EMAIL}")


REFUSALS = {
    "unborn": (_unborn, "branch main has no commit yet"),
    "feature-branch": (_feature_branch, "on branch 'feature/site-tweak', not main"),
    "detached": (_detached, "HEAD is detached, not on branch main"),
    "merge-in-progress": (_merge_in_progress, "is in progress (MERGE_HEAD)"),
    "no-repo-email": (_no_repo_email, "the web repo has no user.email of its own"),
    "fetch-fails": (_fetch_fails, "git fetch origin main failed (exit 128)"),
    "behind": (_behind, "main is 1 commit(s) behind origin/main"),
    "diverged": (_diverged, "main has diverged from origin/main (1 ahead, 1 behind)"),
    "ahead-outside-data": (_ahead_outside_data, "touch files outside data/: app/page.tsx"),
    "ahead-merge": (_ahead_merge, "a merge commit is among the 3 commit(s) ahead of origin/main"),
    "ahead-other-identity": (_ahead_other_identity, "carry another identity than the repo's user.email"),
}


@pytest.mark.parametrize("case", list(REFUSALS))
def test_a_failed_git_precondition_exits_3_before_anything_is_exported(sb, case):
    setup, reason = REFUSALS[case]
    setup(sb)
    sb.scenario(write=sb.data(SESSION))
    remote = sb.remote_head()
    worktree = sb.git(sb.web, "status", "--porcelain", "--untracked-files=all")
    run = sb.run()
    assert run.code == 3, run.log
    assert re.search(rf"^refused: .*{re.escape(reason)}.*; nothing exported or committed$", run.log, re.MULTILINE), run.log
    assert sb.exports() == []
    assert sb.remote_head() == remote
    assert sb.git(sb.web, "status", "--porcelain", "--untracked-files=all") == worktree  # data/ untouched


@pytest.mark.parametrize("variable, ident", [("GIT_AUTHOR_EMAIL", "GIT_AUTHOR_IDENT"),
                                             ("GIT_COMMITTER_EMAIL", "GIT_COMMITTER_IDENT")])
def test_an_identity_overridden_by_the_environment_is_refused(sb, variable, ident):
    sb.scenario(write=sb.data(SESSION))
    remote = sb.remote_head()
    run = sb.run(env={variable: GLOBAL_EMAIL})
    assert run.code == 3, run.log
    assert f"refused: {ident} is not the repo's user.email" in run.log
    assert sb.exports() == []
    assert sb.remote_head() == remote


@pytest.mark.parametrize("where, reason", [("inside", "is a folder inside the work tree of a repo"),
                                           ("plain", "not a git work tree")])
def test_a_web_root_that_is_not_the_top_of_a_repo_is_refused(sb, where, reason):
    root = sb.web / "data" if where == "inside" else sb.root / "plain"
    root.mkdir(exist_ok=True)
    sb.scenario(write=sb.data(SESSION))
    run = sb.run("-WebRoot", str(root))
    assert run.code == 3, run.log
    assert reason in run.log
    assert sb.exports() == []


@pytest.mark.parametrize("export_exit, expected", [(1, 1), (2, 2), (3, 6)])
def test_a_failed_export_commits_and_pushes_nothing(sb, export_exit, expected):
    sb.scenario(exit=export_exit, write=sb.data(SESSION), stderr=STDERR_VI)  # data written, then a failure
    head, remote = sb.head(), sb.remote_head()
    run = sb.run()
    assert run.code == expected, run.log
    (export,) = sb.exports()
    assert export["argv"][0] == "--out-dir" and len(export["argv"]) == 2
    assert Path(export["argv"][1]) == sb.web / "data"
    assert Path(export["cwd"]) == sb.ai
    assert STDERR_VI in run.log  # PYTHONIOENCODING=utf-8: the exporter's Vietnamese stderr reaches the log intact
    assert "> git add" not in run.log and "> git commit" not in run.log and "> git push" not in run.log
    assert (sb.head(), sb.remote_head(), sb.staged()) == (head, remote, [])
    if expected == 6:
        assert "the exporter ended with exit code 3, which it never returns by itself" in run.log


def test_unchanged_data_makes_no_commit_and_no_push(sb):
    sb.scenario(write=sb.data(OLD_SESSION))  # the very bytes already committed
    sb.logs.mkdir(parents=True)
    old_logs = [f"vn30-web_2000010{day}_190000.log" for day in range(1, 7)]
    for name in old_logs:
        (sb.logs / name).write_text("an old run\n", encoding="utf-8")
    head, remote = sb.head(), sb.remote_head()
    run = sb.run("-LogRetentionCount", "5")
    assert run.code == 0, run.log
    assert len(sb.exports()) == 1
    assert "data unchanged: nothing to commit" in run.log
    assert "nothing to push: main is at origin/main" in run.log
    assert "> git commit" not in run.log and "> git push" not in run.log
    assert (sb.head(), sb.remote_head(), sb.staged()) == (head, remote, [])
    assert not sb.records.exists()  # no hook ran: no commit, no push
    kept = sorted(p.name for p in sb.logs.glob("vn30-web_*.log"))
    assert len(kept) == 5 and kept[:4] == old_logs[-4:]  # the newest 5 by name: 4 old runs and this one


def test_changed_data_commits_only_data_json_and_pushes_it(sb):
    files = sb.data(SESSION, close=20500)
    files["data/symbols/ACB.json"] = json.dumps({"symbol": "ACB"}) + "\n"  # a ticker file the repo does not have yet
    sb.scenario(write=files, stdout='{"session": "%s", "changed": 4}' % SESSION)
    sb.write_web({"app/page.tsx": "export default 3;\n"})
    sb.git(sb.web, "add", "app/page.tsx")  # staged by hand, not part of the data commit
    sb.write_web({"data/symbols/MBB.json.tmp": "{}\n"})  # left by an interrupted export
    base = sb.head()
    run = sb.run()
    assert run.code == 0, run.log
    head = sb.head()
    assert sb.remote_head() == head, run.log
    assert sb.git(sb.web, "rev-parse", "HEAD~1").strip() == base
    committed = sb.git(sb.web, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").split()
    assert sorted(committed) == ["data/meta.json", "data/screen.json", "data/symbols/ACB.json", "data/symbols/MBB.json"]
    assert sb.staged() == ["app/page.tsx"]
    assert "?? data/symbols/MBB.json.tmp" in sb.git(sb.web, "status", "--porcelain", "--untracked-files=all")
    raw = subprocess.run([GIT, "cat-file", "commit", head], cwd=sb.remote, env=sb.env, capture_output=True,
                         check=True).stdout  # the commit object the public repo serves
    header, _, body = raw.partition(b"\n\n")
    assert body == MESSAGE.encode("utf-8")  # "phiên" as UTF-8 bytes
    header_lines = header.decode("utf-8").splitlines()
    assert not any(line.startswith("encoding ") for line in header_lines)
    for role in ("author", "committer"):
        (line,) = [line for line in header_lines if line.startswith(role + " ")]
        assert f"<{REPO_EMAIL}>" in line, line
    assert GLOBAL_EMAIL.encode() not in raw
    # the admin key never reaches git; ssh is told never to wait for an answer
    assert sb.records.read_text(encoding="utf-8").splitlines() == [
        "pre-commit key=absent", f"pre-push key=absent ssh={DEFAULT_SSH}"]
    assert '{"session": "%s", "changed": 4}' % SESSION in run.log
    assert "pushed 1 commit(s) to origin/main" in run.log


def test_a_failed_push_is_retried_by_the_next_run_even_when_data_is_unchanged(sb):
    sb.scenario(write=sb.data(SESSION))
    (sb.remote / "reject-push").write_text("", encoding="utf-8")
    base = sb.remote_head()
    first = sb.run()
    assert first.code == 11, first.log  # git push exits 1, lifted to 11 so that it never reads as a backend error
    assert "push rejected by the test remote" in first.log
    data_commit = sb.head()
    assert data_commit != base and sb.remote_head() == base
    (sb.remote / "reject-push").unlink()
    second = sb.run()  # same scenario: the export writes the same bytes again
    assert second.code == 0, second.log
    assert "data unchanged: nothing to commit" in second.log
    assert "pushed 1 commit(s) to origin/main" in second.log
    assert sb.head() == data_commit and sb.remote_head() == data_commit
    assert len(sb.exports()) == 2


def test_only_main_goes_out_whatever_push_default_says(sb):
    sb.git(sb.web, "switch", "--quiet", "-c", "feature")
    sb.git(sb.web, "push", "--quiet", "origin", "feature")
    sb.commit_web({"app/page.tsx": "export default 9;\n"}, "site: not reviewed yet")
    sb.git(sb.web, "switch", "--quiet", "main")
    sb.git(sb.web, "config", "push.default", "matching")  # a bare `git push origin` would push feature too
    feature = sb.git(sb.remote, "rev-parse", "refs/heads/feature").strip()
    sb.scenario(write=sb.data(SESSION))
    run = sb.run()
    assert run.code == 0, run.log
    assert sb.remote_head() == sb.head()
    assert sb.git(sb.remote, "rev-parse", "refs/heads/feature").strip() == feature


@pytest.mark.parametrize("setting, seen", [("GIT_SSH_COMMAND", "ssh -i C:/keys/web -o BatchMode=yes"),
                                           ("GIT_SSH", "<unset>"),
                                           ("core.sshCommand", "<unset>")])
def test_an_ssh_transport_the_user_chose_is_left_as_it_is(sb, setting, seen):
    sb.scenario(write=sb.data(SESSION))
    env = {}
    if setting == "GIT_SSH_COMMAND":
        env[setting] = seen
    elif setting == "GIT_SSH":
        env[setting] = "C:/tools/plink.exe"
    else:
        sb.git(sb.web, "config", setting, "ssh -i C:/keys/web")
    run = sb.run(env=env)
    assert run.code == 0, run.log
    assert sb.records.read_text(encoding="utf-8").splitlines()[-1] == f"pre-push key=absent ssh={seen}"
    assert sb.remote_head() == sb.head()


@pytest.mark.parametrize("deny_all, reason", [(False, "another run holds the lock"), (True, "may not open it")])
def test_a_run_started_while_the_lock_is_taken_exits_5(sb, deny_all, reason):
    sb.scenario(write=sb.data(SESSION))
    sb.logs.mkdir(parents=True)
    leftover = sb.logs / "vn30-web_20000101_000000.log.part"
    leftover.write_text("may belong to the run that holds the lock\n", encoding="utf-8")
    head, remote = sb.head(), sb.remote_head()
    with _mutex(sb.lock, deny_all):
        run = sb.run()
    assert run.code == 5, run.log
    assert reason in run.log
    assert sb.exports() == []
    assert leftover.exists()  # only the lock holder cleans the log folder
    assert (sb.head(), sb.remote_head()) == (head, remote)


def test_an_error_inside_the_wrapper_is_logged_and_exits_6(sb):
    sb.scenario(write=sb.data(SESSION))
    sb.logs.mkdir(parents=True)
    leftovers = [sb.logs / f"vn30-web_20000101_000000.log{suffix}" for suffix in (".part", ".part.err", ".msg")]
    for path in leftovers:
        path.write_text("left by a killed run\n", encoding="utf-8")
    head, remote = sb.head(), sb.remote_head()
    run = sb.run(python=None)  # the task's default <ai>/venv/Scripts/python.exe does not exist in the sandbox
    assert run.code == 6, run.log
    started = re.search(r" python=(.+)$", run.log.splitlines()[0])
    assert started and Path(started.group(1)) == sb.ai / "venv" / "Scripts" / "python.exe"
    assert re.search(r"^ERROR: \w+: .+", run.log, re.MULTILINE), run.log
    assert not any(path.exists() for path in leftovers)
    assert [p.name for p in sb.logs.iterdir() if not p.name.endswith(".log")] == []  # nor this run's own .part files
    assert sb.exports() == []
    assert (sb.head(), sb.remote_head(), sb.staged()) == (head, remote, [])
