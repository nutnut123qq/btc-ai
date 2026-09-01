from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


ROOT = Path(__file__).resolve().parents[2]


def _requirements(path: Path) -> list[Requirement]:
    return [
        Requirement(line)
        for raw in path.read_text(encoding="utf-8").splitlines()
        if (line := raw.strip()) and not line.startswith("#")
    ]


def test_lock_is_exact_and_satisfies_direct_requirements():
    locked = _requirements(ROOT / "requirements.lock.txt")
    direct = _requirements(ROOT / "requirements.txt")
    locked_by_name = {canonicalize_name(item.name): item for item in locked}

    assert len(locked_by_name) == len(locked), "lock contains duplicate packages"
    for item in locked:
        specs = list(item.specifier)
        assert len(specs) == 1 and specs[0].operator == "==", f"lock is not exact: {item}"

    for item in direct:
        name = canonicalize_name(item.name)
        assert name in locked_by_name, f"direct dependency missing from lock: {item.name}"
        locked_version = next(iter(locked_by_name[name].specifier)).version
        assert item.specifier.contains(locked_version, prereleases=True), (
            f"locked {item.name}=={locked_version} violates {item.specifier}"
        )


def test_linux_only_xgboost_dependency_is_explicitly_pinned():
    locked = _requirements(ROOT / "requirements.lock.txt")
    nccl = [item for item in locked if canonicalize_name(item.name) == "nvidia-nccl-cu12"]
    assert len(nccl) == 1
    assert str(nccl[0].specifier) == "==2.27.7"
    assert nccl[0].marker is not None
    assert nccl[0].marker.evaluate({"platform_system": "Linux"})
    assert not nccl[0].marker.evaluate({"platform_system": "Windows"})


def test_ci_and_docker_install_the_reviewed_closure_without_resolving_extras():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "pip install --no-deps --requirement requirements.lock.txt" in workflow
    assert "pip install --no-cache-dir --no-deps -r requirements.lock.txt" in dockerfile
