"""VN30-2 panel evaluator (roadmap 2.3): purged walk-forward, rank IC, DSR with the registry count, PBO via CSCV.

Pure evaluation code for the frozen protocol ``docs/research/protocols/vn30-2.json``. It never fetches data, never
fits a declared model by itself and never registers a trial: the caller registers every trial (status planned) in
``trial_registry.py`` before running it, then passes panels, scores and return series here.

Panel layout: a long ``pandas.DataFrame`` with one row per (decision date, member) and at least the columns
``session`` (integer position of the decision session on the HOSE calendar), ``date`` (ISO date of that session),
``symbol`` and ``target`` (the protocol target: excess return over the equal-weight mean of the members in force).

Choices this module makes where the protocol text is silent (stated so a reviewer can check them):

* Purge and embargo (``purge_embargo_mask``): the label of a decision at session ``s`` spans sessions
  ``[s, s + purge]`` (``purge = 21``, exit at the open of ``d+21``). A training decision is kept only when its label
  ends more than ``embargo`` sessions before the first test decision (``s + purge + embargo < test_start``), or, for
  a training block after the test block, when it starts more than ``purge + embargo`` sessions after the last test
  decision. In a forward-only walk the embargo therefore acts as an extra gap before the test block; this is more
  conservative than the minimal overlap purge.
* Walk-forward (``yearly_folds``): one fold per calendar year of test decisions, refit at the start of that year on
  every earlier decision that survives the purge/embargo (expanding). The HOSE session ``calendar`` is REQUIRED and
  ``yearly_folds`` derives the last session of the window itself (``calendar[last] <= test_to <
  calendar[last + 1]``; a ``last_session_in_window`` passed by the caller must equal it): test decisions whose label
  ends after it (``s + purge > last``) are always dropped, so a development score never reads a holdout return and a
  holdout score never reads a post-holdout (post-FTSE, forward-only) return. A test window that reaches into the
  protocol holdout is refused unless the record of an acquired ``HoldoutLock`` is passed (and the lock lives in the
  protocol evidence root, see ``verify_holdout_lock``); a window past the holdout end is always refused (that regime
  is judged on forward records only). Decision dates before the protocol history start (2018-01-22, the first VN30
  period and the development window start) are refused unless ``allow_pre_history`` is set explicitly.
* One guarded path for every model, baselines included: folds are sealed by ``yearly_folds`` (or ``forward_fold``
  for a forward decision whose label lies past the calendar) and ``walk_forward_scores`` / ``baseline_scores`` /
  ``window_rank_ic`` refuse any other ``Fold`` (``UnguardedFold``) and re-check every fold before use (label end
  inside the window, purge + embargo before the fit session, panel dates equal to the fold's calendar dates, holdout
  lock re-verified on disk). ``window_rank_ic`` scores only the test rows of those folds, whatever else the score
  series holds. ``rank_ic_by_date`` / ``ic_summary`` / ``paired_ic_difference`` stay pure table functions without a
  window; a protocol number must come from ``window_rank_ic``.
* ``fit_predict`` never sees the label: ``walk_forward_scores`` drops the label columns (default ``target``) from the
  test rows it hands over (a declared label column missing from the panel raises; ``feature_columns`` optionally
  restricts the test rows to an explicit allowlist); the evaluator keeps the label to itself.
* Rank IC: Spearman with average ranks for ties, per decision date, NaN when fewer than ``min_names`` usable rows or
  a constant side. Mean and t use only non-overlapping dates (greedy, spacing >= ``horizon`` sessions). A score
  Series is aligned to the panel by index, never by position.
* Bootstrap: circular moving-block bootstrap over decision dates (dates are the clusters; ``block_size = 1`` is the
  plain cluster bootstrap by date).
* DSR (Bailey & Lopez de Prado 2014): ``N`` is the HOSE trial count from ``trial_registry`` at evaluation time
  (statuses run + planned), and the protocol's own entry must be registered (planned or run) with at least as many
  trials as the series evaluated. ``V[SR_n]`` (``SR_VARIANCE_ESTIMATOR``, fixed here before any holdout is opened):
  ``max(cross-trial sample variance (ddof 1) of the per-period Sharpe ratios of this run's trials, mean over those
  trials of the null estimator variance (1 - g3*SR + (g4 - 1)/4*SR^2)/(T - 1))``. The floor keeps highly correlated
  trial series (momentum variants, ridge and composite on the same features) from shrinking ``V`` and inflating the
  DSR; fewer than 2 trial series raise.
* PBO (Bailey, Borwein, Lopez de Prado & Zhu 2017): CSCV with ``S = 16`` equal blocks in time order (the oldest
  ``T mod S`` rows are dropped), all ``C(16, 8) = 12870`` splits, Sharpe ratio as the performance statistic, relative
  rank ``w = rank / (N + 1)`` with average ranks; ``pbo`` counts ``logit(w) <= 0`` (a median rank counts as
  overfit) and ``pboStrict`` counts ``logit(w) < 0``. The gate value is ``gatePbo = pbo`` (the conservative one),
  fixed in code so the choice is not made after the holdout is seen.
* Holdout lock: ``HoldoutLock.acquire`` runs BEFORE any holdout number is computed. It creates an exclusive
  sentinel in the protocol's evidence root and a lock record in the run directory, and refuses when any run of
  the same protocol id already holds one. The evidence root defaults to the protocol output folder of the MAIN
  checkout (resolved through git's common dir), so every worktree shares one lock. ``verify_holdout_lock`` accepts
  only a lock under that root (or under an ``evidence_root`` passed explicitly, e.g. a test directory).
* Provenance: ``evaluator_provenance`` gives the protocol sha256 and the sha256 of this file; every report built on
  this module records both (see docs/research/protocols/vn30-panel-eval-notes.md).
"""

from __future__ import annotations

import bisect
import hashlib
import hmac
import itertools
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import trial_registry

EVALUATOR_PATH = Path(__file__).resolve()
PROTOCOL_PATH = EVALUATOR_PATH.parent / "docs" / "research" / "protocols" / "vn30-2.json"
PROTOCOL_ID = "vn30-2"
PURGE_SESSIONS = 21
EMBARGO_SESSIONS = 20
REBALANCE_SESSIONS = 20
PBO_BLOCKS = 16
GATE_DSR_MIN = 0.95  # protocol gates.holdout: "DSR >= 0.95"
GATE_PBO_MAX = 0.3   # protocol gates.holdout: "PBO < 0.3", applied to gatePbo (= pbo, logit <= 0)
EULER_GAMMA = 0.5772156649015329
LOCK_NAME = "holdout.lock.json"
_NORMAL = NormalDist()
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_FOLD_KEY = os.urandom(32)  # per-process key sealing the folds built here (guards against hand-built folds)


# --------------------------------------------------------------------------------------------- protocol


def protocol_sha256(path: Path | str = PROTOCOL_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_parameters(path: Path | str = PROTOCOL_PATH) -> dict[str, Any]:
    """The validation parameters this evaluator uses, read from the frozen protocol file."""
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    val = spec["validation"]
    return {
        "id": spec["id"],
        "sha256": protocol_sha256(path),
        "purgeSessions": int(val["purgeSessions"]),
        "embargoSessions": int(val["embargoSessions"]),
        "rebalanceEverySessions": int(spec["decisionSchedule"]["rebalanceEverySessions"]),
        "development": dict(val["developmentWindow"]),
        "holdout": {"from": val["holdout"]["from"], "to": val["holdout"]["to"]},
        "rankedModels": [m["id"] for m in spec["models"]["list"] if m["id"] != "B-ew"],
    }


def evaluator_sha256(path: Path | str = EVALUATOR_PATH) -> str:
    """sha256 of this evaluator's source file (recorded next to the protocol sha256 in every report)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluator_provenance(protocol_path: Path | str = PROTOCOL_PATH) -> dict[str, str]:
    return {"protocolSha256": protocol_sha256(protocol_path), "evaluatorSha256": evaluator_sha256()}


def _holdout_window() -> tuple[str, str]:
    holdout = protocol_parameters()["holdout"]
    return str(holdout["from"]), str(holdout["to"])


def history_start() -> str:
    """First decision date the protocol allows (the development window start = the first VN30 period, 2018-01-22)."""
    return str(protocol_parameters()["development"]["from"])


def repo_main_checkout(start: Path | str = Path(__file__).resolve().parent) -> Path:
    """The main checkout of this repo: ``start`` itself, or, in a git worktree, the checkout owning the common dir."""
    start = Path(start)
    dot_git = start / ".git"
    if dot_git.is_file():
        text = dot_git.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir:"):
            gitdir = Path(text[len("gitdir:"):].strip())
            if not gitdir.is_absolute():
                gitdir = (start / gitdir).resolve()
            commondir = gitdir / "commondir"
            if commondir.is_file():
                common = Path(commondir.read_text(encoding="utf-8").strip())
                common = common if common.is_absolute() else (gitdir / common)
                return common.resolve().parent
    return start


def default_evidence_root(protocol_id: str = PROTOCOL_ID) -> Path:
    """The protocol output folder ``docs/research/evidence/<id>`` of the main checkout (shared by every worktree)."""
    return repo_main_checkout() / "docs" / "research" / "evidence" / protocol_id


# --------------------------------------------------------------------------------------------- walk-forward


def purge_embargo_mask(train_sessions: Sequence[int] | np.ndarray, test_start: int, test_end: int, *,
                       purge: int = PURGE_SESSIONS, embargo: int = EMBARGO_SESSIONS) -> np.ndarray:
    """Boolean keep-mask over candidate training decisions for a test block [test_start, test_end] (sessions)."""
    if test_end < test_start:
        raise ValueError("test_end must be >= test_start")
    if purge < 0 or embargo < 0:
        raise ValueError("purge and embargo must be >= 0")
    s = np.asarray(train_sessions, dtype=np.int64)
    before = s + purge + embargo < test_start
    after = s > test_end + purge + embargo
    return before | after


@dataclass(frozen=True)
class Fold:
    """One refit: test decisions, training decisions and the guard data the fold was built with.

    Only folds returned by ``yearly_folds`` or ``forward_fold`` carry a valid ``seal``; every scoring function of
    this module refuses any other fold (a hand-built one, or one changed with ``dataclasses.replace``).
    """

    year: int
    test_sessions: tuple[int, ...]
    train_sessions: tuple[int, ...]
    purged_sessions: tuple[int, ...]
    dropped_label_overrun: tuple[int, ...] = ()
    kind: str = "unguarded"  # "walk-forward" (yearly_folds) or "forward" (forward_fold)
    window: tuple[str, str] | None = None  # (test_from, test_to) of a walk-forward fold
    last_session: int | None = None  # last calendar session of the window (walk-forward) or of the calendar (forward)
    fit_at: int | None = None  # fit decision of a forward fold (first grid decision of its calendar year)
    purge: int = PURGE_SESSIONS
    embargo: int = EMBARGO_SESSIONS
    test_dates: tuple[str, ...] = ()
    train_dates: tuple[str, ...] = ()
    allow_pre_history: bool = False
    holdout_lock_json: str | None = None
    evidence_root: str | None = None
    seal: str = field(default="", repr=False, compare=False)

    @property
    def fit_session(self) -> int:
        return self.fit_at if self.fit_at is not None else self.test_sessions[0]

    def as_dict(self) -> dict[str, Any]:
        return {"year": self.year, "fitAtSession": self.fit_session, "testDates": len(self.test_sessions),
                "trainDates": len(self.train_sessions), "purgedDates": list(self.purged_sessions),
                "droppedLabelOverrun": list(self.dropped_label_overrun)}


class HoldoutSealed(RuntimeError):
    """A test window reaches the protocol holdout without an acquired holdout lock (or past the holdout end)."""


class UnguardedFold(ValueError):
    """A fold that was not built (sealed) by ``yearly_folds`` / ``forward_fold``, or that no longer passes its guard."""


def _fold_digest(fold: Fold) -> str:
    body = {k: v for k, v in asdict(fold).items() if k != "seal"}
    payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(_FOLD_KEY, payload, hashlib.sha256).hexdigest()


def _sealed(**fields: Any) -> Fold:
    fold = Fold(**fields)
    object.__setattr__(fold, "seal", _fold_digest(fold))
    return fold


def validate_fold(fold: Fold, panel: pd.DataFrame | None = None, *,
                  label_columns: Sequence[str] = ("target",)) -> None:
    """Re-check a fold before any score or IC is computed from it; raises ``UnguardedFold`` / ``HoldoutSealed``.

    Checks: the seal of ``yearly_folds`` / ``forward_fold``; every test label ends inside the window (walk-forward:
    ``s + purge <= last``; forward: the decision is the last calendar session, so its label lies past the data);
    every training label plus the embargo ends before the fit session; a window reaching the holdout carries a lock
    that still verifies on disk; with ``panel``: the panel's dates of the fold's sessions equal the fold's calendar
    dates, no decision before the history start (unless allowed), and a forward fold's test rows carry no label.
    """
    if not isinstance(fold, Fold) or not fold.seal or not hmac.compare_digest(fold.seal, _fold_digest(fold)):
        raise UnguardedFold(f"fold {getattr(fold, 'year', '?')} was not built by yearly_folds/forward_fold "
                            "(hand-built or modified folds are refused)")
    if fold.kind not in ("walk-forward", "forward") or fold.last_session is None or not fold.test_sessions:
        raise UnguardedFold(f"fold {fold.year}: incomplete guard data")
    test = np.asarray(fold.test_sessions, dtype=np.int64)
    train = np.asarray(fold.train_sessions, dtype=np.int64)
    if np.any(np.diff(test) <= 0) or (len(train) and np.any(np.diff(train) <= 0)):
        raise UnguardedFold(f"fold {fold.year}: sessions must be strictly increasing")
    if len(fold.test_dates) != len(test) or len(fold.train_dates) != len(train):
        raise UnguardedFold(f"fold {fold.year}: session dates missing")
    if len(train) and int(train.max()) + fold.purge + fold.embargo >= fold.fit_session:
        raise UnguardedFold(f"fold {fold.year}: a training label (plus embargo) reaches the fit session")
    hold_from, hold_to = _holdout_window()
    if fold.kind == "walk-forward":
        if fold.window is None:
            raise UnguardedFold(f"fold {fold.year}: no window")
        lo, hi = fold.window
        if np.any(test + fold.purge > fold.last_session):
            raise UnguardedFold(f"fold {fold.year}: a test label ends after the last session of the window")
        if any(not (lo <= d <= hi) for d in fold.test_dates):
            raise UnguardedFold(f"fold {fold.year}: a test decision lies outside the window [{lo}, {hi}]")
        if hi > hold_to:
            raise HoldoutSealed(f"fold {fold.year}: window ends {hi}, after the holdout end {hold_to}")
        if lo <= hold_to and hi >= hold_from:
            if fold.holdout_lock_json is None:
                raise HoldoutSealed(f"fold {fold.year}: holdout rows without a verified holdout lock")
            verify_holdout_lock(json.loads(fold.holdout_lock_json), evidence_root=fold.evidence_root)
    else:
        if tuple(fold.test_sessions) != (fold.last_session,) or fold.fit_at is None or fold.fit_at > test[0]:
            raise UnguardedFold(f"fold {fold.year}: a forward fold scores only the last session of its calendar")
    start = history_start()
    if not fold.allow_pre_history and any(d < start for d in (*fold.test_dates, *fold.train_dates)):
        raise UnguardedFold(f"fold {fold.year}: decision before the protocol history start {start}")
    if panel is None:
        return
    expected = dict(zip(fold.test_sessions, fold.test_dates)) | dict(zip(fold.train_sessions, fold.train_dates))
    sessions = panel["session"].to_numpy()
    rows = panel[np.isin(sessions, list(expected))]
    if len(rows):
        want = rows["session"].map(expected)
        if not (want.astype(str).to_numpy() == rows["date"].astype(str).to_numpy()).all():
            raise UnguardedFold(f"fold {fold.year}: panel dates differ from the fold's calendar dates")
    if fold.kind == "forward":
        test_rows = panel[np.isin(sessions, fold.test_sessions)]
        for col in label_columns:
            if col in test_rows.columns and test_rows[col].notna().any():
                raise UnguardedFold(f"fold {fold.year}: a forward decision has a label ({col}); its calendar must "
                                    "end at the decision session")


def last_session_on_or_before(calendar_dates: Sequence[str], day: str) -> int:
    """Position of the last HOSE session dated <= ``day`` in the chronological session calendar (index = session)."""
    dates = list(calendar_dates)
    if dates != sorted(dates) or len(set(dates)) != len(dates):
        raise ValueError("calendar dates must be chronological and unique")
    pos = bisect.bisect_right(dates, day) - 1
    if pos < 0:
        raise ValueError(f"no session on or before {day}")
    return pos


def yearly_folds(decision_sessions: Sequence[int], decision_dates: Sequence[str], *, test_from: str, test_to: str,
                 calendar: Sequence[str], last_session_in_window: int | None = None,
                 purge: int = PURGE_SESSIONS, embargo: int = EMBARGO_SESSIONS, min_train_dates: int = 1,
                 holdout_lock: Mapping[str, Any] | None = None, evidence_root: Path | str | None = None,
                 allow_pre_history: bool = False) -> list[Fold]:
    """Expanding walk-forward, one fold per calendar year of test decisions in [test_from, test_to].

    ``decision_sessions`` / ``decision_dates`` list the decision dates (chronological, unique); a session is the
    position of its date in ``calendar`` (the chronological HOSE session calendar, REQUIRED). The last session of
    the window is computed here: ``last`` with ``calendar[last] <= test_to < calendar[last + 1]`` (or the end of the
    calendar); ``last_session_in_window``, when passed, must equal it. A test decision whose label ends after it
    (``s + purge > last``) is always dropped. Training for the fold of year Y is every earlier decision kept by
    ``purge_embargo_mask``. Years with fewer than ``min_train_dates`` training dates are skipped. A decision dated
    before the protocol history start (2018-01-22) is refused unless ``allow_pre_history``.

    A window intersecting the protocol holdout needs ``holdout_lock``, the record returned by
    ``HoldoutLock.acquire`` (verified on disk, under ``evidence_root`` or by default the protocol evidence root of
    the main checkout); a window ending after the holdout end is refused. The folds are sealed: only they are
    accepted by ``walk_forward_scores``, ``baseline_scores`` and ``window_rank_ic``.
    """
    if test_to < test_from:
        raise ValueError("test_to must be >= test_from")
    cal = list(calendar)
    last = last_session_on_or_before(cal, test_to)
    if not (cal[last] <= test_to and (last + 1 == len(cal) or cal[last + 1] > test_to)):
        raise ValueError(f"calendar check failed for the last session <= {test_to}")
    if last_session_in_window is not None:
        if isinstance(last_session_in_window, bool) or not isinstance(last_session_in_window, (int, np.integer)):
            raise TypeError("last_session_in_window must be the integer position of the last session <= test_to")
        if int(last_session_in_window) != last:
            raise ValueError(f"last_session_in_window={last_session_in_window} is not the last session dated <= "
                             f"{test_to} (calendar position {last})")
    hold_from, hold_to = _holdout_window()
    if test_to > hold_to:
        raise HoldoutSealed(f"test window ends {test_to}, after the holdout end {hold_to}: the later regime is "
                            "judged on forward records only")
    lock_json = None
    root = None if evidence_root is None else str(Path(evidence_root))
    if test_from <= hold_to and test_to >= hold_from:
        if holdout_lock is None:
            raise HoldoutSealed(f"test window [{test_from}, {test_to}] reaches the holdout [{hold_from}, {hold_to}]: "
                                "acquire HoldoutLock first and pass its record as holdout_lock")
        verify_holdout_lock(holdout_lock, evidence_root=root)
        lock_json = json.dumps(dict(holdout_lock), sort_keys=True, ensure_ascii=False)
    s = np.asarray(decision_sessions, dtype=np.int64)
    d = list(decision_dates)
    if len(s) != len(d):
        raise ValueError("decision_sessions and decision_dates differ in length")
    if len(s) and (np.any(np.diff(s) <= 0) or d != sorted(d)):
        raise ValueError("decision dates must be chronological and unique")
    if len(s) and (s.min() < 0 or s.max() >= len(cal) or any(cal[int(x)] != day for x, day in zip(s, d))):
        raise ValueError("decision dates differ from the calendar dates of their sessions")
    start = history_start()
    if not allow_pre_history and any(day < start for day in d):
        raise ValueError(f"decision dates before the protocol history start {start} (pass allow_pre_history=True "
                         "only for a declared non-protocol use)")
    folds: list[Fold] = []
    in_window = [i for i, day in enumerate(d) if test_from <= day <= test_to]
    for year in sorted({int(d[i][:4]) for i in in_window}):
        test_idx = [i for i in in_window if int(d[i][:4]) == year]
        overrun = tuple(int(s[i]) for i in test_idx if s[i] + purge > last)
        test_idx = [i for i in test_idx if s[i] + purge <= last]
        if not test_idx:
            continue
        test_s = s[test_idx]
        cand_idx = [i for i in range(len(s)) if s[i] < test_s[0]]
        candidates = s[cand_idx]
        keep = purge_embargo_mask(candidates, int(test_s[0]), int(test_s[-1]), purge=purge, embargo=embargo)
        train = candidates[keep]
        if len(train) < min_train_dates:
            continue
        folds.append(_sealed(
            year=year, test_sessions=tuple(int(x) for x in test_s), train_sessions=tuple(int(x) for x in train),
            purged_sessions=tuple(int(x) for x in candidates[~keep]), dropped_label_overrun=overrun,
            kind="walk-forward", window=(test_from, test_to), last_session=last, purge=purge, embargo=embargo,
            test_dates=tuple(d[i] for i in test_idx),
            train_dates=tuple(d[i] for i, k in zip(cand_idx, keep) if k),
            allow_pre_history=allow_pre_history, holdout_lock_json=lock_json, evidence_root=root))
    return folds


def forward_fold(decision_sessions: Sequence[int], calendar: Sequence[str], *, purge: int = PURGE_SESSIONS,
                 embargo: int = EMBARGO_SESSIONS, allow_pre_history: bool = False) -> Fold:
    """Sealed fold for a forward decision: the last of ``decision_sessions`` must be the last session of
    ``calendar`` (the market cut at the decision close), so its label lies past every bar given.

    Training is the yearly refit of ``yearly_folds``: every earlier decision ``s`` with ``s + purge + embargo < f``,
    ``f`` the first decision of the decision's calendar year.
    """
    cal = list(calendar)
    s = np.asarray(decision_sessions, dtype=np.int64)
    if not len(s) or np.any(np.diff(s) <= 0):
        raise ValueError("decision sessions must be non-empty and strictly increasing")
    i = int(s[-1])
    if i != len(cal) - 1:
        raise ValueError(f"forward decision {i} is not the last session of the calendar ({len(cal) - 1}): cut the "
                         "market at the decision close first")
    if s.min() < 0:
        raise ValueError("negative session")
    d = [cal[int(x)] for x in s]
    start = history_start()
    if not allow_pre_history and any(day < start for day in d):
        raise ValueError(f"decision dates before the protocol history start {start}")
    year = d[-1][:4]
    fit_at = next(int(x) for x, day in zip(s, d) if day[:4] == year)
    cand_idx = [k for k in range(len(s)) if s[k] < fit_at]
    candidates = s[cand_idx]
    keep = (purge_embargo_mask(candidates, fit_at, i, purge=purge, embargo=embargo) if len(candidates)
            else np.zeros(0, dtype=bool))
    return _sealed(year=int(year), test_sessions=(i,), train_sessions=tuple(int(x) for x in candidates[keep]),
                   purged_sessions=tuple(int(x) for x in candidates[~keep]), kind="forward", last_session=i,
                   fit_at=fit_at, purge=purge, embargo=embargo, test_dates=(d[-1],),
                   train_dates=tuple(d[k] for k, kk in zip(cand_idx, keep) if kk),
                   allow_pre_history=allow_pre_history)


FitPredict = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def walk_forward_scores(panel: pd.DataFrame, folds: Sequence[Fold], fit_predict: FitPredict, *,
                        purge: int = PURGE_SESSIONS, label_columns: Sequence[str] = ("target",),
                        feature_columns: Sequence[str] | None = None) -> pd.Series:
    """Out-of-sample scores: for each fold, ``fit_predict(train_rows, test_rows)`` returns one score per test row.

    Every fold must come from ``yearly_folds`` / ``forward_fold`` and is re-checked by ``validate_fold``.
    ``test_rows`` never carry the ``label_columns`` (the out-of-sample label stays with the evaluator); a declared
    label column missing from the panel raises. With ``feature_columns``, test rows carry only ``session``, ``date``,
    ``symbol`` and those columns (an explicit allowlist). Training rows keep every column. Rows outside every fold's
    test block stay NaN. A guard re-checks that no training label reaches the fit date.
    """
    if not panel.index.is_unique:
        raise ValueError("panel index must be unique")
    labels = list(label_columns)
    missing = [c for c in labels if c not in panel.columns]
    if missing:
        raise ValueError(f"declared label column(s) {missing} missing from the panel")
    allow = None
    if feature_columns is not None:
        leaked = [c for c in feature_columns if c in labels]
        if leaked:
            raise ValueError(f"label column(s) {leaked} in the feature allowlist")
        allow = [c for c in ("session", "date", "symbol") if c in panel.columns] + [
            c for c in feature_columns if c not in ("session", "date", "symbol")]
    scores = pd.Series(np.nan, index=panel.index, dtype=float)
    sessions = panel["session"].to_numpy()
    for fold in folds:
        validate_fold(fold, panel, label_columns=labels)
        train = panel[np.isin(sessions, fold.train_sessions)]
        test = panel[np.isin(sessions, fold.test_sessions)]
        if len(train) and int(train["session"].max()) + purge >= fold.fit_session:
            raise AssertionError(f"fold {fold.year}: a training label reaches the fit session")
        if not len(test):
            continue
        handed = test[allow] if allow is not None else test.drop(columns=labels)
        out = np.asarray(fit_predict(train, handed), dtype=float).reshape(-1)
        if len(out) != len(test):
            raise ValueError(f"fold {fold.year}: fit_predict returned {len(out)} scores for {len(test)} rows")
        if not scores.loc[test.index].isna().all():
            raise ValueError(f"fold {fold.year}: test rows scored twice (overlapping folds)")
        scores.loc[test.index] = out
    return scores


def baseline_scores(panel: pd.DataFrame, folds: Sequence[Fold],
                    rule: str | Callable[[pd.DataFrame], np.ndarray], *,
                    label_columns: Sequence[str] = ("target",)) -> pd.Series:
    """Scores of a fixed (unfitted) rule on the guarded folds only: a column name or ``rule(test_rows)``.

    The same guarded path as the fitted models (``walk_forward_scores``); the rule never sees the label.
    """
    if isinstance(rule, str):
        if rule in label_columns:
            raise ValueError(f"{rule!r} is a label column")
        col = rule
        return walk_forward_scores(panel, folds, lambda train, test: test[col].to_numpy(dtype=float),
                                   label_columns=label_columns)
    return walk_forward_scores(panel, folds, lambda train, test: rule(test), label_columns=label_columns)


# --------------------------------------------------------------------------------------------- rank IC


def _average_ranks(v: np.ndarray) -> np.ndarray:
    order = np.argsort(v, kind="mergesort")
    sorted_v = v[order]
    ranks = np.empty(len(v), dtype=float)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> float:
    """Spearman rank correlation with average ranks for ties; NaN pairs dropped; NaN if < 3 pairs or constant."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y differ in length")
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return math.nan
    rx, ry = _average_ranks(x), _average_ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = math.sqrt(float(rx @ rx) * float(ry @ ry))
    return math.nan if den == 0 else float(rx @ ry) / den


def rank_ic_by_date(panel: pd.DataFrame, scores: pd.Series | np.ndarray, *, min_names: int = 10) -> pd.DataFrame:
    """Cross-sectional Spearman IC between score and target per decision date (columns session, date, n, ic).

    A ``pandas.Series`` of scores is aligned to ``panel.index`` by label (every panel row must have a score entry,
    possibly NaN); a plain array must have one value per panel row, in panel order.
    """
    if isinstance(scores, pd.Series):
        if not scores.index.is_unique or not panel.index.is_unique:
            raise ValueError("scores and panel need unique indices to align")
        missing = panel.index.difference(scores.index)
        if len(missing):
            raise ValueError(f"{len(missing)} panel rows have no score entry (index mismatch)")
        values = scores.reindex(panel.index).to_numpy(dtype=float)
    else:
        values = np.asarray(scores, dtype=float).reshape(-1)
        if len(values) != len(panel):
            raise ValueError(f"{len(values)} scores for {len(panel)} panel rows")
    frame = pd.DataFrame({"session": panel["session"].to_numpy(), "date": panel["date"].to_numpy(),
                          "target": panel["target"].to_numpy(dtype=float), "score": values})
    rows = []
    for (session, day), g in frame.groupby(["session", "date"], sort=True):
        ok = g["score"].notna() & g["target"].notna()
        n = int(ok.sum())
        ic = spearman(g.loc[ok, "score"], g.loc[ok, "target"]) if n >= min_names else math.nan
        rows.append({"session": int(session), "date": str(day), "n": n, "ic": ic})
    return pd.DataFrame(rows, columns=["session", "date", "n", "ic"])


def window_rank_ic(panel: pd.DataFrame, scores: pd.Series | np.ndarray, folds: Sequence[Fold], *,
                   min_names: int = 10) -> pd.DataFrame:
    """Guarded rank IC per decision date for any model, baselines included: only the test rows of ``folds``.

    Every fold is re-checked by ``validate_fold`` (sealed by ``yearly_folds``, labels inside the window, holdout
    rows only with a verified lock); forward folds carry no label and are refused. Scores of rows outside the
    folds' test blocks are ignored, so a full-panel feature column cannot reach a date whose label leaves the window.
    """
    if not panel.index.is_unique:
        raise ValueError("panel index must be unique")
    tested: list[int] = []
    for fold in folds:
        validate_fold(fold, panel)
        if fold.kind != "walk-forward":
            raise UnguardedFold(f"fold {fold.year}: only walk-forward folds have labels to score")
        tested.extend(fold.test_sessions)
    if len(tested) != len(set(tested)):
        raise ValueError("overlapping folds: a test decision appears twice")
    mask = np.isin(panel["session"].to_numpy(), tested)
    if isinstance(scores, pd.Series):
        sub_scores: pd.Series | np.ndarray = scores
    else:
        values = np.asarray(scores, dtype=float).reshape(-1)
        if len(values) != len(panel):
            raise ValueError(f"{len(values)} scores for {len(panel)} panel rows")
        sub_scores = values[mask]
    return rank_ic_by_date(panel[mask], sub_scores, min_names=min_names)


def non_overlapping(sessions: Sequence[int], horizon: int = REBALANCE_SESSIONS) -> np.ndarray:
    """Greedy keep-mask: keep a date when it is >= ``horizon`` sessions after the last kept one."""
    s = np.asarray(sessions, dtype=np.int64)
    if len(s) and np.any(np.diff(s) <= 0):
        raise ValueError("sessions must be strictly increasing")
    keep = np.zeros(len(s), dtype=bool)
    last = None
    for i, v in enumerate(s):
        if last is None or v - last >= horizon:
            keep[i] = True
            last = v
    return keep


def mean_t(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "t": None}
    mean = float(v.mean())
    if n < 2:
        return {"n": 1, "mean": mean, "sd": None, "t": None}
    sd = float(v.std(ddof=1))
    return {"n": n, "mean": mean, "sd": sd, "t": None if sd == 0 else mean / (sd / math.sqrt(n))}


def block_bootstrap_mean(values: Sequence[float] | np.ndarray, *, block_size: int = 1, samples: int = 10_000,
                         seed: int = 20261106, alpha: float = 0.05) -> dict[str, Any]:
    """Circular moving-block bootstrap of the mean over a chronological per-date series (dates = clusters)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 2:
        return {"n": n, "blockSize": block_size, "samples": samples, "ci": None, "se": None, "pValueOneSided": None}
    b = max(1, min(int(block_size), n))
    rng = np.random.default_rng(seed)
    blocks = -(-n // b)
    starts = rng.integers(0, n, size=(samples, blocks))
    idx = ((starts[:, :, None] + np.arange(b)[None, None, :]) % n).reshape(samples, -1)[:, :n]
    means = v[idx].mean(axis=1)
    est = float(v.mean())
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    # H0: mean <= 0 vs H1: mean > 0, using the bootstrap distribution re-centred at 0.
    p = float(np.mean(means - est >= est))
    return {"n": n, "blockSize": b, "samples": samples, "seed": seed, "alpha": alpha,
            "ci": [float(low), float(high)], "se": float(means.std(ddof=1)),
            "pValueOneSided": p}


def ic_summary(ic_table: pd.DataFrame, *, horizon: int = REBALANCE_SESSIONS, block_size: int = 1,
               samples: int = 10_000, seed: int = 20261106) -> dict[str, Any]:
    """Primary metric: mean IC and t over non-overlapping dates, plus a block bootstrap CI by date."""
    valid = ic_table[np.isfinite(ic_table["ic"].to_numpy(dtype=float))].sort_values("session")
    keep = non_overlapping(valid["session"].to_numpy(), horizon) if len(valid) else np.zeros(0, dtype=bool)
    chosen = valid[keep]
    out = {"datesAll": int(len(ic_table)), "datesValid": int(len(valid)), "datesNonOverlapping": int(len(chosen)),
           "from": chosen["date"].iloc[0] if len(chosen) else None,
           "to": chosen["date"].iloc[-1] if len(chosen) else None,
           **mean_t(chosen["ic"].to_numpy(dtype=float))}
    out["bootstrap"] = block_bootstrap_mean(chosen["ic"].to_numpy(dtype=float), block_size=block_size,
                                            samples=samples, seed=seed)
    out["hitRate"] = float(np.mean(chosen["ic"] > 0)) if len(chosen) else None
    return out


def paired_ic_difference(ic_a: pd.DataFrame, ic_b: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
    """IC(a) - IC(b) on the decision dates both define (non-overlapping, bootstrapped by date)."""
    merged = ic_a[["session", "date", "ic"]].merge(ic_b[["session", "ic"]], on="session", suffixes=("_a", "_b"))
    merged = merged[np.isfinite(merged["ic_a"]) & np.isfinite(merged["ic_b"])]
    diff = pd.DataFrame({"session": merged["session"], "date": merged["date"], "n": 0,
                         "ic": merged["ic_a"] - merged["ic_b"]})
    return ic_summary(diff, **kwargs)


# --------------------------------------------------------------------------------------------- Sharpe / DSR


def sharpe_moments(returns: Sequence[float] | np.ndarray) -> dict[str, Any]:
    """Per-period Sharpe ratio (no annualisation), sample skewness and non-excess kurtosis (population moments)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 2:
        raise ValueError("need at least 2 returns")
    mean = float(r.mean())
    sd = float(r.std(ddof=1))
    pop = float(r.std(ddof=0))
    skew = float(np.mean((r - mean) ** 3) / pop ** 3) if pop else 0.0
    kurt = float(np.mean((r - mean) ** 4) / pop ** 4) if pop else 3.0
    return {"n": n, "mean": mean, "sd": sd, "sharpe": mean / sd if sd else 0.0, "skew": skew, "kurtosis": kurt}


def probabilistic_sharpe(sr: float, sr_benchmark: float, n_obs: int, skew: float, kurtosis: float) -> float:
    """PSR: P(true SR > benchmark) given an observed per-period SR over n_obs returns (kurtosis non-excess)."""
    if n_obs < 2:
        raise ValueError("n_obs must be >= 2")
    var = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr * sr
    return _NORMAL.cdf((sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(max(var, 1e-12)))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum per-period SR among n_trials unskilled trials with SR variance sr_variance."""
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if sr_variance < 0:
        raise ValueError("sr_variance must be >= 0")
    if n_trials == 1 or sr_variance == 0:
        return 0.0
    z = _NORMAL.inv_cdf
    return math.sqrt(sr_variance) * ((1.0 - EULER_GAMMA) * z(1.0 - 1.0 / n_trials)
                                     + EULER_GAMMA * z(1.0 - 1.0 / (n_trials * math.e)))


def deflated_sharpe(returns: Sequence[float] | np.ndarray, *, n_trials: int, sr_variance: float) -> dict[str, Any]:
    """Deflated Sharpe ratio (Bailey & Lopez de Prado 2014) of one per-period return series."""
    m = sharpe_moments(returns)
    sr0 = expected_max_sharpe(n_trials, sr_variance)
    return {**m, "nTrials": int(n_trials), "srVariance": float(sr_variance), "expectedMaxSharpe": sr0,
            "dsr": probabilistic_sharpe(m["sharpe"], sr0, m["n"], m["skew"], m["kurtosis"]),
            "psrVsZero": probabilistic_sharpe(m["sharpe"], 0.0, m["n"], m["skew"], m["kurtosis"])}


def registry_trial_count(path: Path | str = trial_registry.REGISTRY_PATH, market: str = "HOSE") -> dict[str, Any]:
    """The protocol's N: trials of ``market`` with status run or planned, read now from the verified registry."""
    registry = trial_registry.load(path)
    count = registry.count(market, ("run", "planned"))
    return {"trials": int(count["trials"]), "entries": count["entries"], "lowerBound": count["lowerBound"],
            "statuses": count["statuses"], "market": market, "registryHead": registry.head}


SR_VARIANCE_ESTIMATOR = ("max(cross-trial sample variance (ddof 1) of the per-period Sharpe ratios of the run's "
                         "trials, mean over those trials of (1 - g3*SR + (g4 - 1)/4*SR^2)/(T - 1))")


def sr_variance(trial_returns: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """V[SR_n] for the DSR (``SR_VARIANCE_ESTIMATOR``): cross-trial variance floored at the null estimator variance."""
    if len(trial_returns) < 2:
        raise ValueError("V[SR] needs at least 2 trial series")
    moments = [sharpe_moments(r) for r in trial_returns.values()]
    srs = [m["sharpe"] for m in moments]
    cross = float(np.var(srs, ddof=1))
    floor = float(np.mean([max(0.0, 1.0 - m["skew"] * m["sharpe"] + (m["kurtosis"] - 1.0) / 4.0 * m["sharpe"] ** 2)
                           / (m["n"] - 1) for m in moments]))
    return {"estimator": SR_VARIANCE_ESTIMATOR, "crossTrial": cross, "estimatorFloor": floor,
            "value": max(cross, floor), "floored": floor > cross, "sharpes": dict(zip(trial_returns, srs))}


def protocol_registry_entry(registry: trial_registry.Registry, protocol_id: str = PROTOCOL_ID,
                            market: str = "HOSE", protocol_path: Path | str | None = None) -> dict[str, Any]:
    """The active registry entry of the protocol (source ``.../<id>.json``, status planned or run)."""
    protocol_path = Path(protocol_path) if protocol_path else PROTOCOL_PATH.parent / f"{protocol_id}.json"
    sha = protocol_sha256(protocol_path) if protocol_path.exists() else None
    found = [e for e in registry.active()
             if Path(str(e.get("source", ""))).name == f"{protocol_id}.json" and e["market"] == market
             and e["status"] in ("planned", "run")]
    if not found:
        raise ValueError(f"no planned/run {market} registry entry for protocol {protocol_id}: register it first")
    entry = found[-1]
    if entry.get("specSha256") and sha and entry["specSha256"] != sha:
        raise ValueError(f"registry entry {entry['id']} was registered for another version of {protocol_id}")
    return entry


def deflated_sharpe_from_registry(returns: Sequence[float] | np.ndarray, trial_returns: Mapping[str, Sequence[float]],
                                  *, registry_path: Path | str = trial_registry.REGISTRY_PATH,
                                  market: str = "HOSE", protocol_id: str = PROTOCOL_ID) -> dict[str, Any]:
    """DSR of ``returns`` with N from the registry and V[SR] by ``SR_VARIANCE_ESTIMATOR`` over this run's trials.

    The protocol's own entry must be registered (planned or run) with at least ``len(trial_returns)`` trials.
    """
    count = registry_trial_count(registry_path, market)
    entry = protocol_registry_entry(trial_registry.load(registry_path), protocol_id, market)
    if entry["trials"] < len(trial_returns) or count["trials"] < len(trial_returns):
        raise ValueError(f"registry entry {entry['id']} declares {entry['trials']} trials ({count['trials']} {market} "
                         f"in total) but this run evaluated {len(trial_returns)}: register every trial before "
                         "evaluating it")
    var = sr_variance(trial_returns)
    out = deflated_sharpe(returns, n_trials=count["trials"], sr_variance=var["value"])
    out["registry"] = {**count, "protocolEntry": entry["id"], "protocolEntryTrials": entry["trials"]}
    out["srVarianceDetail"] = var
    out["trialSharpes"] = var["sharpes"]
    return out


# --------------------------------------------------------------------------------------------- PBO / CSCV


def pbo_cscv(returns: np.ndarray | pd.DataFrame, *, n_blocks: int = PBO_BLOCKS) -> dict[str, Any]:
    """Probability of backtest overfitting by combinatorially symmetric cross-validation.

    ``returns`` is T x N: one chronological column of per-period returns per strategy (same periods).
    """
    names = list(returns.columns) if isinstance(returns, pd.DataFrame) else None
    m = np.asarray(returns, dtype=float)
    if m.ndim != 2 or m.shape[1] < 2:
        raise ValueError("returns must be T x N with N >= 2 strategies")
    if not np.all(np.isfinite(m)):
        raise ValueError("returns must be finite (align the strategies on common periods first)")
    if n_blocks < 2 or n_blocks % 2:
        raise ValueError("n_blocks must be an even number >= 2")
    t, n = m.shape
    rows = t // n_blocks
    if rows < 2:
        raise ValueError(f"need at least {2 * n_blocks} periods for {n_blocks} blocks, got {t}")
    dropped = t - rows * n_blocks
    blocks = m[dropped:].reshape(n_blocks, rows, n)
    sums, sq = blocks.sum(axis=1), (blocks ** 2).sum(axis=1)
    combos = np.array([[b in c for b in range(n_blocks)]
                       for c in itertools.combinations(range(n_blocks), n_blocks // 2)], dtype=float)

    def sharpe(mask: np.ndarray) -> np.ndarray:
        k = mask.sum(axis=1, keepdims=True) * rows
        mean = (mask @ sums) / k
        var = ((mask @ sq) - k * mean ** 2) / (k - 1)
        sd = np.sqrt(np.clip(var, 0.0, None))
        return np.divide(mean, sd, out=np.zeros_like(mean), where=sd > 0)

    train, test = sharpe(combos), sharpe(1.0 - combos)
    best = np.argmax(train, axis=1)
    rows_idx = np.arange(len(best))
    best_train = train[rows_idx, best]
    best_test = test[rows_idx, best]
    # average rank (1..N) of the in-sample winner among the out-of-sample Sharpe ratios
    rank = (test < best_test[:, None]).sum(axis=1) + ((test == best_test[:, None]).sum(axis=1) + 1) / 2.0
    omega = rank / (n + 1.0)
    logits = np.log(omega / (1.0 - omega))
    slope = float(np.polyfit(best_train, best_test, 1)[0]) if np.ptp(best_train) > 0 else None
    chosen = np.bincount(best, minlength=n)
    return {
        "method": "CSCV (Bailey, Borwein, Lopez de Prado & Zhu 2017), Sharpe ratio, average ranks",
        "blocks": n_blocks, "periods": t, "periodsUsed": rows * n_blocks, "droppedOldest": dropped,
        "rowsPerBlock": rows, "strategies": n, "combinations": int(len(combos)),
        "pbo": float(np.mean(logits <= 0)), "pboStrict": float(np.mean(logits < 0)),
        "gatePbo": float(np.mean(logits <= 0)), "gatePboStatistic": "pbo (logit <= 0, median rank counts as overfit)",
        "logitMean": float(logits.mean()), "logitMedian": float(np.median(logits)),
        "degradationSlope": slope, "probOosLoss": float(np.mean(best_test < 0)),
        "meanOosSharpeOfBest": float(best_test.mean()),
        "chosenInSample": {(names[i] if names else str(i)): int(c) for i, c in enumerate(chosen)},
    }


# --------------------------------------------------------------------------------------------- holdout lock


class HoldoutAlreadyRead(RuntimeError):
    """The holdout of this protocol was already opened by an earlier evaluation."""


@dataclass(frozen=True)
class HoldoutLock:
    """Read-once guard for a protocol's holdout. ``evidence_root`` holds every run directory of the protocol.

    ``evidence_root`` defaults to ``default_evidence_root(protocol_id)`` (main checkout, shared by worktrees).
    """

    protocol_id: str
    protocol_sha256: str
    evidence_root: Path | None = None

    def __post_init__(self) -> None:
        if not _ID_RE.match(self.protocol_id):
            raise ValueError(f"invalid protocol id {self.protocol_id!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", self.protocol_sha256):
            raise ValueError("protocol_sha256 must be 64 lowercase hex characters")
        root = default_evidence_root(self.protocol_id) if self.evidence_root is None else self.evidence_root
        object.__setattr__(self, "evidence_root", Path(root))

    @property
    def sentinel(self) -> Path:
        return self.evidence_root / f".holdout-{self.protocol_id}.lock"

    def existing(self) -> list[dict[str, Any]]:
        """Every lock already taken for this protocol id (sentinel and per-run records)."""
        found: list[dict[str, Any]] = []
        if self.sentinel.exists():
            found.append(self._read(self.sentinel))
        if self.evidence_root.is_dir():
            for path in sorted(self.evidence_root.glob(f"*/{LOCK_NAME}")):
                record = self._read(path)
                if record.get("protocolId") in (self.protocol_id, None):
                    found.append(record)
        return found

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            return {**record, "path": str(path)} if isinstance(record, dict) else {"path": str(path)}
        except (OSError, json.JSONDecodeError):
            return {"path": str(path), "unreadable": True}

    @property
    def locked(self) -> bool:
        return bool(self.existing())

    def acquire(self, run_dir: Path | str, *, run_id: str, registry_head: str | None = None,
                models: Iterable[str] = (), now: datetime | None = None) -> dict[str, Any]:
        """Take the lock before computing any holdout number; raises HoldoutAlreadyRead on a second attempt."""
        run_dir = Path(run_dir)
        if run_dir.resolve().parent != self.evidence_root.resolve():
            raise ValueError(f"run_dir must be a direct child of {self.evidence_root}")
        if not _ID_RE.match(run_id):
            raise ValueError(f"invalid run id {run_id!r}")
        prior = self.existing()
        if prior:
            raise HoldoutAlreadyRead(f"holdout of {self.protocol_id} already read: {prior[0].get('path')}")
        record = {"schema": "holdout-lock/v1", "protocolId": self.protocol_id,
                  "protocolSha256": self.protocol_sha256, "runId": run_id,
                  "openedAtUtc": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
                  "registryHead": registry_head, "models": list(models)}
        payload = (json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
        self.evidence_root.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            _exclusive_write(self.sentinel, payload)
        except FileExistsError as exc:
            raise HoldoutAlreadyRead(f"holdout of {self.protocol_id} already read: {self.sentinel}") from exc
        _exclusive_write(run_dir / LOCK_NAME, payload)
        return {**record, "path": str(run_dir / LOCK_NAME)}


def verify_holdout_lock(record: Mapping[str, Any], *, protocol_id: str = PROTOCOL_ID,
                        protocol_path: Path | str = PROTOCOL_PATH, evidence_root: Path | str | None = None) -> None:
    """Check that ``record`` is a lock actually acquired for the current frozen protocol (run record + sentinel).

    The lock must live in the protocol evidence root: ``<root>/<run>/holdout.lock.json`` with the sentinel
    ``<root>/.holdout-<id>.lock``, ``root`` = ``default_evidence_root(protocol_id)`` (main checkout) unless an
    ``evidence_root`` is passed explicitly. A lock taken in any other directory is refused.
    """
    if not isinstance(record, Mapping) or record.get("schema") != "holdout-lock/v1":
        raise HoldoutSealed("holdout_lock is not a holdout-lock/v1 record from HoldoutLock.acquire")
    if record.get("protocolId") != protocol_id or record.get("protocolSha256") != protocol_sha256(protocol_path):
        raise HoldoutSealed(f"holdout_lock was taken for another protocol or protocol version than {protocol_id}")
    root = Path(evidence_root) if evidence_root is not None else default_evidence_root(protocol_id)
    path = Path(str(record.get("path", "")))
    if path.name != LOCK_NAME or path.parent.parent.resolve() != root.resolve():
        raise HoldoutSealed(f"holdout lock {path} is not a run record under the protocol evidence root {root}")
    body = {k: v for k, v in record.items() if k != "path"}
    sentinel = root / f".holdout-{protocol_id}.lock"
    for p in (path, sentinel):
        try:
            on_disk = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HoldoutSealed(f"holdout lock file {p} is missing or unreadable") from exc
        if on_disk != body:
            raise HoldoutSealed(f"holdout lock file {p} does not match the record passed")


def _exclusive_write(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0))
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


# --------------------------------------------------------------------------------------------- report


def multiple_testing_report(net_excess: Mapping[str, Sequence[float]], best: str, *,
                            registry_path: Path | str = trial_registry.REGISTRY_PATH, market: str = "HOSE",
                            n_blocks: int = PBO_BLOCKS, protocol_id: str = PROTOCOL_ID) -> dict[str, Any]:
    """DSR of ``best``'s net excess series (N from the registry) and PBO over every given series.

    ``net_excess`` maps each ranked model id to its per-period net excess return, all on the same periods. The
    report states the fixed gate statistics (``gatePbo``, ``SR_VARIANCE_ESTIMATOR``) and the protocol thresholds.
    """
    if best not in net_excess:
        raise ValueError(f"best model {best!r} not among the series")
    lengths = {len(v) for v in net_excess.values()}
    if len(lengths) != 1:
        raise ValueError("all series must cover the same periods")
    matrix = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in net_excess.items()})
    dsr = deflated_sharpe_from_registry(net_excess[best], net_excess, registry_path=registry_path, market=market,
                                        protocol_id=protocol_id)
    pbo = pbo_cscv(matrix, n_blocks=n_blocks)
    return {"best": best, "dsr": dsr, "pbo": pbo, "provenance": evaluator_provenance(),
            "gates": {"dsrMin": GATE_DSR_MIN, "dsr": dsr["dsr"], "dsrPass": dsr["dsr"] >= GATE_DSR_MIN,
                      "pboMax": GATE_PBO_MAX, "pboStatistic": pbo["gatePboStatistic"], "pbo": pbo["gatePbo"],
                      "pboPass": pbo["gatePbo"] < GATE_PBO_MAX, "srVarianceEstimator": SR_VARIANCE_ESTIMATOR}}
