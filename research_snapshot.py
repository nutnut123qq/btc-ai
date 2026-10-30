"""Research-session CLI over the backend research API (EQX v2 §C2/§B).

Talks to the backend HTTP API only (``--backend`` or ``BACKEND_BASE_URL``,
default ``http://localhost:5197``); it never reads the database directly.

Commands::

    readiness --symbol BID [--as-of 2026-10-16T09:00:00Z]
    export    --symbol BID [--as-of ...] --out-dir DIR [--include-portfolio]
    journal record --symbol BID --file note.json [--snapshot DIR/snapshot.json]
    journal list   --symbol BID [--take 50]

``export`` writes ``snapshot.json`` (the backend response bytes as served),
``series-{tf}.csv`` (every stored bar with closeTimeMs <= effectiveCutoffMs,
paged from ``/api/equities/{symbol}/candles``) and ``manifest.json`` (sha256
per file). Each CSV is re-fingerprinted with the backend's canonical line
format and must match the snapshot's dataset fingerprint — a mismatch exits
non-zero and no manifest is written. ``--include-portfolio`` additionally
writes ``portfolio.json`` (position + decision, AdminGuard'd, needs
``EQX_ADMIN_KEY``); position data is never mixed into ``snapshot.json``.

The admin key is only ever sent as the ``X-Admin-Key`` header; it is never
printed or written to disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_BACKEND = "http://localhost:5197"
ADMIN_KEY_ENV = "EQX_ADMIN_KEY"
EXPORT_SCHEMA = "equity-research-export-v1"
SNAPSHOT_SCHEMA = "equity-research-snapshot-v1"
FINGERPRINT_METHOD = "sha256-lf-joined-csv-normalized-decimal-v1"
CSV_HEADER = "openTimeMs,closeTimeMs,open,high,low,close,volume"
DEFAULT_PAGE_LIMIT = 2000
MAX_PAGES = 500

EXIT_OK = 0
EXIT_HTTP = 1
EXIT_USAGE = 2
EXIT_FINGERPRINT = 3

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


class ApiError(RuntimeError):
    def __init__(self, status: int, code: str | None, message: str) -> None:
        super().__init__(f"HTTP {status}{f' {code}' if code else ''}: {message}")
        self.status = status
        self.code = code


class FingerprintMismatch(RuntimeError):
    pass


# --- HTTP ---------------------------------------------------------------------------


class Client:
    def __init__(self, base_url: str, timeout: float = 120.0, admin_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._admin_key = admin_key

    def _request(self, method: str, path: str, query: Mapping[str, Any] | None = None,
                 body: Any = None, admin: bool = False) -> bytes:
        url = self.base_url + path
        if query:
            url += "?" + urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body, ensure_ascii=False, default=_json_default).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if admin:
            if not self._admin_key:
                raise ApiError(0, "ADMIN_KEY_MISSING", f"set {ADMIN_KEY_ENV} for admin endpoints")
            headers["X-Admin-Key"] = self._admin_key
        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            code, message = None, raw.decode("utf-8", errors="replace")[:500]
            try:
                envelope = json.loads(raw)
                if isinstance(envelope, dict):
                    code = envelope.get("code") if isinstance(envelope.get("code"), str) else None
                    for key in ("message", "Message", "title"):
                        if isinstance(envelope.get(key), str):
                            message = envelope[key]
                            break
            except ValueError:
                pass
            raise ApiError(exc.code, code, message) from None
        except urllib.error.URLError as exc:
            raise ApiError(0, "BACKEND_UNREACHABLE", str(exc.reason)) from None
        except TimeoutError:  # a read timeout escapes urlopen unwrapped; it is a backend failure, not usage
            raise ApiError(0, "BACKEND_TIMEOUT", f"no response within {self.timeout:g} s") from None

    def get_bytes(self, path: str, query: Mapping[str, Any] | None = None, admin: bool = False) -> bytes:
        return self._request("GET", path, query, admin=admin)

    def get_json(self, path: str, query: Mapping[str, Any] | None = None, admin: bool = False) -> Any:
        return json.loads(self.get_bytes(path, query, admin), parse_float=Decimal)

    def post_json(self, path: str, body: Any, admin: bool = False) -> Any:
        return json.loads(self._request("POST", path, body=body, admin=admin), parse_float=Decimal)


# --- canonical series ---------------------------------------------------------------


def normalize_number(value: Any) -> str:
    """Same rule as EquityResearchSnapshotService.NormalizeDecimal: no trailing zeros, no exponent."""
    if isinstance(value, bool):
        raise TypeError("boolean is not a price/volume")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):  # only if a caller parsed without Decimal; repr keeps round-trip digits
        value = Decimal(repr(value))
    if isinstance(value, str):
        value = Decimal(value)
    if not isinstance(value, Decimal):
        raise TypeError(f"unsupported numeric type {type(value).__name__}")
    text = format(value.normalize(), "f")
    return "0" if text in ("-0", "0") else text


def series_line(bar: Mapping[str, Any]) -> str:
    return ",".join(
        [str(int(bar["openTimeMs"])), str(int(bar["closeTimeMs"]))]
        + [normalize_number(bar[k]) for k in ("open", "high", "low", "close", "volume")]
    )


def fingerprint(lines: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def fetch_series(client: Client, symbol: str, timeframe: str, cutoff_ms: int,
                 page_limit: int = DEFAULT_PAGE_LIMIT) -> list[dict[str, Any]]:
    """Every stored bar with closeTimeMs <= cutoff, ascending (pages backwards by endTimeMs)."""
    by_open: dict[int, dict[str, Any]] = {}
    end = cutoff_ms
    for _ in range(MAX_PAGES):
        page = client.get_json(
            f"/api/equities/{urllib.parse.quote(symbol)}/candles",
            {"timeframe": timeframe, "limit": page_limit, "endTimeMs": end},
        )
        if not isinstance(page, list):
            raise ApiError(200, "CANDLES_PAYLOAD_INVALID", "candles response is not a list")
        for bar in page:
            if int(bar["closeTimeMs"]) <= cutoff_ms:
                by_open[int(bar["openTimeMs"])] = bar
        if len(page) < page_limit:
            break
        end = min(int(b["openTimeMs"]) for b in page) - 1
    else:
        raise ApiError(200, "CANDLES_TOO_MANY_PAGES", f"more than {MAX_PAGES} pages for {timeframe}")
    return [by_open[k] for k in sorted(by_open)]


# --- commands -----------------------------------------------------------------------


def _write_bytes(path: Path, data: bytes) -> dict[str, Any]:
    path.write_bytes(data)
    return {"path": path.name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _dumps(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n").encode("utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value) if value != value.to_integral_value() else int(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def cmd_readiness(client: Client, args: argparse.Namespace) -> int:
    payload = client.get_json(
        f"/api/equities/{urllib.parse.quote(args.symbol)}/research-readiness", {"asOf": args.as_of}
    )
    sys.stdout.write(_dumps(payload).decode("utf-8"))
    return EXIT_OK


def export(client: Client, symbol: str, as_of: str | None, out_dir: Path,
           include_portfolio: bool = False) -> dict[str, Any]:
    """Export snapshot + verified series (+ optional portfolio). Returns the manifest."""
    if include_portfolio and not client._admin_key:  # fail before any file is written
        raise ApiError(0, "ADMIN_KEY_MISSING", f"--include-portfolio requires {ADMIN_KEY_ENV}")
    sym_path = urllib.parse.quote(symbol)
    raw = client.get_bytes(f"/api/equities/{sym_path}/research-snapshot", {"asOf": as_of})
    snapshot = json.loads(raw, parse_float=Decimal)
    if snapshot.get("contractVersion") != SNAPSHOT_SCHEMA:
        raise ApiError(200, "SNAPSHOT_SCHEMA_MISMATCH", f"unexpected contractVersion {snapshot.get('contractVersion')!r}")
    cutoff = int(snapshot["effectiveCutoffMs"])
    canonical_symbol = snapshot["symbol"]

    verified: list[dict[str, Any]] = []
    csv_payloads: list[tuple[str, bytes]] = []
    for series in snapshot.get("series", []):
        tf = series["timeframe"]
        if series.get("fingerprintMethod") != FINGERPRINT_METHOD:
            raise FingerprintMismatch(f"{tf}: unsupported fingerprint method {series.get('fingerprintMethod')!r}")
        bars = fetch_series(client, canonical_symbol, tf, cutoff, int(series.get("pageLimit") or DEFAULT_PAGE_LIMIT))
        lines = [series_line(b) for b in bars]
        actual = fingerprint(lines)
        expected = series["fingerprint"]
        dataset = (snapshot.get("dataset") or {}).get(tf) or {}
        if actual != expected or len(lines) != int(series["barCount"]) or dataset.get("sha256") != expected:
            raise FingerprintMismatch(
                f"{tf}: fetched {len(lines)} bars sha {actual[:12]} != snapshot {int(series['barCount'])} bars "
                f"sha {str(expected)[:12]} (data changed since the snapshot, or a paging/format bug)"
            )
        verified.append({"timeframe": tf, "barCount": len(lines), "fingerprint": actual, "verified": True})
        csv_payloads.append((f"series-{tf}.csv", ("\n".join([CSV_HEADER, *lines]) + "\n").encode("utf-8")))

    portfolio: dict[str, Any] | None = None
    if include_portfolio:
        portfolio = {
            "schemaVersion": "equity-research-portfolio-v1",
            "symbol": canonical_symbol,
            "fetchedAtUtc": datetime.now(timezone.utc).isoformat(),
            "note": "position-aware data (AdminGuard); never merged into snapshot.json",
            "snapshotId": snapshot["snapshotId"],
            "position": client.get_json(f"/api/equities/{sym_path}/position", admin=True),
            "decision": client.get_json(f"/api/equities/{sym_path}/decision", admin=True),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    files = [_write_bytes(out_dir / "snapshot.json", raw)]
    for name, data in csv_payloads:
        files.append(_write_bytes(out_dir / name, data))
    if portfolio is not None:
        files.append(_write_bytes(out_dir / "portfolio.json", _dumps(portfolio)))
    manifest = {
        "schemaVersion": EXPORT_SCHEMA,
        "symbol": canonical_symbol,
        "snapshotId": snapshot["snapshotId"],
        "snapshotSha256": snapshot["snapshotSha256"],
        "requestedAsOf": snapshot["requestedAsOf"],
        "requestedAsOfMs": snapshot["requestedAsOfMs"],
        "effectiveCutoffMs": cutoff,
        "backend": client.base_url,
        "exportedAtUtc": datetime.now(timezone.utc).isoformat(),
        "fingerprintMethod": FINGERPRINT_METHOD,
        "series": verified,
        "files": files,
    }
    (out_dir / "manifest.json").write_bytes(_dumps(manifest))
    return manifest


def cmd_export(client: Client, args: argparse.Namespace) -> int:
    manifest = export(client, args.symbol, args.as_of, Path(args.out_dir), args.include_portfolio)
    print(f"snapshot {manifest['snapshotId']} -> {args.out_dir}")
    for item in manifest["files"]:
        print(f"  {item['path']}  {item['bytes']} B  sha256 {item['sha256']}")
    return EXIT_OK


def _load_note(path: Path, snapshot_path: Path | None) -> dict[str, Any]:
    note = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(note, dict):
        raise ValueError("note file must contain a JSON object")
    if snapshot_path is not None:
        snap = json.loads(snapshot_path.read_text(encoding="utf-8"))
        note.setdefault("snapshotId", snap["snapshotId"])
        note.setdefault("snapshotSha256", snap["snapshotSha256"])
        note.setdefault("requestedAsOf", str(snap["requestedAsOfMs"]))
        if snap.get("signalBar"):
            note.setdefault("signalBarOpenTimeMs", snap["signalBar"]["openTimeMs"])
    return note


def cmd_journal_record(client: Client, args: argparse.Namespace) -> int:
    note = _load_note(Path(args.file), Path(args.snapshot) if args.snapshot else None)
    result = client.post_json(f"/api/equities/{urllib.parse.quote(args.symbol)}/research-notes", note, admin=True)
    sys.stdout.write(_dumps(result).decode("utf-8"))
    return EXIT_OK


def cmd_journal_list(client: Client, args: argparse.Namespace) -> int:
    result = client.get_json(
        f"/api/equities/{urllib.parse.quote(args.symbol)}/research-notes", {"take": args.take}, admin=True
    )
    sys.stdout.write(_dumps(result).decode("utf-8"))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EQX research snapshot / journal CLI (backend API only).")
    parser.add_argument("--backend", default=os.getenv("BACKEND_BASE_URL") or DEFAULT_BACKEND)
    parser.add_argument("--timeout", type=float, default=180.0, help="per-request timeout (s)")
    sub = parser.add_subparsers(dest="command", required=True)

    readiness = sub.add_parser("readiness", help="per-purpose research readiness")
    readiness.add_argument("--symbol", required=True)
    readiness.add_argument("--as-of", default=None, help="epoch ms or ISO-8601 with offset; default now")
    readiness.set_defaults(func=cmd_readiness)

    exp = sub.add_parser("export", help="snapshot.json + verified series CSV + manifest")
    exp.add_argument("--symbol", required=True)
    exp.add_argument("--as-of", default=None)
    exp.add_argument("--out-dir", required=True)
    exp.add_argument("--include-portfolio", action="store_true",
                     help=f"also write portfolio.json (needs {ADMIN_KEY_ENV})")
    exp.set_defaults(func=cmd_export)

    journal = sub.add_parser("journal", help="session research journal (admin)")
    jsub = journal.add_subparsers(dest="journal_command", required=True)
    record = jsub.add_parser("record", help="POST a research note")
    record.add_argument("--symbol", required=True)
    record.add_argument("--file", required=True, help="note JSON (EquityResearchNoteRequest)")
    record.add_argument("--snapshot", default=None,
                        help="snapshot.json to fill snapshotId/sha/requestedAsOf/signal bar when absent")
    record.set_defaults(func=cmd_journal_record)
    listing = jsub.add_parser("list", help="list research notes")
    listing.add_argument("--symbol", required=True)
    listing.add_argument("--take", type=int, default=50)
    listing.set_defaults(func=cmd_journal_list)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = Client(args.backend, args.timeout, os.getenv(ADMIN_KEY_ENV) or None)
    try:
        return args.func(client, args)
    except FingerprintMismatch as exc:
        print(f"error: fingerprint mismatch: {exc}", file=sys.stderr)
        return EXIT_FINGERPRINT
    except ApiError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE if exc.code == "ADMIN_KEY_MISSING" else EXIT_HTTP
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
