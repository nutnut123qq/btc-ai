"""VN30-WEB: export the data of the public site (contract vn30-web-data/v2, evidence/vn30-web-20261103/data-contract.md).

Reads the newest VN30-SCREEN table (<screen-dir>/<session>.json) and the backend HTTP API, GET only (--backend or
BACKEND_BASE_URL): /api/equities (labels) and per ticker /candles (1d, 260 bars), /technical?bars=1 (latest 1d
indicators, relative strength) and, with X-Admin-Key from EQX_ADMIN_KEY, research-notes (take NOTES_TAKE, the
backend's cap; every mode and horizon counts against it) and research-notes/direction-scores (take SCORES_TAKE items;
horizons[] always covers the whole journal). Writes meta.json, screen.json and symbols/<TICKER>.json to --out-dir.

Everything written is published. Every output object is built field by field from the contract and no backend
object is copied whole: notes carry actionView/actionBasis/probability/scenarios/snapshot ids, which never leave.

Units: every return is a fraction (0.05 = +5 %). relativeStrength["20"/"60"] is the `excess` of the VNINDEX
benchmark (instrument minus VNINDEX close-to-close return over 20/60 aligned 1d sessions), exported only while that
benchmark's status is "ok". notes are the newest forward notes with horizonSessions HORIZON that pass market_only
(below), at most NOTE_LIMIT of them, joined with their direction scores by note id; a superseded note (a later one on
a bar already taken) is dropped. The backend status "resolved" is exported as "scored"; a direction, status or
pendingReason outside the contract makes the notes part malformed (a dropped note is read only as far as its join:
its direction-score item must exist and match it). returnPct is direction-score-v1's target close / signal close - 1.
score is the H=20 total over the ticker's whole journal, copied from the direction-scores horizons[] entry with
horizonSessions HORIZON (resolved, hits, baselinePrior.hits, status) and never computed here, so it does not depend
on which notes are exported: it counts a note that market_only drops once the backend has scored it, and the notes
list can be shorter than the set of notes the backend scores (NOTE_LIMIT cuts it too).

A note is exported only when its actionBasis is market-only or insufficient-input and its actionView is wait or none
(market_only: PUBLIC_BASES, PUBLIC_VIEWS). Any other value, a missing field or a value that is not a string drops the
whole note, like a superseded one: it has no row, takes no slot of NOTE_LIMIT (the list goes on to the next note that
passes and is not superseded, so the newest note listed is the newest such note) and is not counted as a withheld
thesis; the summary line counts it in notesDropped. A note written in a session that knew the user's holding (hold,
reduce, sell, buy-more; position-aware) can carry it in its thesis, and both repos are public, so even its row with
the thesis hidden could tell a reader that the ticker had a call tied to a position (user decision of 2026-11-04,
VN30-WEB r5; r4 exported that row with its thesis withheld). When the dropped note was the first one on its bar, the
daily note written later on that bar is superseded, so that bar has no exported note at all (when it is the newest
bar, the newest exported note is from an earlier bar). The check reads only these two fields, so a
note recorded with wait and market-only in a session that knew the holding passes it and only the keyword filter
below reads its thesis (the skill writes hold only when the user says they hold the stock; wait with market-only is
its default). The previous run's files do not carry actionView/actionBasis, so they cannot be filtered again: from
VN30-WEB r5 on, a kept note passed this check when it was first exported (Parts); r4 to r4c exported the row of a note
that failed it, with its thesis withheld, so from r4 on, a kept thesis passed this check when it was first exported,
and the theses exported before r4 all came from wait/market-only notes.

The thesis of an exported note is published verbatim once it passes a fail-closed keyword filter that reads the
whole text. The filter is a denylist (PRIVATE_TEXT): the listed words about positions, holdings, quantities, cost
basis, money, settlement (T+n),
stops, orders, trade set-ups and trading actions, the first person (we, our, my, I'm) and their Vietnamese
counterparts with and without diacritics. An English word matches in the forms its pattern lists (reduce, reduces,
reduced, reducing), which are not all the forms of every word ("stopping", "sidelining", "positioned", "going flat"
and "went flat" are not listed). The words of a listed phrasal verb or compound match with a space or a dash between
them (stopped out, stopped-out; scale in, stand aside, take profit, new money, falling knife, risk reward, break even,
under water, margin call, net asset value, average price, target price, off the table); these match only with spaces:
on margin, at an average of, on the books, basis of/is <number>, thousands of shares, I am/have/own, profits
taken/booked, cut the loss, an order, place an order, order at <number>, long above <number>, a long at, long from,
already long, already in the stock, short the bounce, fade the rally, lock in gains, book profits, fill at, pick up
shares, close the trade, keep the stock, existing investors/owners, dot com (lock in gains, pick up shares and cut the
loss also with a dash right after their first word), and so does every Vietnamese phrase; "sideline" matches only as
one word or with a dash. "break even" right before lower, higher, further, more or deeper is chart prose ("broke even
lower") and does not match; written with a dash or as one word, the noun always matches ("the break-even higher",
"breakeven higher"). So the spaced form before those words passes even when it is about a cost ("broke even more than
a week ago").
Amounts: between a number and the word after it, and between k, m or b and the currency word, there may be nothing,
one space or one dash, and before a VND currency word also Vietnamese, Vietnam, Viet Nam, Việt Nam or VN and one
space (1.2B Vietnam dong, 63,000,000 Việt Nam đồng); before k, m or b only an optional space (63 m dong, VND 120 m,
1.2 B USD); after VND, vnđ, ₫, USD or $ an optional space (after VND, vnđ or ₫ also an optional colon first). Caught:
a number and a scale word (million, mil, mio, mn, billion, bn, triệu, tỷ, tỉ, củ, tr: 63 million, a 63-million
reserve); a number, k, m or b and a VND currency word (VND, vnđ, đ, ₫, đồng, dong, dongs: 50k VND, 120m dong, a
900k-dong buffer, 1.2B VND); a VND amount of a million or more ("," or "." or a space between the thousands, or 7 to
16 digits; up to two decimals) and VND, vnđ, đ, ₫, dong or dongs (63,000,000.00 VND, 63 000 000 VND, 63000000-dong,
63,000,000 Vietnam dong; here đ also stands for any word that starts with it, such as đồng, while after k, m or b it
counts only as a word of its own); VND, vnđ or ₫ before 7 to 15 digits, a number of a million or more with these
separators or a number and k, m or b (VND: 63,000,000, VND 63 000 000, VND 120000000, VND 120m, VND1.2B); $ or USD
before a number; a number, k, m or b if any, and USD or dollar(s) (USD 50k, 2.5m dollars, 1.2B USD, a 2,500-dollar
reserve). A price such as 18,750 VND, 118500 VND or 61,000 dong is not an amount (a VN30 price has at most six
digits: 999999 VND is published, 1200000 VND is not), and a currency word that does not stand next to the number is
not caught ("63,000,000 in VND terms").
Other patterns: a share count (3,000 shares, 3,000-share, 3 nghìn cp, thousands of shares); "<number> <ticker>"
(3,000 MBB, 3K MBB, 3 nghìn MBB, also across a full stop; not after a dash, so not the end of a date, and not one of
the indicator abbreviations SMA, EMA, RSI, ATR, OBV, POC, VAL, VAH, nor VND); a number that ends with a comma is read
as a list ("+6.8, VCB +5.1" passes) except in three cases, which withhold: the ticker after it has a count of its own,
after one space or after one or two of ":", ";", ",", "(", "[", "~", "×", "=" and "≈" (each with an optional space
on either side), " x ", one or two dashes with a space on each side, about, roughly, around, some, approximately,
approx., approx, nearly and almost ("MBB 3,000, BID 1,500", "MBB: 3,000, BID: 1,500", "BID 1,200, MBB (2,600)", "BID
1,200, MBB: ~2,600", and so "MBB 18900, BID 31700" and "at 18,750, BID 1d" too); the comma is glued to the ticker
("2,600,MBB"); or the number has no sign and the ticker is followed by the end of the text or by a character other
than a space, a comma or an apostrophe, whitespace at the end of the text counting as the end ("3,000, MBB."; "+6.8,
VCB", "18,550, MBB bounced", "In 2025, MBB, BID and VCB" and "18900, MBB's lowest" pass). A sign, "'s", at, near,
of, with or another word or symbol after the ticker keeps the list reading ("MBB at 18900, BID at 31700" passes), so
a count written first and followed by a word is not caught ("3,000, MBB and 1,500, BID at the close"). Also caught:
URLs (scheme://, www.), host names (label.tld, localhost, a Windows default name); dotted IPv4 addresses (four
numbers), IPv6 addresses written with "::" or in eight groups, host:port, a port of
4 or 5 digits after a colon that does not follow a letter, a digit or another colon (at :5197) and the word port; file
paths (a backslash, a drive letter, a path that starts with /, ./, ../ or ~/ and has a second slash, a listed root
such as /home or evidence/, a listed file extension, a few folder names); e-mail addresses (@); hex strings and digit
runs (16 or more hex characters, 8 to 15 with a letter and a digit, 12 or more digits, 0x and 8 hex digits, a run of
32 or more letters, digits, +, /, =, _ or -); internal ids and secrets. A few
chart phrases are blanked first, each only in its chart context ("range position 0.45", "selling pressure", "the
sell-off", "support holds": TECHNICAL_PHRASES, CHART_BEFORE). One match anywhere withholds the whole thesis (null), and
so does any character outside the short allowlist the lists can read (READABLE, _unreadable: printable ASCII, Latin-1,
Vietnamese and a little typography; so a format or bidi control, a thin space, a Cyrillic, Greek, IPA or small-capital
look-alike, an enclosed letter or a box-drawing dash withholds). When in doubt it withholds, so a purely technical
thesis can be withheld too ("volatility reduced", "price is holding above SMA50"). Being a list of keywords it does
not catch every way of phrasing advice or a position: advice written around the listed words passes almost always
(cold-check r3: 25 of 25 such English sentences, still 25 after r4 and r4b), and so do natural position sentences
without a listed word (11 of the 17 position sentences of that cold-check still pass after r4 and r4b; of the 60 new
leak sentences of cold-check r4's privacy lens, 25 still pass after r4b; in evidence/vn30-web-20261103:
coldcheck-r3.md, r4/W-exporter/probe-r4.log, review-fix/probe-r4-after-review.log, r4b/W-exporter/logs/claims-r4b.log
and probe-r4b.log). Host names without a dot, an abbreviated or decimal IPv4 address ("127.1:5197"),
relative paths without an extension, short hex ids, amounts written in words ("sixty-three million dong") and words
glued to "_" or a digit ("buy18600") are not guaranteed to be caught. Writers following
ai/skills/equity-technical/SKILL.md v7 or later are told not to write any of these (Step 4); notes written under an
earlier version of the skill were not. A published thesis longer than THESIS_MAX characters is cut to THESIS_MAX - 1
characters followed by CUT.

Parts: candles, technical (indicators + relativeStrength) and notes (notes + score) are read independently. A part
that fails (an HTTP error, a 429 after the retry, a malformed payload or, for notes, a missing admin key) keeps the
previous symbols/<TICKER>.json's values when the previous meta.json has schema SCHEMA and those values validate; a
candles failure keeps the whole previous file and reads nothing else for that ticker. Otherwise the part is written
empty (null, [], score null). A truthful empty answer is never replaced: no 1d indicator row, a VNINDEX benchmark that
is not "ok", or 400 UNSUPPORTED_CAPABILITY (a part the backend does not serve for that ticker). meta.unavailable
lists every part not read fresh, with a plain Vietnamese reason that names no internal code (UNSUPPORTED, HTTP_FAILED,
MALFORMED, NO_INDICATORS, NO_STRENGTH, NO_KEY); a failure's reason ends with KEPT or NOTHING_KEPT. The details (HTTP
status and envelope code, exception name and message, VNINDEX benchmark status) go to stderr only, one ASCII line per
meta.unavailable entry: "unavailable <SYMBOL> <part>: <detail>".

/technical shares the backend's "expensive" limit (10 per minute per client, no queue) with the daily analysis run
that writes the notes: its calls start at least TECH_GAP_S apart (at most 4 per minute), and an HTTP 429 on any call
waits RETRY_AFTER_S and retries once. A file is rewritten only when its bytes change, and meta.json keeps its
generatedAtUtc when nothing else changed. Exit 1 when the backend is unreachable (nothing is written), 2 when the
screen file is missing or invalid. The summary line on stdout (ASCII) counts the unavailable parts, the parts kept
from the previous run, the theses the filter withheld from exported notes (thesesWithheld) and the notes market_only
dropped (notesDropped: a note dropped as superseded, or one past the NOTE_LIMIT notes taken, is not counted, and a kept
notes part counts none). The nightly wrapper writes this line to its local log only; no public file carries either
count.

Usage (from ai/, venv python):
  python export_vn30_web.py [--out-dir DIR] [--screen-dir DIR] [--backend URL]
"""
from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from vn30_screen import DEFAULT_BACKEND, session_of
from vn30_screen import DEFAULT_OUT as DEFAULT_SCREEN_DIR

SCHEMA = "vn30-web-data/v2"
SCREEN_SCHEMA = "vn30-screen/v1"
SOURCE = "VCI, giá điều chỉnh, qua backend của dự án"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "web" / "data"
CANDLE_LIMIT = 260
HORIZON = 20  # the site text, the daily task and the scoring copy are all 20 sessions
NOTE_LIMIT = 60
NOTES_TAKE = 200  # EquityResearchNoteService.ListAsync clamps take to 1..200
SCORES_TAKE = 500  # EquityDirectionScoringService.MaxTake: take cuts items only, never horizons[]
THESIS_MAX = 1500
CUT = chr(0x2026)  # U+2026 HORIZONTAL ELLIPSIS, one character
TECH_GAP_S = 15.0  # Program.cs "expensive": fixed window of 10 per minute, QueueLimit 0
RETRY_AFTER_S = 60.0
KEPT = "; giữ dữ liệu lượt trước"
NOTHING_KEPT = "; chưa có dữ liệu lượt trước"
# meta.unavailable reasons: shown on the public site, so plain Vietnamese and no internal code (details go to stderr)
UNSUPPORTED = "backend chưa hỗ trợ mã này"  # 400 UNSUPPORTED_CAPABILITY: never kept, so never a suffix
HTTP_FAILED = "lỗi HTTP {status}"
MALFORMED = "phản hồi sai định dạng"
NO_INDICATORS = "backend chưa có chỉ báo 1d"
# the VNINDEX benchmark is not "ok": "unavailable" (not ingested, too few aligned sessions) or "stale" (VNINDEX or the
# aligned window ends before the ticker's newest session), EquityContextService.Compare
NO_STRENGTH = "dữ liệu VNINDEX chưa khớp tới phiên mới nhất của mã"
NO_KEY = "thiếu quyền đọc nhật ký nhận định"
BANKS = frozenset({"MBB", "BID", "VCB", "CTG", "ACB", "HDB", "LPB", "SHB", "SSB", "STB", "TCB", "VIB", "VPB"})
INDICATORS = ("sma50", "sma200", "ema12", "ema26", "ema50", "ema200", "rsi14", "macd", "macdSignal",
              "macdHistogram", "bollingerUpper", "bollingerMiddle", "bollingerLower", "atr14")
LOOKBACK = ("3", "6", "12")
RS_WINDOWS = ("20", "60")
TICKER = re.compile(r"[A-Z0-9]{2,10}")  # also a file name: no path characters
DIRECTIONS = frozenset({"up", "down", "sideways", "no-view"})
SUPERSEDED = "superseded-not-scored"
STATUS_OUT = {"resolved": "scored", "pending": "pending", "no-view": "no-view",
              "out-of-window-not-scored": "out-of-window-not-scored",
              "window-undetermined-not-scored": "window-undetermined-not-scored"}
NOTE_STATUSES = frozenset(STATUS_OUT.values())
PENDING_REASONS = frozenset({"immature", "target-bar-missing", "calendar-coverage-expired", "signal-bar-missing"})
# The only actionBasis and actionView values of a note that may be exported at all (contract r4 X1; from r5, D1, a
# note with any other value is dropped whole, row and thesis). The backend accepts actionBasis position-aware,
# market-only, insufficient-input and actionView buy-more, hold, reduce, sell, wait, none (EquityResearchNoteService.cs
# ActionBases, ActionViews). A note written in a session that knew the user's holding (hold, reduce, sell, buy-more;
# position-aware) can carry it in its thesis, and even its row would hint at a holding, so no part of it leaves.
PUBLIC_BASES = frozenset({"market-only", "insufficient-input"})
PUBLIC_VIEWS = frozenset({"wait", "none"})
SAMPLE_STATUSES = frozenset({"no-data", "insufficient-sample", "ok"})
SYMBOL_FIELDS = ("symbol", "label", "isBank", "candles", "indicators", "relativeStrength", "notes", "score")
INDICATOR_FIELDS = ("date", "close", *INDICATORS)
NOTE_FIELDS = ("signalDate", "horizonSessions", "direction", "thesis", "status", "pendingReason", "targetDate",
               "returnPct", "hit")
SCORE_FIELDS = ("scored", "hits", "baselineHits", "sampleStatus")

# Chart phrases blanked before the privacy check, each only where the writer's chart language puts it (CHART_BEFORE
# reads what stands right before the phrase), so that a blanked phrase cannot take a private word along:
# - frac: a range position as a fraction ("position 0.45"; never a percentage, "position: 30%" is a weight) at the
#   start of the text or of a clause ("(", ", ", "; ", ": ", ". ", "i.e. "), after "is", "at", "of" or
#   "<N>-day|session|bar" ("60-day position 0.34"), never after another word ("build a small position 0.5");
# - rpos: "<range|band|channel> position" with its fraction, in the same places, after "and", "was", "with" or "its",
#   and after an article that stands in one of these places ("The range position is 0.45", "at a range position of
#   0.45"; never "open a range position of 0.5");
# - inrange: "position (with)in [the|its] <N>-session range" (band, channel, window; bar, day) followed by its
#   fraction ("... range 0.255", "... range is 0.31") or by ")", in the places of frac and after "and", "its",
#   "price" or "'s" (a trade has no fraction: "stay patient and position in the 20-session range near its low");
# - press: "<buying|selling> <pressure|climax|volume|flow|wave|momentum|exhaustion>" (never "buying power", "buying
#   interest" or "buying activity", which can be the writer's own);
# - selloff: "sell-off" as a noun only: after "a", "the", "its", "this", "that", "(", a date or a percentage, and
#   before ",", ";", ")", ".", "on", "broke", "of" or the end ("Then sell-off half at 20,500" and "a partial sell-off
#   near 20,500" withhold);
# - holds: "<support|resistance|floor|level|low|zone|range> holds".
# Every other "position", "buying", "selling", "sell" or "holds" still withholds. By then whitespace is a single
# space and every dash the filter reads is "-".
_FRACTION = r" ?[:=]? ?0?\.\d+(?!\d|[.,]\d)"
_LINK = r"(?: (?:is|was|of|at|near|around))?"
TECHNICAL_PHRASES = re.compile(
    r"(?P<frac>\bposition" + _FRACTION + ")"
    r"|(?P<rpos>\b(?:range|band|channel) position" + _LINK + _FRACTION + ")"
    r"|(?P<inrange>\bposition (?:with)?in (?:(?:the|its) )?\d{1,3}-(?:session|bar|day) (?:range|band|channel|window)"
    r"(?:" + _LINK + _FRACTION + r"|(?=\))))"
    r"|(?P<press>\b(?:buying|selling) (?:pressure|climax|volume|flows?|waves?|momentum|exhaustion)\b)"
    r"|(?P<selloff>\bsell-?offs?(?=[,;)]|\.(?!\d)| ?\Z| (?:on|broke|of)\b))"
    r"|(?P<holds>\b(?:support|resistance|floor|level|low|zone|range) holds\b)",
    re.IGNORECASE)
_START = r"\A|\(|[(,;:.] "  # the start of the text or of a clause
CHART_BEFORE = {  # what may stand right before each phrase; a phrase found anywhere else is kept (and withholds)
    "frac": re.compile(r"(?:" + _START + r"|\b(?:is|at|of) |\d-(?:day|session|bar) )\Z", re.IGNORECASE),
    "rpos": re.compile(r"(?:" + _START + r"|\b(?:is|was|at|of|with|and|its) |\d-(?:day|session|bar) "
                       r"|(?:" + _START + r"|\b(?:is|was|at|of|with|and) )(?:a|an|the) )\Z", re.IGNORECASE),
    "inrange": re.compile(r"(?:" + _START + r"|\b(?:is|at|of|and|its|price) |['’]s )\Z", re.IGNORECASE),
    "selloff": re.compile(r"(?:\(|\b(?:a|the|its|this|that) |\b\d{4}-\d{2}-\d{2} |\d% )\Z", re.IGNORECASE),
}


def _chart_phrase(m: re.Match) -> str:
    """A space for a chart phrase in its chart context (TECHNICAL_PHRASES, CHART_BEFORE), else the phrase unchanged."""
    before = CHART_BEFORE.get(m.lastgroup)
    if before is None or before.search(m.string, max(0, m.start() - 40), m.start()):
        return " "
    return m.group(0)


# Fail-closed denylist: one match anywhere withholds the whole thesis. A listed word matches in the forms its pattern
# lists (add/adds/added/adding; not every form of every word), and the words of a phrasal verb or compound with a
# space or a dash between them (stopped out, stopped-out; "sideline" without the space), but many phrases only with
# spaces (on the books, fill at, keep the stock; the module docstring lists which phrases take a dash and which do
# not), so a purely technical sentence can lose its thesis too ("MBB added 1.2%", "the stock sold off");
# "held" is deliberately not listed ("support held" is everyday chart language), nor a bare "stopped" or
# "accumulation". It does not catch every phrasing of advice or of a position (see the module docstring). The
# patterns are lowercase: they read the lowercased probe (_private). Every repetition is bounded or delimited, so a
# 4000-character thesis is scanned in linear time.
# The VND currency word after an amount may follow the country's name and one space ("1.2B Vietnam dong", "63,000,000
# Việt Nam đồng"): the same words in the k, m or b amount and in the amount of a million or more below.
_NATION = r"(?:(?:vietnamese|vietnam|viet nam|việt nam|vn) )?"
PRIVATE_PATTERNS = (
    # positions, holdings, quantities, cost basis, money and settlement
    r"\bpositions?\b", r"\bholding", r"holders?\b", r"\bholds?\b", r"\bportfolios?\b", r"\bexposure\b",
    r"\ballocat", r"\b(?:over|under)weight\b", r"\bcash\b", r"\bbudget", r"\bcapital\b", r"\bcosts?\b",
    r"\bmargin[- ](?:loans?|call(?:s|ed|ing)?|accounts?|debt|trading|ratio|requirements?|balances?|financ(?:ed|ing))\b",
    r"\bon margin\b",
    # a cost basis: purchase, entry, cost and buy are listed words of their own ("average purchase price"), and the
    # volume-weighted average price is a chart line
    r"\b(?:avg\.?|(?<!volume[- ]weighted )average|acquisition)[- ]?(?:(?:per[- ])?(?:share|fill|execution|unit)[- ])?"
    r"prices?\b",
    r"\bat an average (?:price )?of\b",
    # "broke even lower" (further, ...) is chart prose: only the verb written with a space; "break-even" and "breakeven"
    # (the noun, as in r3) always match ("the break-even higher up at 21,450")
    r"\b(?:break(?:s|ing)?|broke)(?:-?even\b| even\b(?! (?:lower|higher|further|more|deeper)\b))", r"\bp ?[&/] ?l\b",
    r"\bpnl\b", r"\bunreali[sz]ed\b", r"\bunder[- ]?water\b",
    r"\bpa(?:y|ys|ying|id)\b", r"\bown(?:s|ed|ing)\b", r"\bon (?:the|its|their) books\b", r"\bnav\b",
    r"\bnet[- ]asset[- ]values?\b", r"\bstakes?\b", r"\bbasis (?:is |was |of |at |near )?\d",
    r"\b(?:full|half|normal|starter|initial|partial|double|quarter|planned|usual|max|maximum|trade|lot)"
    r"[- ]size[sd]?\b",
    r"\b(?:hundreds?|thousands?|millions?|billions?) (?:of )?(?:shares?|units?|lots?)\b",
    # a number read from its first digit (not after a digit, a comma or "<digit>."), then a share count or an amount: a
    # scale word right after it or one space or dash apart ("a 63-million reserve"), or k, m or b (an optional space
    # before: "63 m dong") and, right after it or one space or dash apart, a VND currency word, also after _NATION
    # ("120m dong", "a 900k-dong buffer", "1.2B VND", "1.2B Vietnam dong")
    r"\d(?<![\d,]\d)(?<!\d\.\d)[\d,.]{0,40}(?: ?(?:k|m|nghìn|ngàn)?[ -]?(?:shares?|units?|lots?|cp|cổ)\b"
    r"|[ -]?(?:million|mil|mio|mn|billion|bn|triệu|tỷ|tỉ|củ|tr)\b| ?(?:k|m|b)[ -]?" + _NATION
    + r"(?:vnd|vnđ|đồng|đ\b|₫|dongs?\b))",
    # a VND amount of a million or more: with ",", "." or a space between the thousands, or as 7 to 16 digits without
    # them (12 or more are a digit run anyway; a VN30 price has at most 6: "118500 VND", "VND 18750" and "61,000 dong"
    # are published), with up to two decimals, then right after it or one space or dash apart a VND currency word ("đ"
    # also starts "đồng"), also after _NATION; or after VND, vnđ or ₫, an optional colon and an optional space: 7 to 15
    # digits, two or more thousands groups, or a number and k, m or b ("VND 120m", "VND1.2B"; million, bn... after a
    # number withhold anyway)
    r"(?:\d{1,3}(?:[,. ]\d{3}){2,6}|\d(?<!\w\d)\d{6,15})(?:[.,]\d{1,2})?[ -]?" + _NATION + r"(?:vnd|vnđ|đ|₫|dongs?\b)",
    r"(?:\bvnd|\bvnđ|₫):? ?(?:\d{7,15}|\d{1,3}(?:[,. ]\d{3}){2,6}|\d[\d,.]{0,20} ?[kmb]\b)",
    # a dollar amount: $ or USD (an optional space after) before a number; or a number, then k, m or b if any (an
    # optional space before), then right after it or one space or dash apart USD or dollar(s)
    r"\$ ?\d", r"\busd ?\d", r"\d(?: ?[kmb])?[ -]?(?:usd|dollars?)\b",
    r"\bt ?\+ ?\d", r"\bsellable\b", r"\bsettlement\b", r"\bbrokers?\b", r"\busers?\b", r"\byou(?:rs?)?\b",
    # the writer's own: a thesis about the chart has no first person
    r"\b(?:we|our|ours|my|mine)\b", r"\bi(?:['’](?:m|ve|d|ll)| (?:am|have|had|own|would|will))\b",
    # stops and trading actions in the forms listed; a phrasal verb with a space or a dash between its words
    r"\bstop(?:s|-?loss(?:es)?)?\b", r"\bstop(?:ped|ping)[- ]out\b",
    r"\b(?:take|takes|taking|took|taken)(?:[- ][a-z]+){0,2}[- ]?profits?\b",
    r"\bprofits? (?:were |was |are |is |being |got )?(?:taken|booked|locked)\b",
    r"\bprofit[- ]?taking\b", r"\bcut(?:s|ting)?[- ]?(?:the |some )?loss", r"\bloss[- ]?cut",
    r"\bbuys?\b", r"\bbuying\b", r"\bbought\b", r"\bsells?\b", r"\bselling\b", r"\bsold\b",
    r"\bre-?(?:entr(?:y|ies)|enter(?:s|ed|ing)?|buy(?:s|ing)?|bought|sold|sell(?:s|ing)?|purchas(?:e[ds]?|ing))\b",
    r"\breduc(?:e[ds]?|ing)\b", r"\badd(?:s|ed|ing)?\b", r"\baccumulat(?:e[ds]?|ing)\b", r"\btrim(?:s|med|ming)?\b",
    r"\benter(?:s|ed|ing)?\b", r"\bentr(?:y|ies)\b", r"\bexit(?:s|ed|ing)?\b", r"\bwait(?:s|ed|ing)?\b",
    r"\b(?:stand|stands|standing|stood)[- ]aside\b", r"\bside-?line[sd]?\b", r"\b(?:new|fresh)[- ](?:money|funds?)\b",
    r"\b(?:limit|market)[- ]?orders?\b",
    r"\b(?:place[sd]?|placing|put|puts|putting|set|sets|setting|an?|the)(?: [a-z]+)? orders?\b",
    r"\b(?:orders?|offers?) (?:at|near|above|below|around) \d",
    r"\bscal(?:e[sd]?|ing)[- ](?:in|into|out)\b", r"\baverag(?:e[sd]?|ing)[- ]down\b", r"\bdoubl(?:e[sd]?|ing)[- ]down\b",
    r"\b(?:go|goes|going|went|gone)[- ](?:long|short)(?!-)\b", r"\bgo[- ]flat\b", r"\bshort(?:ed|ing)\b",
    r"\bshortsell", r"\b(?:longs|shorts)\b",  # "short sell", "short-sell": sell is a listed word
    r"\b(?:long|short)s? (?:above|below|over|under|near|at|into|toward|towards) \d",
    r"\bshorts? (?:the|this|it|them|into|strength)\b",
    r"\bfad(?:e|es|ed|ing) (?:the |any |every |a |this )?(?:rall(?:y|ies)|bounces?|strength|rips?|pops?|breakouts?)\b",
    r"\brecommend",
    # more of the same kind
    r"\bawait(?:s|ed|ing)?\b", r"\bpurchas(?:e[ds]?|ing)\b", r"\bhedg(?:e[ds]?|ing)\b", r"\bliquidat(?:e[ds]?|ing)\b",
    r"\bacquir(?:e[ds]?|ing)\b", r"\binvest(?:s|ed|ing)?\b", r"\bdump(?:s|ed|ing)?\b",
    r"\b(?:get|gets|getting|got|gotten)[- ](?:long|short|in|out)(?!-)\b",
    r"\blighten(?:s|ed|ing)?[- ]up\b", r"\bload(?:s|ed|ing)?[- ]up\b", r"\bbail(?:s|ed|ing)?[- ]out\b",
    r"\bbottom[- ]?fish", r"\bfalling[- ]knife\b", r"\block(?:s|ed|ing)?[- ]in (?:[a-z]+ ){0,3}?(?:gains?|profits?)\b",
    r"\bbook(?:s|ed|ing)? (?:[a-z]+ ){0,2}?(?:gains?|profits?|losses?)\b", r"\boff[- ](?:of[- ])?the[- ]table\b",
    r"\bstep(?:s|ped|ping)?[- ]aside\b", r"\bstay(?:s|ed|ing)?[- ]away\b(?! from)", r"\bset(?:s|ting)?[- ]aside\b",
    r"\bfill(?<!\bgap fill)(?<!\bgap was fill)(?:s|ed)? (?:at|near|around)\b",
    r"\bpick(?:s|ed|ing)?[- ]up (?:(?:some|more|the|a few|cheap) )?(?:shares?|stock)\b",
    r"\b(?:clos|open|manag)(?:e|es|ed|ing|s)? (?:the|this|that|a|any|each|new) trades?\b",
    r"\bkeep(?:s|ing)? the (?:stock|shares|name)\b",
    r"\b(?:a|new) (?:long|short)s? (?:at|near|above|below)\b",
    r"\b(?:long|short) (?:from|since)\b", r"\bexisting (?:longs?|shorts?|investors?|owners?)\b",
    r"\balready (?:(?:long|short)(?!-)|in the (?:stock|name|trade))\b",
    r"\bstay(?:s|ed|ing)?[- ](?:(?:long|short)(?!-)|out|aside|flat)\b", r"\b(?:target[- ]price|price[- ]target)s?\b",
    # a trade set-up and its costs (from the held-out measurement, review-fix/heldout)
    r"\br ?[:/] ?r\b", r"\brisk[ /-]?(?:to[ -])?reward\b", r"\breward[ /-]?(?:to[ -])?risk\b",
    r"\b(?:fees?|commissions?|slippage)\b", r"\bquantit(?:y|ies)\b", r"\baffordab", r"\borders\b",
    r"\bmục tiêu\b", r"\bnhắm\b", r"\bgiá vào\b(?! vùng)", r"\bchi phí\b", r"\bphí giao dịch\b",
    # the same in Vietnamese, and typed without diacritics
    r"\b(?:mua|bán|giữ|gom|chờ|nên|lệnh|lãi|lời|lỗ|vốn)\b", r"\bđợi\b", r"\bxả\b", r"\bnắm\b", r"\bnhặt\b",
    r"\bchốt\b(?! (?:phiên|tuần|tháng|quý|năm|ngày))", r"\bcắt\b(?! (?:lên|xuống|qua|ngang))", r"\bthoát\b(?! khỏi)",
    r"\bcầm (?:tiếp|hàng|cổ|chắc)\b", r"\bgiải ngân\b", r"\bxuống tiền\b", r"\bôm hàng\b", r"\btất tay\b",
    r"\blướt sóng\b", r"\bhàng về\b", r"\btài khoản\b", r"\bgiá (?:mua )?bình quân\b", r"\bbình quân giá\b",
    r"\bbắt đáy\b", r"\bvào hàng\b", r"\btham gia\b", r"\bsở hữu\b", r"\bnhàn rỗi\b", r"\bsố dư\b",
    r"\btrung bình giá\b", r"\bgia tăng (?:quanh|tại|thêm)\b", r"\b(?:mức|điểm) dừng\b", r"\bđang âm\b",
    r"\bt[ỷỉ] trọng\b", r"\bdanh mục\b", r"\bvị thế\b", r"\btiền mặt\b", r"\bngân sách\b",
    r"\bkhuy(?:ên|ến)\b", r"\bhành động\b", r"\bđứng ngoài\b", r"\bk[ýí] quỹ\b", r"\b(?:tôi|chúng ta|mình|bạn)\b",
    r"\b(?:ban (?:ra|bot|het)|cat lo|chot (?:loi|lai|mot phan|bot|het)|nen (?:ban|cho)|dung ngoai"
    r"|giai ngan|bat day|vao hang|thoat hang|so huu|nhan roi|so du|trung binh gia|cat giam|muc dung"
    r"|diem dung|dang am|dang loi|t[yi] trong|danh muc|vi the|tien mat|ngan sach|khuyen nghi|hanh dong|k[yi] quy"
    r"|xuong tien|om hang|tat tay|luot song|hang ve|tai khoan|gia von|gia binh quan|giu|lenh|trieu)\b",
    # URLs (scheme://, www.), host names (label.tld, a Windows default name), dotted IPv4 (four numbers) and IPv6
    # ("::" or eight groups) addresses, host:port, ":port" after no letter, digit or colon, the word port
    r"[a-z][a-z0-9+.-]{0,30}://", r"\bwww\.", r"\blocalhost\b", r"\bloopback\b", r"\b(?:tailscale|ngrok)\b",
    r"\bbtc-(?:ai|be|fe)\b", r"\b(?:desktop|laptop)-[a-z0-9]{5,15}\b",
    r"[a-z0-9-]\.[a-z]{2,24}\b", r"\bdot (?:com|net|org|vn|io|dev|app)\b", r"\d(?<!\w\d)\d{0,2}(?:\.\d{1,3}){3}\b",
    r"::", r"\b(?:[0-9a-f]{1,4}:){7}[0-9a-f]{1,4}\b", r"\b[a-z][a-z0-9.-]{0,62}:\d{2,5}\b", r":(?<![\w:]:)\d{4,5}\b",
    r"\bports?\b",
    # file paths
    r"\\", r"\b[a-z]:/", r"(?<![\w)\]~./])[~.]{0,2}/[\w.-]+/",
    r"/(?<![\w)\]~./]/)(?:home|tmp|var|etc|mnt|opt|srv|root|api)\b",  # "/users": a listed word
    r"\b(?:evidence|scripts|skills|backend|frontend|contracts)/",
    r"\w\.(?:json|md|py|ps1|psm1|cs|csproj|sln|tsx?|m?js|cjs|txt|log|csv|tsv|xml|html?|exe|dll|db|sqlite|sql|env"
    r"|ya?ml|ini|cfg|conf|toml|bat|cmd|sh|joblib|pkl|parquet|zip|7z|bak|clixml|pem|crt)\b",  # ".dump": a listed word
    r"\b(?:research-sessions|node_modules|appdata)\b", r"\.ops\b",
    # e-mail addresses, hex strings and digit runs, tokens, internal ids and secrets
    r"@", r"[0-9a-f](?<!\w[0-9a-f])[0-9a-f]{15,}\b", r"\b(?=[0-9a-f]{8,15}\b)(?=[0-9a-f]*[a-f])[0-9a-f]*\d",
    r"\d(?<!\w\d)\d{11,}\b", r"\b0x[0-9a-f]{8,}",
    r"[a-z0-9+/=_-]{32,}",  # also every uuid (36 characters of hex digits and dashes)
    r"\bsnap[/:_-]", r"\bsnapshot", r"\bsessions?/",
    r"\b(?:note|forecast|request)[ _-]?ids?\b", r"\bx-admin", r"\badmin[ _-]?key", r"\bapi[ _-]?keys?\b",
    r"\bbearer\b", r"\bpassw", r"\bsecrets?\b", r"\btokens?\b", r"\bcredentials?\b", r"\beqx_",
)
# The same, where case decides: read on the probe as written
_NOT_TICKER = r"(?!SMA|EMA|RSI|ATR|OBV|POC|VAL|VAH|VND)"  # the writer's indicator abbreviations, the currency
# A count of the ticker's own right after it: one space, or one or two of ":", ";", ",", "(", "[", "~", "×", "=" or "≈"
# (an optional space on each side; "[" is a literal inside the class), " x ", one or two dashes with a space on each
# side, about, roughly, around, some, approximately, approx., approx, nearly or almost ("MBB 3,000", "MBB: ~2,600",
# "MBB (about 2,600)", "MBB -- 2,600"). Not a sign, "'s", at, near or another word or symbol: "VCB +5.1", "MBB's
# 20-session low", "BID at 31700" are lists of a chart.
_COUNT_AFTER_TICKER = (r"(?:(?: ?[:;,([~×=≈] ?| x ?| --? | ?(?:about|roughly|around|some|approximately"
                       r"|approx\.|approx|nearly|almost) ){1,2}| )\d")
CASED_PATTERNS = (
    # "<number> <ticker>" (3,000 MBB, 3K MBB, 3 nghìn MBB; also across a full stop: "closed 18750. MBB"), but not one of
    # the abbreviations of _NOT_TICKER (20 EMA, 18,750 VND; any other one withholds: "the 200 DMA") and not a number
    # that ends with a comma: a list ("+6.8, VCB +5.1", "18900, MBB's lowest close") ...
    r"\d(?<![\w.,/:-]\d)[\d,.]{0,20}(?<!,) ?(?:[kK]|nghìn|ngàn)? ?" + _NOT_TICKER + r"[A-Z]{3}\b",
    # ... unless the ticker after the comma has a count of its own: a list of counts written after the tickers ("MBB
    # 3,000, BID 1,500", "MBB: 3,000, BID: 1,500", "BID 1,200, MBB (2,600)"; so "MBB 18900, BID 31700" withholds too),
    r"\d(?<![\w.,/:-]\d)[\d,.]{0,20}[kK]?, ?" + _NOT_TICKER + r"[A-Z]{3}" + _COUNT_AFTER_TICKER,
    # the comma is glued to the ticker ("2,600,MBB"),
    r"\d," + _NOT_TICKER + r"[A-Z]{3}\b",
    # or the number has no sign and the ticker ends the clause: a count written first ("3,000, MBB."; not before a
    # space and a word, a comma or an apostrophe: "18,550, MBB bounced", "In 2025, MBB, BID and VCB", "18900, MBB's";
    # "+6.8, VCB" is a list). The space the probe makes of whitespace that ends the text is the end ("3,000, MBB\n").
    r"\d(?<![+\w.,/:-]\d)[\d,.]{0,20}, ?" + _NOT_TICKER + r"[A-Z]{3}\b(?![,'’]| \S)",
    r"\b[Bb]ids?\b",  # an order; the ticker BID is written in capitals
    r"\b(?:TP|SL)\d?\b",  # take-profit and stop-loss levels ("TP 21,000, SL 18,500")
    r"\b[Kk]eep(?:s|ing)? " + _NOT_TICKER + r"[A-Z]{3}\b",  # "Keep MBB while ..."
)


def _alternation(patterns: tuple[str, ...]) -> str:
    r"""One regex for the alternatives, laid out for speed: every alternative that starts with \b and a word goes under
    a single \b(?:...), so that sre skips each branch whose first literal does not match the character at hand
    without entering it (with one top-level alternative each, every one of them re-checks \b at every position)."""
    def word(p: str) -> bool:
        return p.startswith(r"\b") and (p[2:3].isalnum() or p[2:5] == "(?:")
    return r"\b(?:" + "|".join(p[2:] for p in patterns if word(p)) + ")|" + "|".join(p for p in patterns if not word(p))


PRIVATE_TEXT = re.compile(_alternation(PRIVATE_PATTERNS))
CASED_TEXT = re.compile("|".join(CASED_PATTERNS))
# The characters the keyword lists can read (an allowlist: any other character, in the text as written or after NFKC,
# withholds the thesis, _unreadable): printable ASCII with tab and newline; the Latin-1 letters and signs, but not the
# soft hyphen, the ordinal and micro signs, the spacing accents that NFKC turns into a space and a combining mark
# (U+00A8, U+00AF, U+00B4, U+00B8) or the fractions that NFKC writes with U+2044 (U+00BC-U+00BE); the Vietnamese
# letters (U+0102, U+0103, U+0110, U+0111, U+0128, U+0129, U+0168, U+0169, U+01A0, U+01A1, U+01AF, U+01B0 and
# U+1EA0-U+1EF9) and the combining marks of decomposed Vietnamese; and a short list of typography: the dashes
# U+2010-U+2015, curly quotes, bullet, ellipsis, per mille, primes, the arrows U+2190-U+2193, U+2248, U+2260, U+2264,
# U+2265, the minus sign U+2212, U+20AB (dong) and U+20AC (euro).
READABLE = frozenset(
    "\t\n\r" + "".join(map(chr, range(0x20, 0x7f)))
    + "".join(chr(c) for c in range(0xa0, 0x100)
              if c not in (0xa8, 0xaa, 0xad, 0xaf, 0xb4, 0xb5, 0xb8, 0xba, 0xbc, 0xbd, 0xbe))
    + "\u0102\u0103\u0110\u0111\u0128\u0129\u0168\u0169\u01a0\u01a1\u01af\u01b0"
    + "".join(map(chr, range(0x1ea0, 0x1efa)))
    + "\u0300\u0301\u0302\u0303\u0306\u0309\u031b\u0323"
    + "\u2010\u2011\u2012\u2013\u2014\u2015\u2018\u2019\u201c\u201d\u2022\u2026\u2030\u2032\u2033"
    + "\u2190\u2191\u2192\u2193\u2248\u2260\u2264\u2265\u2212\u20ab\u20ac")
# The dashes _probe reads as "-": category Pd (of READABLE: U+2010-U+2015) and U+2212 MINUS SIGN
DASHES = frozenset("\u2212")

Fetch = Callable[[str, dict], Any]


class ScreenError(ValueError):
    pass


class BackendError(RuntimeError):
    """The backend answered /api/equities, but not with the payload shape this script reads."""


def http_fetch(backend: str, timeout: float = 120.0) -> Fetch:
    def fetch(path: str, headers: dict) -> Any:
        request = urllib.request.Request(backend.rstrip("/") + path, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    return fetch


def _num(x: Any) -> int | float | None:
    if x is None:
        return None
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise TypeError(f"not a finite number: {x!r}")
    return x


def _count(x: Any) -> int:
    if isinstance(x, bool) or not isinstance(x, int) or x < 0:
        raise TypeError(f"not a count: {x!r}")
    return x


def _enum(x: Any, allowed: frozenset[str] | dict[str, str]) -> str:
    if not isinstance(x, str) or x not in allowed:
        raise ValueError(f"not a contract value: {x!r}")
    return x


def _iso(x: Any) -> str:
    if not isinstance(x, str) or date.fromisoformat(x).isoformat() != x:
        raise ValueError(f"not an ISO date: {x!r}")
    return x


def _ticker(x: Any) -> str:
    if not isinstance(x, str) or not TICKER.fullmatch(x):
        raise ScreenError(f"not a ticker: {x!r}")
    return x


def screen_doc(raw: dict) -> dict:
    if raw["schema"] != SCREEN_SCHEMA:
        raise ScreenError(f"schema {raw['schema']!r}")
    months = raw["ranking"]["lookbackMonths"]
    rows = [{"symbol": _ticker(r["symbol"]), "close": _num(r["close"]),
             "returns": {m: _num(r["returns"][m]) for m in LOOKBACK},
             "excessVsIndex": {m: _num(r["excessVsIndex"][m]) for m in LOOKBACK},
             "rank": _num(r["ranks"][str(months)]), "top": r["topByRankMonths"] is True} for r in raw["rows"]]
    if not rows or len({r["symbol"] for r in rows}) != len(rows):
        raise ScreenError("rows empty or a symbol repeats")
    return {
        "session": date.fromisoformat(raw["session"]).isoformat(),
        "basket": {"effectiveDate": date.fromisoformat(raw["basket"]["effectiveDate"]).isoformat(),
                   "possiblyStale": raw["basket"]["possiblyStale"] is True},
        "indexReturns": {m: _num(raw["indexReturns"][m]) for m in LOOKBACK},
        "rankingMonths": _num(months), "topK": _num(raw["ranking"]["topK"]),
        "missingSessionBar": [_ticker(s) for s in raw["missingSessionBar"]],
        "notUpToDate": [_ticker(s) for s in raw["notUpToDateNow"]],
        "rows": rows,
    }


def latest_screen(screen_dir: Path) -> dict:
    files = sorted(p for p in screen_dir.glob("*.json") if re.fullmatch(r"\d{4}-\d{2}-\d{2}\.json", p.name))
    if not files:
        raise ScreenError(f"no <session>.json in {screen_dir}")
    try:
        return screen_doc(json.loads(files[-1].read_text(encoding="utf-8")))
    except (KeyError, TypeError, AttributeError, ValueError) as e:  # ScreenError and JSONDecodeError are ValueErrors
        raise ScreenError(f"{files[-1].name}: {type(e).__name__}: {e}") from e


class Pacer:
    """Starts calls at least `gap` seconds apart."""

    def __init__(self, gap: float, sleep: Callable[[float], None], clock: Callable[[], float]):
        self.gap, self.sleep, self.clock, self.last = gap, sleep, clock, None

    def __call__(self) -> None:
        if self.last is not None and (wait := self.last + self.gap - self.clock()) > 0:
            self.sleep(wait)
        self.last = self.clock()


def _get(fetch: Fetch, path: str, headers: dict, sleep: Callable[[float], None], pace: Pacer | None = None) -> Any:
    for attempt in (1, 2):
        if pace is not None:
            pace()
        try:
            return fetch(path, headers)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 2:
                raise
        sleep(RETRY_AFTER_S)


def _reason(e: urllib.error.HTTPError) -> tuple[str, bool, str]:
    """(public Vietnamese reason, whether it is 400 UNSUPPORTED_CAPABILITY: a part the backend does not serve, detail
    for stderr: the status and the envelope's code)."""
    try:
        code = json.loads(e.read().decode("utf-8")).get("code")
    except Exception:  # an error body that is not the JSON envelope
        code = None
    detail = f"HTTP {e.code}" + (f" {code}" if isinstance(code, str) and re.fullmatch(r"[A-Z_]{1,40}", code) else "")
    if e.code == 400 and code == "UNSUPPORTED_CAPABILITY":
        return UNSUPPORTED, True, detail
    return HTTP_FAILED.format(status=e.code), False, detail


def _part(read: Callable[[], Any]) -> tuple[Any, str | None, bool, str]:
    """(value, None, False, ""), or (None, reason, keep, detail) for an HTTP error or a malformed payload, where keep
    says the previous run's values may stand in (not for UNSUPPORTED_CAPABILITY) and detail is for stderr only; an
    unreachable backend still raises."""
    try:
        return read(), None, False, ""
    except urllib.error.HTTPError as e:
        reason, unsupported, detail = _reason(e)
        return None, reason, not unsupported, detail
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        return None, MALFORMED, True, f"{type(e).__name__}: {e}"


def _detail(text: str) -> str:
    """One short ASCII line for stderr: a redirected stderr is cp1252, and a payload value may be anything."""
    return " ".join(str(text).split())[:300].encode("ascii", "backslashreplace").decode("ascii")


def candle_rows(payload: list) -> list[list]:
    rows = [[session_of(c["openTimeMs"]), _num(c["open"]), _num(c["high"]), _num(c["low"]), _num(c["close"]),
             _num(c["volume"])] for c in payload]
    return [r for r in rows if r[5] > 0][-CANDLE_LIMIT:]


def technical_parts(payload: dict) -> tuple[dict | None, dict | None, str]:
    """(latest 1d indicators, excess vs VNINDEX over 20/60 sessions, why that excess is None, for stderr); either
    value may be None."""
    day = next((t for t in payload["timeframes"] if t["timeframe"] == "1d"), None)
    indicators = None
    if day is not None and day["indicators"] is not None:
        i = day["indicators"]
        close = next((c["close"] for c in day["candles"] if c["openTimeMs"] == i["openTimeMs"]), None)
        indicators = {"date": session_of(i["openTimeMs"]), "close": _num(close),
                      **{f: _num(i.get(f)) for f in INDICATORS}}
    bench = next((b for b in payload["relativeStrength"]["benchmarks"] if b["symbol"] == "VNINDEX"), None)
    strength, why = None, "no VNINDEX benchmark"
    if bench is not None:
        why = f"VNINDEX benchmark status {bench['status']!r}, reason {bench.get('reason')!r}"
        if bench["status"] == "ok":  # "stale" keeps an old window, not the latest session
            strength, why = {w: _num((bench["returns"].get(w) or {}).get("excess")) for w in RS_WINDOWS}, ""
    return indicators, strength, why


def _probe(text: str) -> str:
    """What the privacy filter reads, of a text of READABLE characters: NFKC (Vietnamese composed), every dash
    (category Pd: U+2010 that NFKC makes of U+2011, en and em dash..., and DASHES) as "-", control characters and
    stray combining marks removed, whitespace runs as one space, chart phrases in their place blanked (_chart_phrase)."""
    text = unicodedata.normalize("NFKC", text)
    table: dict[int, str | None] = {}  # built over the distinct characters only
    for c in set(text):
        category = unicodedata.category(c)
        if c not in "\t\n\r" and category in ("Cc", "Cf", "Mn", "Me"):
            table[ord(c)] = None
        elif category == "Pd" or c in DASHES:
            table[ord(c)] = "-"
    return TECHNICAL_PHRASES.sub(_chart_phrase, re.sub(r"\s+", " ", text.translate(table)))


def _unreadable(text: str) -> bool:
    """A character of the text as written outside READABLE, which the keyword lists cannot read: a format or bidi
    control (U+200B, U+202A-U+202E, U+2066-U+2069: the page would show what the filter does not read), any other
    control or space character (a thin space inside "b uy"), a letter outside Latin-1 and Vietnamese (a Cyrillic,
    Greek or IPA look-alike, a small capital, a dotless i, a mathematical, full-width, enclosed or modifier letter), a
    dash look-alike (a box-drawing line) or any other symbol; or a readable mark that composes with its letter into
    one outside READABLE ("s" + U+0301 is U+015B, which the page shows as an s). A word glued to "_" or a digit
    ("buy18600") is readable but not a word, so it passes: the writer does not write that (left out on purpose)."""
    return not (READABLE.issuperset(text) and READABLE.issuperset(unicodedata.normalize("NFKC", text)))


def _private(text: str) -> bool:
    """A denylist match: PRIVATE_TEXT on the lowercased probe, CASED_TEXT on the probe as written. (No character of
    READABLE lowers to a combining mark; U+0130, which lowers to "i" and U+0307, is not readable.)"""
    probe = _probe(text)
    return PRIVATE_TEXT.search(probe.lower()) is not None or CASED_TEXT.search(probe) is not None


def public_thesis(raw: Any) -> tuple[str | None, bool]:
    """(the thesis to publish or None, whether the privacy filter withheld it). The filter reads the whole text."""
    if not isinstance(raw, str) or not raw.strip():
        return None, False
    if _unreadable(raw) or _private(raw):
        return None, True
    return (raw if len(raw) <= THESIS_MAX else raw[:THESIS_MAX - 1] + CUT), False


def score_of(horizons: list) -> dict:
    """The contract's score: the direction-scores horizons[] entry with horizonSessions HORIZON (whole journal)."""
    entries = [h for h in horizons if h["horizonSessions"] == HORIZON]
    if len(entries) != 1:
        raise ValueError(f"{len(entries)} horizons[] entries with horizonSessions {HORIZON}")
    h = entries[0]
    score = {"scored": _count(h["resolved"]), "hits": _count(h["hits"]),
             "baselineHits": _count(h["baselinePrior"]["hits"]), "sampleStatus": _enum(h["status"], SAMPLE_STATUSES)}
    if max(score["hits"], score["baselineHits"]) > score["scored"]:
        raise ValueError("more hits than scored notes")
    return score


def market_only(note: dict) -> bool:
    """Whether a note may be exported at all (contract r4 X1; from r5, D1, a note that fails is dropped whole):
    actionBasis in PUBLIC_BASES and actionView in PUBLIC_VIEWS. Any other value, a missing field or a value that is not
    a string says no (fail-closed)."""
    basis, view = note.get("actionBasis"), note.get("actionView")
    return isinstance(basis, str) and basis in PUBLIC_BASES and isinstance(view, str) and view in PUBLIC_VIEWS


def note_rows(notes: dict, scores: dict) -> tuple[list[dict], dict, int, int]:
    """(the newest forward H=20 notes that are market_only and not superseded, at most NOTE_LIMIT; the whole-journal
    H=20 score; theses withheld; notes dropped); a malformed payload raises. A note that is not market_only is dropped
    like a superseded one (no row, no slot of NOTE_LIMIT, not a withheld thesis) and counts in the last number unless
    it is superseded or past the NOTE_LIMIT notes taken. The score is the backend's and still counts that note once it
    is scored, so the list can hold fewer scored notes than the score counts."""
    score = score_of(scores["horizons"])
    by_id = {i["noteId"]: i for i in scores["items"]}
    forward = sorted((n for n in notes["items"] if n["mode"] == "forward" and n["horizonSessions"] == HORIZON),
                     key=lambda n: (n["signalBarOpenTimeMs"], n["availableTimeMs"]), reverse=True)
    out, withheld, dropped = [], 0, 0
    for n in forward:
        if len(out) == NOTE_LIMIT:
            break
        s = by_id[n["noteId"]]  # a note without its score fails the whole part: the site needs a status
        if (s["signalBarOpenTimeMs"], s["horizonSessions"], s["direction"]) != \
                (n["signalBarOpenTimeMs"], n["horizonSessions"], n["direction"]):
            raise ValueError("a direction score does not match its note")
        if s["status"] == SUPERSEDED:  # a later note on a bar already taken: never scored, would pose as the latest call
            continue
        if not market_only(n):  # the writer may have known the user's holding: neither the row nor the thesis leaves
            dropped += 1
            continue
        status = STATUS_OUT[_enum(s["status"], STATUS_OUT)]
        hit = s["hit"] if status == "scored" else None
        if status == "scored" and not isinstance(hit, bool):
            raise TypeError("a scored note without a boolean hit")
        target, reason = s["targetBarOpenTimeMs"], s["pendingReason"]
        body = n.get("body")
        thesis, hidden = public_thesis(body.get("thesis") if isinstance(body, dict) else None)
        withheld += hidden
        out.append({"signalDate": session_of(_num(n["signalBarOpenTimeMs"])), "horizonSessions": HORIZON,
                    "direction": _enum(n["direction"], DIRECTIONS), "thesis": thesis, "status": status,
                    "pendingReason": None if reason is None else _enum(reason, PENDING_REASONS),
                    "targetDate": None if target is None else session_of(_num(target)),
                    "returnPct": _num(s["returnPct"]), "hit": hit})
    return out, score, withheld, dropped


# --- the previous run's parts: our own earlier output, validated again (and theses filtered again) before reuse ---
# market_only cannot be checked again: the files do not carry actionView/actionBasis, so a kept note passed it when it
# was first exported (from VN30-WEB r5; r4 to r4c exported the row of a note that failed it with its thesis withheld,
# so from r4 a kept thesis passed it, and every thesis exported before r4 came from a wait/market-only note).

def _fields(x: Any, names: tuple[str, ...]) -> dict:
    if not isinstance(x, dict) or set(x) != set(names):
        raise ValueError("fields differ from the contract")
    return x


def _old_indicators(x: Any) -> dict | None:
    if x is None:
        return None
    x = _fields(x, INDICATOR_FIELDS)
    return {"date": _iso(x["date"]), **{f: _num(x[f]) for f in INDICATOR_FIELDS[1:]}}


def _old_strength(x: Any) -> dict | None:
    if x is None:
        return None
    x = _fields(x, RS_WINDOWS)
    return {w: _num(x[w]) for w in RS_WINDOWS}


def _old_score(x: Any) -> dict | None:
    if x is None:
        return None
    x = _fields(x, SCORE_FIELDS)
    score = {"scored": _count(x["scored"]), "hits": _count(x["hits"]), "baselineHits": _count(x["baselineHits"]),
             "sampleStatus": _enum(x["sampleStatus"], SAMPLE_STATUSES)}
    if max(score["hits"], score["baselineHits"]) > score["scored"]:
        raise ValueError("more hits than scored notes")
    return score


def _old_notes(x: Any) -> tuple[list[dict], int]:
    if not isinstance(x, list) or len(x) > NOTE_LIMIT:
        raise ValueError("notes")
    out, withheld = [], 0
    for n in x:
        n = _fields(n, NOTE_FIELDS)
        status = _enum(n["status"], NOTE_STATUSES)
        hit = n["hit"]
        if (status == "scored") != isinstance(hit, bool) or (status != "scored" and hit is not None):
            raise ValueError("hit")
        if n["horizonSessions"] != HORIZON:
            raise ValueError("horizon")
        thesis, hidden = public_thesis(n["thesis"])
        withheld += hidden
        out.append({"signalDate": _iso(n["signalDate"]), "horizonSessions": HORIZON,
                    "direction": _enum(n["direction"], DIRECTIONS), "thesis": thesis, "status": status,
                    "pendingReason": None if n["pendingReason"] is None else _enum(n["pendingReason"], PENDING_REASONS),
                    "targetDate": None if n["targetDate"] is None else _iso(n["targetDate"]),
                    "returnPct": _num(n["returnPct"]), "hit": hit})
    if any(a["signalDate"] < b["signalDate"] for a, b in zip(out, out[1:])):
        raise ValueError("notes not newest first")
    return out, withheld


def _old_candles(x: Any) -> list[list]:
    if not isinstance(x, list) or len(x) > CANDLE_LIMIT:
        raise ValueError("candles")
    rows = []
    for r in x:
        if not isinstance(r, list) or len(r) != 6 or any(v is None for v in r[1:]) or _num(r[5]) <= 0:
            raise ValueError("candle row")
        rows.append([_iso(r[0]), *(_num(v) for v in r[1:])])
    if any(a[0] >= b[0] for a, b in zip(rows, rows[1:])):
        raise ValueError("candles not ascending")
    return rows


class PreviousRun:
    """The previous run's files in out_dir; usable only when its meta.json has schema SCHEMA. Each kept_* method
    returns None when there is nothing to keep: no usable file, values that do not validate, or a part that was
    itself empty (not read) in the previous run."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        try:
            meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = None
        self.usable = isinstance(meta, dict) and meta.get("schema") == SCHEMA

    def doc(self, symbol: str) -> dict | None:
        if not self.usable:
            return None
        try:
            doc = json.loads((self.out_dir / "symbols" / f"{symbol}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return doc if isinstance(doc, dict) and doc.get("symbol") == symbol else None

    def kept_technical(self, symbol: str) -> tuple[dict | None, dict | None] | None:
        try:
            doc = self.doc(symbol)
            kept = None if doc is None else (_old_indicators(doc["indicators"]), _old_strength(doc["relativeStrength"]))
        except (KeyError, TypeError, ValueError, AttributeError):
            return None
        return None if kept is None or kept == (None, None) else kept

    def kept_notes(self, symbol: str) -> tuple[list[dict], dict, int] | None:
        try:
            doc = self.doc(symbol)
            score = None if doc is None else _old_score(doc["score"])
            if score is None:  # no file, or the previous run had no notes part either
                return None
            notes, withheld = _old_notes(doc["notes"])
        except (KeyError, TypeError, ValueError, AttributeError):
            return None
        return notes, score, withheld

    def kept_whole(self, symbol: str) -> tuple[dict, int] | None:
        try:
            doc = self.doc(symbol)
            if doc is None:
                return None
            doc = _fields(doc, SYMBOL_FIELDS)
            candles = _old_candles(doc["candles"])
            if not candles:  # the previous run had no candles either: read the other parts fresh
                return None
            if not isinstance(doc["label"], str) or not isinstance(doc["isBank"], bool):
                raise TypeError("label/isBank")
            score = _old_score(doc["score"])
            notes, withheld = _old_notes(doc["notes"])
            if score is None and notes:
                raise ValueError("notes without a score")
            kept = {"symbol": symbol, "label": doc["label"], "isBank": doc["isBank"], "candles": candles,
                    "indicators": _old_indicators(doc["indicators"]),
                    "relativeStrength": _old_strength(doc["relativeStrength"]), "notes": notes, "score": score}
        except (KeyError, TypeError, ValueError, AttributeError):
            return None
        return kept, withheld


def build(fetch: Fetch, screen: dict, admin_key: str | None, out_dir: Path,
          sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
          counts: dict[str, int] | None = None) -> tuple[dict[str, dict], list[dict], int]:
    """(symbol -> document, meta.unavailable, theses withheld). A document kept whole is in the result too. When
    counts is given, counts["notesDropped"] is set to the notes market_only dropped from the fresh notes parts (a kept
    part drops none): for the summary line only, never written to a file."""
    try:
        labels = {i["symbol"]: i["label"] for i in _get(fetch, "/api/equities", {}, sleep)["items"]}
    except (KeyError, TypeError, AttributeError) as e:
        raise BackendError(f"/api/equities: {type(e).__name__}: {e}") from e
    previous = PreviousRun(out_dir)
    pace = Pacer(TECH_GAP_S, sleep, clock)
    admin = {"X-Admin-Key": admin_key} if admin_key else None
    docs, unavailable, withheld, dropped = {}, [], 0, 0
    for s in sorted(r["symbol"] for r in screen["rows"]):
        def miss(part: str, reason: str, detail: str, keep: bool = False, kept: bool = False) -> None:
            suffix = "" if not keep else KEPT if kept else NOTHING_KEPT  # never kept: UNSUPPORTED_CAPABILITY
            unavailable.append({"symbol": s, "part": part, "reason": reason + suffix})
            print(f"unavailable {s} {part}: {_detail(detail)}", file=sys.stderr)

        candles, why, keep, detail = _part(lambda: candle_rows(
            _get(fetch, f"/api/equities/{s}/candles?timeframe=1d&limit={CANDLE_LIMIT}", {}, sleep)))
        if why is not None:
            whole = previous.kept_whole(s) if keep else None
            miss("candles", why, detail, keep, whole is not None)
            if whole is not None:
                docs[s] = whole[0]
                withheld += whole[1]
                continue
            candles = []

        technical, why, keep, detail = _part(lambda: technical_parts(
            _get(fetch, f"/api/equities/{s}/technical?bars=1", {}, sleep, pace)))
        if why is None:
            indicators, strength, no_strength = technical
            if indicators is None:
                miss("technical", NO_INDICATORS, "no 1d indicator row")
            if strength is None:
                miss("technical", NO_STRENGTH, no_strength)
        else:
            kept = previous.kept_technical(s) if keep else None
            miss("technical", why, detail, keep, kept is not None)
            indicators, strength = kept or (None, None)

        if admin is None:
            part, why, keep, detail = None, NO_KEY, True, "EQX_ADMIN_KEY is not set"
        else:
            part, why, keep, detail = _part(lambda: note_rows(
                _get(fetch, f"/api/equities/{s}/research-notes?take={NOTES_TAKE}", admin, sleep),
                _get(fetch, f"/api/equities/{s}/research-notes/direction-scores?take={SCORES_TAKE}", admin, sleep)))
        if why is not None:
            kept = previous.kept_notes(s) if keep else None
            miss("notes", why, detail, keep, kept is not None)
            part = ([], None, 0, 0) if kept is None else (*kept, 0)  # a kept file holds exported notes only
        notes, score, hidden, drops = part
        withheld += hidden
        dropped += drops

        label = labels.get(s)
        docs[s] = {"symbol": s, "label": label if isinstance(label, str) else s, "isBank": s in BANKS,
                   "candles": candles, "indicators": indicators, "relativeStrength": strength, "notes": notes,
                   "score": score}
    if counts is not None:
        counts["notesDropped"] = dropped
    return docs, unavailable, withheld


def _dump(doc: dict) -> bytes:
    return (json.dumps(doc, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")


def write_if_changed(path: Path, data: bytes) -> bool:
    if path.exists() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)  # readers never see a half-written file
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return True


def write_outputs(out_dir: Path, screen: dict, docs: dict[str, dict], unavailable: list[dict]) -> list[str]:
    for stale in [*out_dir.glob("*.json.tmp"), *(out_dir / "symbols").glob("*.json.tmp")]:
        stale.unlink()  # left by an interrupted run: never published
    files = [("screen.json", screen)] + [(f"symbols/{s}.json", d) for s, d in docs.items()]
    changed = [rel for rel, doc in files if write_if_changed(out_dir / rel, _dump(doc))]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = {"schema": SCHEMA, "generatedAtUtc": now, "session": screen["session"],
            "symbols": sorted(r["symbol"] for r in screen["rows"]), "unavailable": unavailable, "source": SOURCE}
    try:
        old = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = None
    if not changed and isinstance(old, dict) and {**old, "generatedAtUtc": now} == meta:
        meta = old  # nothing changed: keep the timestamp, so an idle day makes no commit
    if write_if_changed(out_dir / "meta.json", _dump(meta)):
        changed.append("meta.json")
    return changed


def main(argv: list[str] | None = None, fetch: Fetch | None = None,
         sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--screen-dir", type=Path, default=DEFAULT_SCREEN_DIR)
    ap.add_argument("--backend", default=os.getenv("BACKEND_BASE_URL") or DEFAULT_BACKEND)
    args = ap.parse_args(argv)
    try:
        screen = latest_screen(args.screen_dir)
    except ScreenError as e:
        print(f"screen error: {e}", file=sys.stderr)
        return 2
    counts: dict[str, int] = {}
    try:
        docs, unavailable, withheld = build(fetch or http_fetch(args.backend), screen,
                                            os.getenv("EQX_ADMIN_KEY") or None, args.out_dir, sleep, clock, counts)
    except (urllib.error.URLError, http.client.HTTPException, OSError, BackendError, ValueError) as e:
        print(f"backend error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    changed = write_outputs(args.out_dir, screen, docs, unavailable)
    parts: dict[str, int] = {}
    kept: dict[str, int] = {}
    for u in unavailable:
        parts[u["part"]] = parts.get(u["part"], 0) + 1
        if u["reason"].endswith(KEPT):
            kept[u["part"]] = kept.get(u["part"], 0) + 1
    print(json.dumps({"session": screen["session"], "symbols": len(screen["rows"]),
                      "written": len(docs) - kept.get("candles", 0), "changed": changed,
                      "unavailableByPart": parts, "keptByPart": kept, "thesesWithheld": withheld,
                      "notesDropped": counts["notesDropped"]}))  # ASCII: a redirected stdout is cp1252; local log only
    return 0


if __name__ == "__main__":
    sys.exit(main())
