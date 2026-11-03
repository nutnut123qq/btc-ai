"""VN30-WEB exporter, contract vn30-web-data/v2: only contract fields reach the public files; notes are the newest
forward 20-session notes and score is the backend's whole-journal H=20 total; a value outside the contract enums makes
the notes part malformed; a thesis passes a fail-closed privacy filter and carries a cut marker; a part that fails
keeps the previous v2 run's values (400 UNSUPPORTED_CAPABILITY never does); /technical calls start 15 s apart; a file
is rewritten only when its bytes change.

Round r3 (evidence/vn30-web-20261103/r3/contract.md): the filter's action words match in every inflection (E1) and
every dash it reads is "-"; the technical phrases it blanks are anchored so that they cannot carry a private word (E2);
positions, quantities, cost basis and money amounts are withheld (E3), and so is a thesis with a character the lists
cannot read (E4). The leak sentences of cold-check r2 (R2_LEAKS) are pinned verbatim. meta.unavailable reasons are
plain Vietnamese without internal codes; the details go to stderr (E5).

Review of round r3 (evidence/vn30-web-20261103/r3/W-exporter/review-fix): the reviewer's probe sentences
(REVIEW_LEAKS) are pinned verbatim; the phrases the review named match with a space or a dash between their words; each
chart phrase is blanked only in its chart context (E2_PRIVATE has a private sentence around each context rule); the
characters the lists can read are an allowlist, in the text as written and after NFKC (E4); and every single rule,
alternative and guard of the filter is locked: taking any one of them out turns a test red (ONE_RULE, one sentence per
rule, and the tuples above; review-fix/mutants.py).

Round r4 (evidence/vn30-web-20261103/r4/contract.md, findings of cold-check r3): a thesis is published only for a note
with actionBasis market-only or insufficient-input and actionView wait or none, every other value, a missing field or a
value that is not a string withholds it and keeps the row (X1; the fixture notes carry the daily task's wait /
market-only); a number that ends with a comma is a list, not "<number> <ticker>" (X2); break even, under water and the
phrases margin call, net asset value, average price, target price and off the table also with a space or a dash, and
VND amounts written with "dong" or without thousands separators (X3); the module docstring says which phrases take a
dash (X4: SPACE_OR_DASH, SPACE_ONLY).

Review of round r4 (evidence/vn30-web-20261103/r4/review-W-exporter, fixes in r4/W-exporter/review-fix): the money
forms the docstring names are caught (a dash before a scale word or a currency word, "đ" and "dong(s)" after k or m, a
space between the thousands, decimals, "Vietnamese"/"VN dong", a colon after VND, USD before a number, k or m before
USD or dollar(s)) and the one it names as not caught is published ("63,000,000 in VND terms"); a list of counts written
after the tickers with a space before each count ("MBB 3,000, BID 1,500") is withheld again (round r4b widens this:
see below); "broke even lower" (higher, further, more, deeper) is chart prose; the cheap word forms of margin call, net
asset value, average price and off the table are listed; and the docstring and data-contract.md give the same
examples, each pinned to what the filter does (DOC_WITHHELD, DOC_PUBLISHED; data-contract.md is read only in the btc
workspace).

Round r4b (evidence/vn30-web-20261103/r4b/contract.md, findings of cold-check r4): what r3 withheld in the two families
r4 relaxed is withheld again, and what r4 opened on purpose stays published (X2_LISTS). B1: "break-even" and
"breakeven" (the noun) withhold before lower, higher, further, more and deeper too; only the verb written with a space
("broke even lower") is chart prose (BREAK_EVEN_NOUN, BREAK_EVEN_PROSE). B2: a list of counts written after the
tickers withholds in any of the connectors of _COUNT_AFTER_TICKER (a space, ":", ";", ",", "(", "[", "~", "×", "=",
"≈", " x ", a spaced dash, about, roughly, around, some, approximately, approx., nearly, almost, or two of them), and so
do a comma glued to the ticker ("2,600,MBB") and an unsigned count written first at the end of a clause ("3,000,
MBB."); a sign, "'s", at, near or another word after the ticker is still a list (X2_COUNTS, X2_LISTS). B3: the VND
amount rules start at seven digits (VND_EDGE). B4: b (billion) before a currency word and after VND/₫, and "Vietnam" /
"Viet Nam dong" (COLDCHECK_R4_LEAKS, X3_PRIVATE, X3_TECHNICAL). B5: the docstring and data-contract.md state the limit
of the action check and that kept theses passed it from r4 on, and name the connectors the filter reads
(test_the_docstring_names_the_connectors_of_a_count_and_the_limit_of_the_action_check and the data-contract test). The
sentences of cold-check r4 are pinned verbatim, alone, in a carrier and inside a real thesis (COLDCHECK_R4_LEAKS).

Review of round r4b (evidence/vn30-web-20261103/r4b/review-W-exporter, fixes in r4b/W-exporter/review-fix): the
optional space before k, m or b is pinned in each of its three rules (X3_PRIVATE, DOC_WITHHELD); the country's name
(Vietnamese, Vietnam, Viet Nam, Việt Nam, VN) also stands between k, m or b and the currency word, and Việt Nam also
after an amount of a million or more, while a price written with it stays published (X3_PRIVATE, X3_TECHNICAL);
approx without its dot and two dashes with a space on each side are connectors of a count (X2_COUNTS); whitespace
that ends the text ends the clause, and a count in double quotation marks withholds
(test_an_unsigned_count_written_first_withholds_at_the_end_of_the_text_too); the docstring and data-contract.md also
pin the skill's default and where the theses exported before r4 came from, and name what the chart exception of the
spaced "broke even" lets through (DOC_PUBLISHED). The reviewer's sentences are pinned verbatim like the cold-check ones
(REVIEW_R4B_LEAKS)."""
import io
import json
import os
import re
import sys
import urllib.error
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

AI_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(AI_ROOT))

import export_vn30_web as web  # noqa: E402

KEY = "test-admin-key-0123456789"
SESSION = "2026-11-02"
SYMBOLS = ("VIC", "MBB", "VCB", "ACB")  # the screen's rank order, not alphabetical
KEPT = "; giữ dữ liệu lượt trước"
NOTHING_KEPT = "; chưa có dữ liệu lượt trước"
# meta.unavailable reasons as the public site shows them (contract r3 E5), typed here rather than imported
UNSUPPORTED = "backend chưa hỗ trợ mã này"
NO_KEY = "thiếu quyền đọc nhật ký nhận định"
NO_STRENGTH = "dữ liệu VNINDEX chưa khớp tới phiên mới nhất của mã"
NO_INDICATORS = "backend chưa có chỉ báo 1d"
MALFORMED = "phản hồi sai định dạng"
INDICATOR_NAMES = ["sma50", "sma200", "ema12", "ema26", "ema50", "ema200", "rsi14", "macd", "macdSignal",
                   "macdHistogram", "bollingerUpper", "bollingerMiddle", "bollingerLower", "atr14"]
META_KEYS = ["schema", "generatedAtUtc", "session", "symbols", "unavailable", "source"]
SCREEN_KEYS = ["session", "basket", "indexReturns", "rankingMonths", "topK", "missingSessionBar", "notUpToDate", "rows"]
ROW_KEYS = ["symbol", "close", "returns", "excessVsIndex", "rank", "top"]
SYMBOL_KEYS = ["symbol", "label", "isBank", "candles", "indicators", "relativeStrength", "notes", "score"]
INDICATOR_KEYS = ["date", "close", *INDICATOR_NAMES]
NOTE_KEYS = ["signalDate", "horizonSessions", "direction", "thesis", "status", "pendingReason", "targetDate",
             "returnPct", "hit"]
SCORE_KEYS = ["scored", "hits", "baselineHits", "sampleStatus"]
FORBIDDEN = {"actionView", "actionBasis", "probability", "scenarios", "evidence", "counterEvidence", "snapshotId",
             "snapshotSha256", "noteId", "mode", "requestedAsOfUtc", "skillVersion", "methodVersion", "forecastIds",
             "limitations", "missingData", "outcomeCheck", "signalClose", "targetClose", "realized", "baselineCall",
             "methods", "obv", "vwap", "macdNorm", "noteBinding", "timeIso", "caveat", "anchorSessions",
             "unknownToBackend", "ranks", "position", "cash", "body", "horizons", "items", "resolved", "baselinePrior",
             "hitRate", "nEffective", "configId", "bestConstantInHindsight", "pendingReasons", "rule", "totalNotes"}
DIRS = ("up", "down", "sideways")
# Planted in one fixture thesis: the filter must withhold that thesis, so none of these may reach any output file.
THESIS_SECRETS = ("100.83.92.101:5197", "someone@example.com", "D:\\code\\btc\\data\\research-sessions",
                  "120,000,000 VND", "f0e364fd")
PLANTED_THESIS = ("MBB 1d closed below SMA50. User holds 3,000 MBB at avg cost 21,500, cash 120,000,000 VND; see "
                  "100.83.92.101:5197 and D:\\code\\btc\\data\\research-sessions\\MBB, note session/MBB/1792108800000/"
                  "f0e364fd, mail someone@example.com.")
LONG_THESIS = ("The 1d close sits above SMA50 while RSI14 stays near 50 and the 20-session range is intact. " * 25)[:2000]
EXACT_THESIS = ("Bollinger middle band 18,900 caps the 1d close; ATR14 is flat. " * 30)[:1500]
# The three theses published in web/data on 2026-11-03 (copied verbatim from symbols/MBB.json and BID.json).
REAL_THESES = (
    "MBB 1d signal bar 2026-11-02 closed 18750 (-2.85% on the day, volume 2.1x its 20-session average), the low end "
    "of the 100-bar window (20/60-session low 18600 touched intraday). Trend states are bearish (close below SMA50 "
    "19877 and below the 100-bar volume-profile VAL 18852, EMA12 < EMA26, MACD histogram negative), but momentum is "
    "stretched rather than accelerating: Wilder RSI14 34.0, close near the lower Bollinger band 18730, 1h Wilder "
    "RSI14 21.3. Relative strength vs VNINDEX is flat over 20 sessions. Mixed evidence over 20 sessions: view "
    "sideways (R between -4% and +5%), for reference only.",
    "MBB 1d signal 2026-10-30 close 19300 (adjusted). Mixed technical picture: 1d regime is sideways/high-vol, price "
    "has held the 20-session range 18650-20100 (position 0.45) while sitting below SMA50 (-3.0%) and SMA200 (-5.9%); "
    "1h is in a bear/high-vol regime and the signal bar closed at its low (BearishMarubozu). 1d MACD histogram turned "
    "slightly positive while EMA12 < EMA26. Neither the +5% (>= 20265) nor the -4% (<= 18528) H20 barrier is "
    "favoured by the evidence: both lie just outside the 20-session range. View: sideways over 20 sessions, for "
    "reference only.",
    "BID 1d signal bar 2026-11-02 closed 32000 (-4.19% on the day, volume 2.85x its 20-session average), a new low "
    "of the 100-bar window (intraday low 31700), below the lower Bollinger band 32625. Trend is clearly bearish "
    "(20-session -7.5%, 60-session -10.0%, close below SMA50 35177 and SMA200 38502, EMA12 < EMA26, underperforming "
    "VNINDEX by 5.1 points over 20 sessions), but the move is stretched: Wilder RSI14 28.0 on 1d, 1h Wilder RSI14 "
    "25.1 (Cutler 7.0). Continuation and mean reversion both have support; over 20 sessions the view is sideways (R "
    "between -4% and +5%), for reference only.",
)
# Round r4: the three newer theses of the real export of 2026-11-03 (evidence/vn30-web-20261103/r3/gate/real-export),
# copied verbatim from their notes: MBB and BID 20261103-2002 (skill v7) and ACB 20261103-2023 (v6). Each comes from a
# forward note with actionView wait and actionBasis market-only, like the three above.
R4_THESES = (
    # MBB/20261103-2002-technical
    "Technical-only view over the next 20 sessions. The daily close on 2026-11-03 was 18900, a 0.8% gain on the "
    "session that followed a 2.8% decline on 2026-11-02 with volume 2.1x its 20-session average; the latest session "
    "carried 1.7x its average. Over five sessions the close fell 3.8%, over 20 sessions 1.6% and over 60 sessions "
    "5.1%. The close sits below EMA12 at 19387, below EMA26 at 19524, below SMA50 at 19849 and below SMA200 at "
    "20481, and the daily MACD is negative with a widening negative histogram. Wilder RSI14 is 37.6 on the daily "
    "bars, while the simple-average RSI14 in the conditions module reads 42.1, both in the neutral zone. The latest "
    "bar has a hammer shape with a low of 18550, which is also the 20- and 60-session low; range position is 0.23 "
    "of the 20-session range and 0.13 of the 60-session range. The close is just above the 100-bar value-area low "
    "near 18852 and below the point of control near 20133. The regime label is sideways with normal volatility. "
    "Relative to the market index the 20-session return is 1.8 points better and the 60-session return 1.4 points "
    "worse; against the bank peers it is ahead of two of three on both windows. The daily trend is weak but the "
    "decline met heavy volume near a prior low, so the evidence is mixed. The view is sideways: a move to plus 5% "
    "near 19845 or minus 4% near 18144 within 20 sessions is not yet confirmed by the candles. For reference only; "
    "this kind of call has not beaten an always-sideways answer in blind tests.",
    # BID/20261103-2002-technical
    "Technical-only view over the next 20 sessions. The daily close on 2026-11-03 was 31700, down 0.9% on the "
    "session after a 4.2% decline on 2026-11-02 that carried volume 2.85x its 20-session average; the latest "
    "session carried 1.4x its average. Over five sessions the close fell 5.8%, over 20 sessions 8.4% and over 60 "
    "sessions 10.6%. The low of 31600 is the lowest of the 100-bar window, and range position is 0.03 of the "
    "20-session range. The close is below EMA12 at 33243, below EMA26 at 33905, below SMA50 at 35094 and below "
    "SMA200 at 38433, and below the lower Bollinger band near 32246. The daily MACD is negative with a widening "
    "negative histogram. The close is under the 100-bar value-area low near 33582 and well under the point of "
    "control near 36464. Wilder RSI14 on the daily bars is 26.3, in the oversold zone, while the simple-average "
    "RSI14 in the conditions module reads 36.8, in the neutral zone; the two readings disagree and both are "
    "reported. The regime label is still sideways with normal volatility. Relative strength is weak: over 20 "
    "sessions the return trails the market index by 5.0 points and trails each of the three bank peers. The daily "
    "structure is a decline to a new window low on expanding volume, so the view is down, meaning a 20-session "
    "return at or below minus 4% near 30432. The stretched oscillator and the close below the lower band leave room "
    "for a rebound first. For reference only; this kind of call has not beaten an always-sideways answer in blind "
    "tests.",
    # ACB/20261103-2023-technical
    "Daily candles only. On 23 October a wide green candle on volume 6.0x its 20-session average lifted the close "
    "to 21,300; four flat sessions capped at 21,450 followed before a full retracement: down 1.4% on 30 October and "
    "3.3% on 2 November, when the session opened 1.7% lower on volume 1.9x its average and printed a 60-session low "
    "of 20,050. On 3 November the close was 20,250, with a small body, a lower shadow half the candle's range and "
    "volume 0.7x its average. Support at 20,050 to 20,200 has held since 9 October, with no daily close below "
    "20,150. The trend leans down: EMA50, SMA200, EMA200 and SMA50 sit 4.9% to 5.8% above the close, EMA12 is below "
    "EMA26, MACD is negative with a slightly positive histogram, the regime label is a low-volatility downtrend, "
    "and the close is under the 100-session volume-profile value area (VAL 22,029, POC 22,346). Wilder RSI14 is "
    "36.5 and a simple-average RSI14 is 46.3, both neutral; ATR14 is 2.1% of the close. Over 20 sessions the stock "
    "fell 1.7% against 3.4% for the VNINDEX; over 60 sessions, 8.6% against 3.7%. View for the next 20 sessions: "
    "sideways, a close 20 sessions later between 19,440 and 21,262.5 (-4% to +5%). The -4% line lies 3% under the "
    "20,050 low; the +5% line falls inside the cluster of averages where the late-October advance stalled. A daily "
    "close below 20,050 would invalidate the view in favor of down; one above 21,450, in favor of up. Reference "
    "only: such calls showed no edge in blind tests and were never tested on this stock.",
)
TECHNICAL_THESES = REAL_THESES + R4_THESES + (
    "Buyers stepped in at the 20-session low and the support holds; price held above SMA50.",
    "Selling pressure eased after a sell-off on 2.1x volume; buying pressure returned near the lower band.",
    "RSI14 is oversold on 1d and overbought on 1h; short-term momentum is weak and the long lower shadow shows demand.",
    "The range position is 0.45 (position 0.45, i.e. position within the 20-session range) after a selloff.",
    "Money flow and accumulation/distribution diverge; volume-weighted average price 18,900; +5%/-4% barriers.",
    "Price stopped at the 60-session low and rebounded; sellers were absorbed, e.g. on 2026-11-02 (R(20) inside).",
)
# The strings cold-check round 1 planted in a thesis (coldcheck-r1.md, lens-leak and its triage).
COLDCHECK_PRIVATE = (
    "http://127.0.0.1:5197/api/equities/MBB/technical",
    "10.11.12.13:5197",
    "D:\\code\\btc\\data\\research-sessions\\MBB\\20261102-1726-technical",
    "C:\\Users\\someone\\AppData",
    "snap/MBB/1793346300000/2bb65c32d3d5",
    "session/MBB/1792108800000/f0e364fd",
    "0f" * 32,
    "someone@example.com",
    "User holds 3,000 MBB at avg cost 21,500 (T+2 sellable 1,000); stop -8%; cash 120,000,000 VND, buy budget 50m",
    "View: buy more",
    "Hành động: hold - không bán bớt; người đang giữ giữ trên 19.500",
)
PRIVATE_THESES = COLDCHECK_PRIVATE + (
    "for a holder the market view alone supports hold", "Market-only: for new money wait for a close above 20,100",
    "the stop at the low of 18,600", "your position is 2,000 shares", "portfolio weight 10%",
    "take profit at 21,000", "cost basis 21,500", "T+2 sellable", "reduce on a close below 18,500",
    "add on a retest of 18,600", "Khuyến nghị mua thêm khi vượt 20.100", "cắt lỗ dưới 18.500", "danh mục chưa có",
    "tiền mặt 50 triệu", "ngân sách 100 triệu", "chờ xác nhận", "btc-be.example.com", "see vietstock.vn",
    "localhost:3000", "100.83.92.101", "fe80::1", "(/api/equities/MBB/candles)", "~/code/btc/ai", "D:\\work\\notes",
    "saved under D:/notes", "noteId attached",
    "X-Admin-Key header", "ai/skills/equity-technical/SKILL.md",
    "b\u200buy now",  # a zero-width space inside the word
    "ba\u0301n bớt",  # Vietnamese typed decomposed (NFD)
    "\uff42\uff55\uff59 the dip",  # full-width letters
    "T\uff0b2",  # a full-width plus sign
)
# Cold-check r2 (coldcheck-r2.md): every distinct leak sentence of the main reviewer's, the privacy lens's and the
# triages' probes, each published by the r2 filter or used there as a leak control; frozen from their scripts by
# evidence/vn30-web-20261103/r3/W-exporter/collect_vectors.py and gen_test_vectors.py. Left out on purpose (the
# filter does not claim them, contract r3 E4/E6): "BUY_ZONE 18600" and "buy18600" (a word glued to "_" or a
# digit), "see the backend on my-pc" (a free-form host name) and "127.1" (an abbreviated IPv4 address reads as a
# decimal number).
R2_LEAKS = (
    # reviewer
    "Consider adding on dips toward 18,600.", "Adding on a retest of 18,600 is reasonable.",
    "Reducing into strength above 20,100.", "Trimming above 20,100 makes sense.", "Exiting on a close below 18,500.",
    "Entering on a close above 20,100.", "Waiting for a close above 20,100 is prudent.", "Taking profits near 21,000.",
    "Profit-taking near 21,000 is likely.", "Average down below 18,500.", "Scale in on weakness.",
    "Lighten up above 20,100.", "Liquidate on a close below 18,500.", "We are long from 24,000.", "We own 2,000 MBB.",
    "Position: 45% of NAV.", "position 45%", "Investors should consider accumulating.", "Short it below 18,500.",
    "Hedge the exposure", "added 1000 MBB yesterday", "mua them khi vuot 20.100", "ban bot tren 20.100",
    "cat lo duoi 18.500", "xả hàng trên 20.100", "nắm giữ tiếp", "giá mục tiêu 21.000", "buy", "buys", "buying",
    "bought", "sell", "sells", "selling", "sold", "hold", "holding", "reduce", "reduces", "reduced", "reducing", "add",
    "adds", "added", "adding", "wait", "waiting", "exit", "exiting", "enter", "entering", "trim", "trimming",
    "take profit", "taking profits",
    # lens
    "Waiting for a daily close above 20100 before acting is the more conservative choice.",
    "Staying on the sidelines until a close above 20100 is prudent.",
    "New money is better kept on the sidelines until the range resolves.", "Consider reducing on a close below 18600.",
    "Reducing into strength near 20100 would be reasonable.", "Adding near the 18600 support is premature.",
    "Accumulating on dips toward 18600 is reasonable.",
    "Trimming into strength near 20500 locks in part of the rebound.",
    "Taking profits near the upper Bollinger band is sensible.", "Profit-taking near 21000 is advisable.",
    "Exiting on a break of 18600 limits the downside.", "Entering here offers a poor risk/reward.",
    "Purchasing near 18600 is better than chasing 20100.", "Averaging down below 18600 is not advised.",
    "Avoid catching the falling knife until the price bases.",
    "A long from 18700 targets 20100 with invalidation at 18500.", "Get long near 18600 and get out below 18500.",
    "Lighten up near 20500; load up again near 18600.", "Place a limit order at 18600.",
    "Bids at 18600 and 18300 are reasonable; offer at 20500.",
    "Investors already in the stock can keep it while price stays above 18600.", "Stay long while 18600 holds.",
    "Liquidate below 18500.", "Bail out on a close below 18500.", "Cuts losses quickly below 18500.",
    "Open a position in the 18600-18900 range with invalidation below 18500.",
    "Build a position within the 18600-18900 band.", "Scale into a position in the lower band.",
    "Start a position in a buy range of 18600-18900.", "MBB position: 30% of NAV.",
    "Buying power of 50 million VND remains.", "Remaining buying power is 120m VND, enough for 3,000 more.",
    "Sell-off half at 20500.", "The 3,000-share lot acquired at 21,500 is 12.8% underwater.",
    "The average purchase price of 21,500 is 12.8% above the close.", "My average price is 21,500.",
    "Down 12.8% from the 21,500 average paid.", "3,000 MBB are owned at 21,500.",
    "With 3,000 MBB on the books at 21,500, the drawdown is 12.8%.", "Of the 3,000 MBB, 1,000 settle on Wednesday.",
    "The account has 120 million VND available.", "Funds available: 120 million VND.",
    "Dry powder of 50m VND is set aside for MBB.", "Weight: 30% of NAV.", "Long 3,000 MBB since 21,500.",
    "Đợi giá đóng cửa trên 20.100 rồi mới tham gia.", "Có thể bắt đáy quanh 18.600.",
    "Canh vào hàng khi giá về 18.600.", "Thoát hàng nếu thủng 18.600.", "Xả bớt khi chạm 20.500.",
    "Trung bình giá xuống quanh 18.600.", "Đang âm 12%.", "Tiền nhàn rỗi 120 triệu.",
    "Cân nhắc chốt một phần quanh 20.500.", "Cắt giảm khi thủng 18.600.", "Quan sát thêm, chưa tham gia.",
    "Gia tăng quanh 18.600.", "Nắm 3 nghìn MBB từ 21.500.", "Mức dừng 18.500.",
    "nen ban ra, cho xac nhan, cat lo duoi 18.500", "BUY.", "SELL!", "(hold)", "Wait-and-see.", "buy/sell",
    "action=buy", "MUA", "BÁN", "GIỮ", "CHỜ", "ĐỨNG NGOÀI", "Mua.", "bán,", "giữ;", "\u0441ell near 20500",
    "b\u0443y the dip", "\u202eyub", "s\u00adell", "\U0001d41b\U0001d42e\U0001d432", "ba\u02can", "POS\u0130T\u0130ON",
    "Blind tests (evidence/blind-test-20261030) found no edge.", "levels from scripts/levels output", "DESKTOP-7QK2LM",
    "the previous call f0e364fd", "binding 2bb65c32d3d5", "requested as of 1792315293201",
    "someone at example dot com", "backend at :5197", "loopback backend", "Take\u2011profit at 21000.",
    "Cut-loss at 18500.", "Cutloss below 18500.", "Keep buying power in reserve until a close above 20100.",
    "Stand aside until a close above 20100.", "Patience: no new money until a close above 20100.",
    "For existing longs, 18500 is the line in the sand.", "For new money, the better entry is lower.",
    # triage1
    "Reduce on a close below 18600.", "Wait for a daily close above 20100.", "Add on a retest of 18600.",
    "Take profit near 20500.", "Cut losses below 18500.", "Exit on a break of 18600.", "Chờ giá đóng cửa trên 20.100.",
    "Chốt lời quanh 20.500.", "Exposure is best reduced on a close below 18600.",
    "The view waits for a daily close above 20100.", "Adding on a retest of 18600 is reasonable.",
    "Trimming into strength near 20500 is sensible.", "Entering near 18600 offers a tight invalidation.",
    "Profit-taking near 20500 is sensible.", "Take\u2013profit at 21000.", "Stopped out below 18500.",
    "Stay on the sidelines until a close above 20100.", "Scale in near 18600 and scale out near 20500.",
    "Average down only above 18500.", "Lighten up near 20500.", "Open a long near 18600 with invalidation at 18500.",
    "Bottom-fishing near 18600 is premature.", "Hạ tỉ lệ nắm giữ khi thủng 18.600.", "Đặt lệnh ở 18.600.",
    "Tham gia khi giá vượt 20.100.", "Đứng ngoài cho tới khi vượt 20.100.",
    # triage2
    "Open a position in the 18,600-18,900 range.", "Open a position in the 18.600-18.900 range.",
    "Open a position in the range.", "Hold a position in the range.", "Add a position in the lower band.",
    "Take a position in the support band near 18600.", "Initiate a position in a 3% band.", "MBB position 30%.",
    "Position = 45% of NAV.", "Selling power is limited to 1,000.", "My buying power is 50m.",
    "Our position in the MBB channel is 30% of NAV.", "We sold out; position: 0% now.",
    "Open a position in the 18600-18900 range with invalidation below 18500",
    "Build a position within the 18600-18900 band", "Scale into a position in the lower band",
    "Start a position in a buy range of 18600-18900", "Buying power of 50 million VND remains",
    "Remaining buying power is 120m VND, enough for 3,000 more", "MBB position: 30% of NAV",
    "Keep buying power in reserve until a close above 20100", "Open a position in the range",
    "Build a position within the 20-session range", "Take a position in the 60-session band near 18600",
    "Selling power is limited to 1,000",
    # triage3
    "Buying power: 120 million VND.", "Available buying power is 120 million VND.", "Position: 30% of NAV.",
    "Purchase price 21,500; the close is 12.8% below it.", "Acquired 3,000 MBB at 21,500.",
    "VND 120m is available for MBB.", "1.2 bn VND sits idle.", "3,000-share position",
    "Giá trung bình 21.500, đang âm 12%.", "Số dư tài khoản 120 triệu.", "Sở hữu 3.000 MBB giá 21.500.",
    "Còn 1,2 tỷ đồng chưa dùng.", "User holds 3,000 MBB at avg cost 21,500.", "Your 3,000 shares at 21,500.",
    "The 3,000 shares acquired at 21,500.", "avg price 21,500", "cost basis 21,500", "cash 120 million VND",
    "tiền mặt 120 triệu", "giá vốn 21.500", "đang nắm 3.000 cp", "Số dư 120.000.000 đồng.",
)
# The reviewer's probe sentences of round r3 (review-W-exporter/probe_leaks.py, groups A-F), every one published by the
# reviewed filter; pinned verbatim like R2_LEAKS.
REVIEW_LEAKS = (
    # A: a form of the canonical list
    "We shorted MBB near 20,100", "Got stopped-out below 18,500 on the last attempt", "Use a limit-order at 18,600",
    "A scale-in near 18,600 keeps the risk small", "Take quick profits near 21,000", "TP 21,000, SL 18,500",
    "Loss-cut at 18,500", "Reentry above 20,100 is cleaner", "Rebuy near 18,600",
    "An average-down below 18,500 is not advised", "We are paying up above 20,100",
    "The book has gotten long near 18,600", "Stopping out below 18,500 is the plan", "Market-orders at the open",
    "Scaled-out near 20,500", "Take early profits near 21,000", "Taking small profits near 21,000",
    "Resold near 20,500", "Reentered above 20,100",
    # B: synonyms of the listed actions
    "Long above 20,100, short below 18,500", "Short the bounce toward 20,100",
    "Fade rallies toward 20,100 while price is below SMA50", "Lock in gains near 20,500", "Book profits near 21,000",
    "Take some money off the table near 21,000", "Step aside until a close above 20,100",
    "Stay away until a close above 20,100", "Fresh money is better deployed above 20,100", "Place an order at 18,600",
    "Offer at 20,500", "Get in near 18,600", "Pick up shares near 18,600", "Dump the stock below 18,500",
    "Close the trade below 18,500", "Keep MBB while the support holds", "Go flat below 18,500",
    "Cover shorts near 18,600", "Double down below 18,500", "Acquired at 21,500 in October",
    "Favor longs above 20,100", "Shorts are favored below 18,500",
    # C: position, quantity, cost basis, money
    "Our weighted average price is 21,500", "My average share price is 21,500", "Average fill price 21,500",
    "MBB is on our books at 21,500", "Weight: 30% of net asset value", "3K MBB remain", "We own MBB from 21,500",
    "We are invested since 21,500", "We got in at 21,500", "Filled at 21,500 last week", "Our basis is 21,500",
    "Underwater by 12.8% from 21,500", "Three thousand shares at 21,500", "2,000 units of MBB at 21,500",
    "Our stake is 2% of equity", "Half size near 18,600, full size above 20,100", "$5,000 is set aside for MBB",
    "120 mil VND set aside", "Our loss is 12.8% since 21,500", "MBB: 3,000 at an average of 21,500",
    # D: Vietnamese
    "Có thể xuống tiền quanh 18.600", "Tiếp tục ôm hàng trên 18.600", "Cầm tiếp nếu còn trên 18.600",
    "Nhặt dần quanh 18.600", "Tất tay quanh 18.600", "Thoát nếu thủng 18.600", "Cắt nếu thủng 18.600",
    "Chốt hết khi chạm 20.500", "Đang lời 8%", "Đang có 3 nghìn cổ MBB", "Còn 120 củ chưa dùng", "Tài khoản còn 120tr",
    "Con 120 trieu tien mat", "Còn 120.000.000đ chưa dùng", "Giá bình quân 21.500", "Hàng về ngày thứ Tư",
    "bat day quanh 18.600", "chot mot phan quanh 20.500", "Lướt sóng trong biên 18.600-20.100",
    # E: a chart phrase around a private word
    "Build a small position 0.5 of full size near 18,600", "Keep the position 0.5 of normal size",
    "Open a range position of 0.5 near 18,600", "Stay patient and position in the 20-session range near its low",
    "Close position in the 20-session range near 20,100", "Our buying interest sits near 18,600",
    "We have buying interest near 18,600", "Plan a partial sell-off near 20,500", "Then sell-off half at 20,500",
    "Plan: position in the 20-session range near its low", "Build a starter position 0.3 near 18,600",
    # F: Latin look-alikes, spaces and dashes outside the readable set
    "\u0299\u1d1c\u028f the dip", "\ua731\u1d07\u029f\u029f near 20,500", "ho\u01c0d above 18,600",
    "\u0251dd near 18,600", "sell\u0131ng near 20,500", "b\u200auy the dip", "s\u2009ell near 20,500",
    "\u249d\u24b0\u24b4 the dip", "\U0001f171\U0001f184\U0001f188 the dip", "take\u2500profit at 21,000",
    "cut\u23afloss at 18,500", "the 3,000\u2500share lot",
)
# Contract r3 E1: every word of the canonical action list (the equity-technical skill names the same list) in every
# form, and the Vietnamese additions; withheld inside a technical sentence.
E1_FORMS = (
    "buy", "buys", "buying", "bought", "sell", "sells", "selling", "sold", "hold", "holds", "holding", "holder",
    "holders", "reduce", "reduces", "reduced", "reducing", "add", "adds", "added", "adding", "accumulate",
    "accumulates", "accumulated", "accumulating", "trim", "trims", "trimmed", "trimming", "enter", "enters", "entered",
    "entering", "entry", "entries", "exit", "exits", "exited", "exiting", "wait", "waits", "waited", "waiting", "stop",
    "stops", "stop-loss", "stopped out", "take profit", "takes profit", "taking profits", "took profit",
    "taken profits", "take some profit", "profit-taking", "profit taking", "cut loss", "cuts losses",
    "cutting losses", "cut-loss", "cutloss", "stand aside", "stands aside", "standing aside", "stood aside",
    "sideline", "sidelines", "sidelined", "new money", "limit order", "limit orders", "market order", "market orders",
    "scale in", "scales out", "scaled in", "scaling out", "scale into", "average down", "averages down",
    "averaged down", "averaging down", "go long", "go short", "going long", "goes short", "went long", "shorting",
    "short-sell", "short sell", "recommend", "recommends", "recommended", "recommendation",
    # review r3: a dash between the words of a phrase, more words between take and profit, more forms
    "stopped-out", "stopping out", "stopping-out", "take quick profits", "take early profits", "taking small profits",
    "loss-cut", "losscut", "stand-aside", "new-money", "limit-order", "market-orders", "limitorder", "scale-in",
    "scaled-out", "average-down", "go-long", "gone short", "shorted", "side-line", "sidelines",
)
E1_VI = ("đợi", "xả", "bắt đáy", "vào hàng", "thoát hàng", "tham gia", "chốt lời", "chốt lãi", "chốt một phần",
         "chốt bớt", "nắm", "sở hữu", "nhàn rỗi", "số dư", "trung bình giá")
# Left out of the list on purpose, so published: "held" (everyday chart language), a bare "stopped", "accumulation".
E1_NOT_LISTED = (
    "Support held at 18600 twice and the low held again on 2026-11-02.",
    "Price stopped at the low of the day.",
    "Accumulation/distribution turned up while selling pressure eased.",
    # "long"/"short" before "-term" are chart words, not the position phrases of ACTION_EXTRAS
    "RSI14 is already short-term oversold and the trend stays long-term bearish.",
    "Giá vào vùng 18.600 rồi bật lên.",  # "giá vào vùng": the price reaches a zone, not an entry price
)
# More action words and position phrasings of the same kind, each the only listed word of its sentence (the cold-check
# r2 leak sentences in R2_LEAKS use them inside longer text).
ACTION_EXTRAS = (
    "awaiting a close above 20,100", "Purchasing near 18600 is better than chasing 20100", "hedge below 18,500",
    "Liquidate below 18500", "Get long near 18600", "get out below 18500", "Lighten up near 20500",
    "load up near 18600", "Bail out on a close below 18500", "Bottom-fishing near 18600 is premature",
    "Avoid catching the falling knife", "Open a long near 18600 with invalidation at 18500", "Remain long from 24,000",
    "For existing longs, 18500 is the line in the sand", "Investors already in the stock can keep it",
    "Stay long above 18600", "Stay flat until a close above 20,100", "Short it below 18,500", "We're long",
    "price target 21,000",
    "Bids at 18600 and 18300 are reasonable; offer at 20500", "giá mục tiêu 21.000", "Cắt giảm khi thủng 18.600",
    "Gia tăng quanh 18.600", "Mức dừng 18.500", "Đang âm 12%", "nen ban ra, cho xac nhan, cat lo duoi 18.500",
    "the backend on loopback", "DESKTOP-7QK2LM", "someone at example dot com", "backend at :5197",
    "Blind tests (evidence/blind-test-20261030) found no edge", "the previous call f0e364fd",
    "binding 0123456789abcdef0123", "requested as of 1792315293201",
    "s\u0336ell near 20500",  # a combining mark outside the readable set withholds (review r3: READABLE)
    "the 3,000 M\u0336BB lot",  # ... before the case-sensitive rules would read it too
    "b\u0301uy the dip",  # a readable combining mark that composes with nothing is removed before the lists read it
    "the 3,000 MB\u0301B lot",  # ... before the case-sensitive rules read it too
)

# Contract r3 E2: each technical phrase the filter blanks, in the writer's words and in each place it may stand, is
# published unchanged ...
E2_TECHNICAL = (
    "Excess -3.04% vs VNINDEX; position in 60-session range 0.255.",  # a real claim, MBB 20261030-1535
    "Price is mid-range (60-day position 0.34) and position within its 20-session range is 0.31.",
    "The close sits at a range position of 0.45; channel position 0.8 on 1h.",
    "Buying climax volume faded after the gap and selling exhaustion followed near the lower band.",
    "Volume rose on the day (sell-off, 2.1x its 20-session average); the 2026-10-29 sell-off broke the range.",
    "If the 18600 level holds, the 20-session range stays intact; the 60-session low holds too.",
    # review r3: every place of CHART_BEFORE, each in a sentence of its own
    "Position 0.45 of the 20-session range.", "The close is at position 0.45.", "The close is position 0.45.",
    "The close sits mid-range, position 0.45.", "The close sits mid-range: position 0.45.",
    "The close sits mid-range. Position 0.45 is the reading.", "The close sits in the top half of position 0.9.",
    "The 20-session position 0.45 is mid-range.", "The 20-bar position 0.45 is mid-range.",
    "Range position 0.45 on 1d.", "The range position was 0.45 on 1d.", "Its range position 0.45 is mid-range.",
    "RSI14 is 34 and range position is 0.03 of the 20-session range.",  # the v7 thesis BID 20261103-2002
    "RSI14 is 34 with range position 0.45.", "The close is at a range position of 0.45.", "It is the range position 0.45.",
    "It was a range position of 0.45.", "RSI14 is 34 with a range position of 0.45.", "The 60-day range position 0.4.",
    "RSI14 is 34 and the range position is 0.4.", "The close of a range position 0.45.",
    "The band position at 0.9 is high.", "Its 60-session band position is 0.2.",
    "Position in the 20-session range 0.25 on 1d.", "The close's position in the 20-session range is 0.25.",
    "Price position in the 20-session range 0.25.", "Its position in the 20-session range is 0.25.",
    "The close is position in the 20-session range 0.25.", "The close is at position in the 20-session range 0.25.",
    "A reading of position in the 20-bar band 0.25.", "Mid-range (position in the 20-session range) on 1d.",
    "Buying pressure and selling flows, buying waves, selling momentum and buying volume all eased.",
    "After the sell-off on 2026-11-02 the close fell 4%.", "Its sell-off broke the low.", "That sell-off of 4% ended.",
    "This sell-off; then a doji.", "The 2026-11-02 sell-off, on 2.85x volume.", "A 4.2% sell-off on heavy volume.",
    "The low held after a sell-off", "The support holds; the floor holds; the zone holds; resistance holds; range holds.",
)
# ... and a private sentence built around each one is withheld: the blanked phrase never takes the private word along.
E2_PRIVATE = (
    "Position: 30%",  # "position N%" is a weight, never a range position
    "Build a position 0.5 of full size near 18600", "Keep our position 0.5 of full size",
    "Open a range position near 18600",  # "<range> position" only with its fraction
    "Build a small position within the 20-session range",  # "position in the N-session range" only after "(", ...
    "Open a position in the 18600-18900 range", "Take a long position in the 20-session range",
    "Keep buying power in reserve until a close above 20100",  # never "buying power"
    "Sell-off half at 20500",  # "sell-off" only as a noun
    "Hold above 18600 while the support holds",  # "<support> holds" blanks two words, never a third
    # review r3: one sentence per context rule, which only that rule withholds
    "Build a starter position 0.3 near 18,600", "Build my position 0.5 of full size", "Open a long position 0.5",
    "Keep the short position 0.5", "Take an position 0.5 near 18,600",  # frac: not after another word
    "Open a range position of 0.5 near 18,600", "Keep the range position 0.5 near 18,600",  # rpos: nor after a verb
    "Stay patient and position in the 20-session range near its low",  # inrange: a trade has no fraction
    "Build a position in the 20-session range 0.5 near 18,600",  # inrange: not after an article
    "Close position in the 20-session range 0.5 near 20,100",  # inrange: not after "close"
    "Buying interest is planned near 18,600", "Selling activity is planned above 20,500",  # press: not interest
    "Then sell-off half at 20,500", "Then sell-off on strength",  # selloff: not after another word
    "Do a sell-off near 20,500",  # selloff: not before another word
    "The fund holds MBB",  # holds: only after support, level...
)

# Contract r3 E3: holdings, quantities, cost basis and money amounts, each pattern alone in its sentence ...
E3_PRIVATE = (
    "The 3,000-share lot is 12.8% under water", "My average price is 21,500", "acquisition price 21,500",
    "Down 12.8% from the 21,500 average paid", "The stock is owned since 21,500", "The family owns it",
    "owning it since 21,500", "on the books since 21,500", "Weight: 30% of NAV",
    "The account has 120 million VND available", "1.2 bn VND sits idle", "Còn 1,2 tỷ đồng chưa dùng",
    "Dry powder of 50m VND is set aside", "a balance of 120,000,000 VND", "VND 120m is available for MBB",
    "VND 120,000,000 is available", "Of the 3,000 MBB, 1,000 settle on Wednesday", "3k MBB remain",
    # review r3: each pattern and alternative alone in its sentence
    "The weighted average price is 21,500", "The average share price is 21,500", "Average fill price 21,500",
    "Average execution price 21,500", "Average cost price 21,500", "Average unit price 21,500",
    "MBB sits on their books at 21,500", "MBB is on its books at 21,500", "Weight: 30% of net asset value",
    "3K MBB remain", "Còn 120 triệu chưa dùng", "1.2 billion VND sits idle", "Còn 1,2 tỉ chưa dùng",
    "Còn 1,2 tỷ chưa dùng", "1.2 bn sits idle", "120 million sits idle", "3 nghìn cổ phiếu", "3 ngàn cp",
    "3.000 cổ phiếu", "3,000 shares", "3,000 share lot", "3k shares", "3m shares", "3 mn shares", "3 million shares",
    "Còn 120.000.000 đồng", "120.000.000₫ available", "120.000.000 VNĐ available", "Có 3 nghìn MBB", "Có 3 ngàn MBB",
    "MBB: 3,000 at an average of 21,500", "Underwater by 12.8% from 21,500", "Under-water by 12.8%",
    "The basis is 21,500", "A stake of 2% in MBB", "Half size near 18,600", "Full-size near 18,600",
    "A starter size near 18,600", "Three thousand shares at 21,500", "Hundreds of units at 21,500", "2,000 units",
    "2,000 lots", "$5,000 is set aside", "5,000 USD is set aside", "5,000 dollars is set aside", "120 mil set aside",
    "120 mio set aside", "120 mn set aside", "Còn 120 củ chưa dùng", "Còn 120tr chưa dùng", "50 m VND set aside",
    "50k VND set aside", "Còn 120.000.000đ chưa dùng", "The paying side near 20,100", "It pays 2% a year",
)
# ... while an indicator abbreviation after a number, a price in VND and a volume-weighted average price are published.
E3_TECHNICAL = (
    "Price crossed above the 20 EMA and the 50 SMA; on 2026-11-02 MBB closed at 18,750 VND on 1.4x volume.",
    "Close 18750 below the 14 RSI midline; 14 ATR 1.6%, 20 OBV falling; 100 POC 20133, VAL 18852, VAH 20560.",
    "Volume weighted average price 18,900 caps the close; the 6m and 12m excess returns are negative.",
    "The volume-weighted average price 18,900 caps the close; the close is 18.600đ on a daily basis.",
)

# Review r3 (finding 5): every rule of PRIVATE_PATTERNS and CASED_PATTERNS, every alternative and every character of a
# class of literal characters, in a sentence that only it withholds (review-fix/hits.py lists one pattern for each, and
# review-fix/mutants.py takes each out in turn and needs a red case). Several are not private by themselves: they pin a
# word of the denylist.
ONE_RULE = (
    # positions, holdings, cost basis
    "the exposure is large", "an allocation of 10%", "MBB is overweight", "MBB is underweight", "cash remains",
    "the budget", "the capital", "the cost", "margin loans", "a margin call", "a margin account", "margin debt",
    "margin trading", "the margin ratio", "a margin requirement", "held on margin", "avg price 21,500",
    "avg. price 21,500", "average price 21,500", "acquisition price 21,500", "break-even at 21,500",
    "breakeven at 21,500", "the p&l is red", "the p/l is red", "the pnl is red", "unrealised", "unrealized",
    "pay up", "paid up", "basis 21,500", "basis was 21,500", "basis of 21,500", "basis at 21,500", "basis near 21,500",
    "normal size", "initial size", "partial size", "double size", "quarter size", "planned size", "usual size",
    "max size", "maximum size", "trade size", "the lot size", "half sized", "full sizes", "stakes",
    "hundreds of shares", "millions of shares", "billions of shares", "thousand lots", "thousands of units",
    # numbers: a share count or an amount
    "3,000 units", "3,000 lots", "3,000 cp", "3 nghìn cổ", "3,000-share", "3 k shares", "120 million", "120 mil",
    "120 mio", "120 mn", "1.2 billion", "1.2 bn", "120 triệu", "1,2 tỷ", "1,2 tỉ", "120 củ", "120tr", "50k VND",
    "50m VND", "50k vnđ", "50k đồng", "50k ₫", "120,000,000 VND", "120.000.000 vnđ", "120.000.000đ",
    "120.000.000 ₫", "VND 120m", "vnđ 120k", "₫120m", "VND 120,000,000", "VNĐ 120.000.000", "₫ 120,000,000",
    "VND 120.000.000", "$5,000 remains", "5,000 usd remains", "5,000 dollars remain", "1 dollar remains",
    "T+2 on Wednesday", "sellable from Wednesday", "the settlement", "the broker", "the brokers", "the user",
    "the users", "if you like", "your view", "yours",
    # first person
    "we see", "our view", "ours", "my view", "mine", "I'm", "I\u2019m", "I've", "I'd", "I'll", "I am", "I have",
    "I had", "I own", "I would", "I will",
    # stops and trading actions
    "stoploss", "stop-losses", "profit was taken", "profits were taken", "profits are taken", "profit is taken",
    "profits being taken", "profits got taken", "profits taken", "profits booked", "profits locked",
    "take-partial-profits", "cut the loss", "cut some losses", "loss cut", "reentry", "reentries", "reenter",
    "reenters", "reentered", "reentering", "rebuy", "rebuys", "rebuying", "rebought", "resold", "resell", "resells",
    "reselling", "repurchase", "repurchases", "repurchased", "repurchasing", "new funds", "fresh fund",
    "fresh money", "place order", "places order", "placed order", "placing order", "put order", "puts order",
    "putting order", "set order", "sets order", "setting order", "an order", "the order", "order at 18,600",
    "order near 18,600", "offer above 20,500", "offer below 20,500", "offer around 20,500", "double down",
    "doubles down", "doubled down", "doubling down", "double-down", "go flat", "go-flat", "shortselling", "the longs",
    "the shorts", "long above 20,100", "short below 18,500", "long over 20,100", "long under 18,500",
    "long near 18,600", "long at 18,600", "long into 20,100", "long toward 20,100", "long towards 20,100",
    "short this", "short them", "short into strength", "short strength", "short it", "fades the rally",
    "faded any bounce", "fading every rip", "fade a pop", "fade this breakout", "fade rallies", "fade bounces",
    "fade strength", "fade the rips", "fade the pops", "fade breakouts", "await", "awaits", "awaited",
    "purchase", "purchases", "purchased", "purchasing", "hedge", "hedges", "hedged", "hedging", "liquidate",
    "liquidates", "liquidated", "liquidating", "acquire", "acquires", "acquiring", "invest", "invests", "invested",
    "investing", "dump", "dumps", "dumped", "dumping", "get long", "gets short", "getting long", "got in",
    "gotten out", "get-out", "lightens up", "lightened up", "lightening up", "lighten-up", "loads up", "loaded up",
    "loading up", "load-up", "bails out", "bailed out", "bailing out", "bail-out", "bottom fishing",
    "bottomfishing", "falling-knife", "lock in gains", "locks in gains", "locked in profits", "locking in gains",
    "lock-in gains", "book profits", "books gains", "booked losses", "booking profits", "off the table",
    "steps aside", "stepped aside", "stepping aside", "step-aside", "stays away", "stayed away", "staying away",
    "stay-away", "set aside", "sets aside", "setting aside", "set-aside", "fill at 21,500", "fills at 21,500",
    "filled near 21,500", "filled around 21,500", "picks up shares", "picked up shares", "picking up shares",
    "pick up some shares", "pick up more shares", "pick up the shares", "pick up a few shares", "pick up cheap shares",
    "pick up stock", "pick-up shares", "open the trade", "manage the trade", "closes the trade", "closed the trade",
    "closing the trade", "opens the trade", "close this trade", "close that trade", "close a trade", "close any trade",
    "close each trade", "close new trades", "keep the stock", "keeps the stock", "keeping the stock",
    "keep the shares", "keep the name", "a long near the low", "a short above the high", "new long below the low",
    "a long at the low", "long from the low", "short from the high", "long since the low", "existing long",
    "existing short", "existing investors", "existing owners", "already long", "already short", "already in the name",
    "already in the trade", "already in the stock", "stays long", "stayed short", "staying out", "stay aside",
    "stay-out", "stay flat", "target price 21,000", "price targets",
    # a trade set-up and its costs (review r3, the held-out measurement)
    "an r:r of 2", "r/r 1.5", "risk reward", "risk/reward", "risk-reward", "risk to reward", "risk-to-reward",
    "reward risk", "reward/risk", "reward-risk", "reward to risk", "reward-to-risk", "the fee", "a commission",
    "slippage", "the quantity", "quantities", "affordable", "affordability", "rejects orders", "mục tiêu", "nhắm",
    "giá vào", "chi phí", "phí giao dịch",
    # Vietnamese, and typed without diacritics
    "gom", "nên", "lãi", "lỗ", "lời", "lệnh", "vốn", "mua", "bán", "giữ", "chờ", "nhặt", "chốt", "cắt", "thoát",
    "cầm tiếp", "cầm hàng", "cầm cổ", "cầm chắc", "giải ngân", "xuống tiền", "ôm hàng", "tất tay", "lướt sóng",
    "hàng về", "tài khoản", "giá bình quân", "bình quân giá", "vào hàng", "gia tăng tại", "gia tăng thêm",
    "gia tăng quanh", "mức dừng", "điểm dừng", "the tỷ trọng", "the tỉ trọng", "danh mục", "vị thế", "tiền mặt",
    "ngân sách", "khuyên", "khuyến nghị", "hành động", "đứng ngoài", "ký quỹ", "kí quỹ", "tôi", "chúng ta", "mình",
    "bạn", "mục tiêu giá", "ban ra", "ban bot", "ban het", "cat lo", "chot loi", "chot lai", "chot mot phan",
    "chot bot", "chot het", "nen ban", "nen cho", "dung ngoai", "giai ngan", "bat day", "vao hang", "thoat hang",
    "so huu", "nhan roi", "so du", "trung binh gia", "cat giam", "muc dung", "diem dung", "dang am", "dang loi",
    "ty trong", "ti trong", "danh muc", "vi the", "tien mat", "ngan sach", "khuyen nghi", "hanh dong", "ky quy",
    "ki quy", "xuong tien", "om hang", "tat tay", "luot song", "hang ve", "tai khoan", "gia von", "gia binh quan",
    "giu", "lenh", "trieu",
    # URLs, host names, IP addresses, ports
    "ftp://x", "www. site", "localhost", "tailscale", "ngrok", "btc-ai", "btc-be", "btc-fe", "desktop-abc12",
    "laptop-abc12", "vietstock.vn", "dot com", "dot net", "dot org", "dot vn", "dot io", "dot dev", "dot app",
    "100.83.92.101", "fe80::1", "1:2:3:4:5:6:7:8", "host:8080", "at :5197", "the port", "the ports",
    # file paths
    "a \\ sign", "d:/notes", "(/abc/)", "./abc/", "../abc/", "~/abc/", "/home", "/tmp", "/var", "/etc",
    "/mnt", "/opt", "/srv", "/root", "/api", "evidence/", "scripts/", "skills/", "backend/", "frontend/",
    "contracts/", "_.json", "_.md", "_.py", "_.ps1", "_.psm1", "_.cs", "_.csproj", "_.sln", "_.ts", "_.tsx", "_.js",
    "_.mjs", "_.cjs", "_.txt", "_.log", "_.csv", "_.tsv", "_.xml", "_.htm", "_.html", "_.exe", "_.dll", "_.db",
    "_.sqlite", "_.sql", "_.env", "_.yml", "_.yaml", "_.ini", "_.cfg", "_.conf", "_.toml", "_.bat", "_.cmd", "_.sh",
    "_.joblib", "_.pkl", "_.parquet", "_.zip", "_.7z", "_.bak", "_.clixml", "_.pem", "_.crt",
    "research-sessions", "node_modules", "appdata", "the .ops folder",
    # e-mail addresses, hex strings and digit runs, internal ids and secrets
    "at @ sign", "0123456789abcdef", "f0e364fd", "1792315293201", "0xdeadbeef", "contract-definitions-sha-mismatch",
    "snap/x", "snap:x", "snap_x", "snap-x", "the snapshot", "session/x", "sessions/x", "note id", "note_id",
    "note-id", "noteid", "note ids", "forecast id", "request id", "x-admin", "admin key", "admin_key", "admin-key",
    "adminkey", "api key", "api_key", "api-key", "apikey", "api keys", "bearer", "password", "secret", "secrets",
    "token", "tokens", "credential", "credentials", "eqx_x",
    # where case decides
    "3,000 MBB", "3K MBB", "3k MBB", "3 nghìn MBB", "3 ngàn MBB", "a bid", "Bid near the low", "bids",
    "TP 21,000", "SL 18,500", "TP1 21,000", "keep MBB", "Keep MBB", "keeps MBB", "keeping MBB",
)
# Withheld where the sentence starts the thesis (the chart-phrase contexts read what stands before a phrase).
ALONE_PRIVATE = ("Sell-off on strength near 20,500", "Range position near 18,600, full size",
                 "The range position near 18,600 is the plan")

# Contract r3 E4: a character the keyword lists cannot read withholds the thesis, even an otherwise technical one.
E4_UNREADABLE = (
    "\u202eyub the dip",  # RIGHT-TO-LEFT OVERRIDE: the page shows "buy", the filter would read "yub"
    "\u202abuy\u202c", "\u202dsell\u202c", "\u2067yub\u2069", "\u2066RSI14 34\u2069",  # bidi
    "RSI14 34\u200b", "sell\u00adoff", "\ufeffMBB 1d close",  # other format characters (Cf)
    "\u0441ell near 20500", "b\u0443y the dip",  # Cyrillic look-alikes
    "\u0392uy the dip", "the \u03bfpen interest",  # Greek look-alikes
    "ba\u02can", "\u02b0old", "RSI14 34\u2093",  # modifier letters (Lm; U+2093 is named LATIN)
    "\U0001d41b\U0001d42e\U0001d432", "\uff42\uff55\uff59",  # mathematical and full-width letters
    "MBB\x07 1d close", "\ue000 private use",  # a control character, a private-use code point
)
E4_READABLE = ("MBB 1d close 18,750 \u2014 below SMA50 (\u22123.0%); RSI14 \u2265 30 \u2192 \u201cstretched\u201d, "
               "not accelerating\u2026 (giá đóng cửa 18.750, Wilder RSI14 34).")
# Review r3: the characters the lists can read are an allowlist (READABLE). Outside it, even where NFKC would fold the
# character into a readable one ...
E4_UNREADABLE_MORE = (
    "b\xb4uy the dip", "se\xa8ll near 20,500", "b\xafuy the dip", "b\xb8uy the dip",  # NFKC: a space and a mark
    "\xb5 of RSI14", "RSI14 34\xaa", "RSI14 34\xba",  # micro, ordinal signs
    "RSI14 ≤ 34 \u21d2 oversold",  # an arrow outside the short list
    "MBB 1d close\u2009near SMA50", "MBB 1d close\u202fnear SMA50",  # thin and narrow no-break spaces
    "bu\u030cy the dip",  # a combining caron outside Vietnamese: NFKC makes a u with caron (U+01D4)
    "s\u0301ell near 20,500", "the 3,000 M\u0301BB lot",  # a readable mark that composes into a letter
    # outside the allowlist (U+015B, U+1E3E): the page shows "sell" and "MBB"
    "\xbd of the range",  # NFKC writes the fraction with U+2044
    "se\u0308ll near 20,500",  # a mark outside the readable set that composes into a readable letter (U+00EB)
    "\u0130", "\u0131", "\u0138", "\u017f", "\u01c0", "\u0251", "\u0299", "\u1d07", "\u2500", "\u23af", "\u24d1",
    "\u2043", "\u02d7", "\u2796", "\u2e3a", "\ufe63",
)
# ... and every character of the allowlist outside ASCII stays readable in a technical sentence.
E4_READABLE_MORE = (
    "Close 18,750\xa0VND • RSI14 ≈ 34 ≠ 30 ≤ 35; ATR 1.6‰ ← ↑ ↓; "
    "‘band’ at 5′ and 10″; € and ₫; ‐ ‑ ‒ – — ― −.",
    "Latin-1: " + "".join(chr(c) for c in range(0xa1, 0x100)
                          if c not in (0xa8, 0xaa, 0xad, 0xaf, 0xb4, 0xb5, 0xb8, 0xba, 0xbc, 0xbd, 0xbe)),
    "Tiếng Việt: Ă ă Đ đ Ĩ ĩ Ũ ũ Ơ ơ Ư ư " + " ".join(map(chr, range(0x1ea0, 0x1efa))),
    "gia\u0301 đo\u0301ng cư\u0309a cao nhâ\u0301t, ă\u0300 ê\u0303 ơ\u0323 ă\u0306",
)
# Every dash reads as "-" (category Pd: U+2011 becomes U+2010 under NFKC, not "-"), and so does U+2212 MINUS SIGN.
DASHES = ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2015", "\u2212")

# Round r4 X1: the backend's enums (EquityResearchNoteService.cs, ActionViews and ActionBases) and the pairs whose
# thesis may be published. The daily task writes market-only with wait, or insufficient-input with none for a no-view
# without data (SKILL.md Step 3); a hold (the user said they hold the stock), reduce, sell or buy-more view, or a
# position-aware basis, can carry the user's holding. ("insufficient-input", "wait") is refused by the backend
# (insufficient-input requires none), so it is never served; the contract allows it as a pair of the two sets.
ACTION_VIEWS = ("buy-more", "hold", "reduce", "sell", "wait", "none")
ACTION_BASES = ("position-aware", "market-only", "insufficient-input")
PUBLIC_ACTIONS = {("market-only", "wait"), ("market-only", "none"), ("insufficient-input", "wait"),
                  ("insufficient-input", "none")}
# Fail-closed: a missing field, a value that is not a string or one outside the enums withholds the thesis, and the
# row is still exported (never a malformed notes part).
ODD_ACTIONS = {
    "actionView missing": lambda n: n.pop("actionView"),
    "actionBasis missing": lambda n: n.pop("actionBasis"),
    "both missing": lambda n: (n.pop("actionView"), n.pop("actionBasis")),
    "actionView null": lambda n: n.update(actionView=None),
    "actionBasis null": lambda n: n.update(actionBasis=None),
    "actionView a list": lambda n: n.update(actionView=["wait"]),
    "actionBasis an object": lambda n: n.update(actionBasis={"market-only": True}),
    "actionView a number": lambda n: n.update(actionView=0),
    "actionBasis true": lambda n: n.update(actionBasis=True),
    "actionView capitalised": lambda n: n.update(actionView="Wait"),
    "actionBasis with a trailing space": lambda n: n.update(actionBasis="market-only "),
    "actionBasis own-position (not a backend value)": lambda n: n.update(actionBasis="own-position"),
    "actionView watch (not a backend value)": lambda n: n.update(actionView="watch"),
}
# Round r4 X2: a number that ends with a comma is a list, not "<number> <ticker>": the bank-peer excess returns and a
# low named after a ticker are published (cold-check r3, lens privacy finding 4: skill_vs_filter.log, and the claim of
# the MBB v7 note of 2026-11-03) ...
X2_LISTS = (
    "+6.8, VCB", "18900, MBB's lowest",
    "Over 20 sessions the excess return is +1.8 points vs VNINDEX and vs BID +6.8, VCB +5.1, CTG -2.4 points.",
    "Against the bank benchmarks over 60 sessions: MBB -1.1, BID +3.0, VCB -0.6, CTG +0.9 points.",
    "The close was 18900, MBB's lowest close since July.",
    "BID fell 0.9% to 31700, BID's lowest close in the 100-bar window.",
    "20-session excess return vs VNINDEX +1.82 points; vs BID +6.8, VCB +5.1, CTG -2.4 points.",
    # review of r4 (x2_probe.py): lists the rule for counts written after the tickers must not take
    "Against the peers over 20 sessions: VCB +5.1, CTG -2.4, BID +6.8 points.",
    "The close was 31700, BID's lowest close since July.",
    "MBB at 18900, BID at 31700 and VCB at 61000 all sit below SMA50.",
    # round r4b: a signed value after a connector of _COUNT_AFTER_TICKER is a list ...
    "Against the bank peers over 20 sessions: BID: +6.8, VCB: +5.1, CTG: -2.4 points.",
    "vs BID +6.8, VCB (+5.1) and CTG (-2.4) points over 20 sessions.",
    # ... and a count written first withholds only at the end of a clause: not before a space, a comma or an
    # apostrophe (straight or curly), and never after a signed number ("+6.8, VCB" above)
    "After touching 18,550, MBB bounced to 18,900.", "In 2025, MBB, BID and VCB all rallied into the summer.",
    "The close was 18900, MBB’s lowest close since July.",
)
# ... while a number right before a ticker is still withheld, also across a full stop (SKILL.md v7 Step 4 tells the
# writer to avoid that, and test_equity_technical_skill.py pins it as withheld), and so is a list of counts written
# after the tickers (review of r4, x2_probe.py: r3 withheld the comma forms through the comma-ended count and r4 had
# opened them; r3 and r4 both published the K forms).
X2_COUNTS = ("closed 18750. MBB", "It closed 18750. MBB is weak", "3,000 MBB", "3.000 MBB", "3k MBB", "3K MBB",
             "3 nghìn MBB", "Book: MBB 3,000, BID 1,500", "Lines: MBB 3,000, BID 1,500, VCB 800",
             "As of Friday: MBB 3,000, BID 1,500", "In the account: MBB 3K, BID 1.5K",
             "In the account: MBB 3k, BID 1.5k",
             # round r4b (cold-check r4, privacy lens MINOR 2 and its triage): one sentence for each connector of
             # _COUNT_AFTER_TICKER, written so that only that connector withholds it (a K count, or a space before
             # the connector, keeps the end-of-clause rule out: review-fix/mutants.py takes each one out in turn) ...
             "As of Friday: MBB 3K, BID: 1.5K", "On the ledger, 2.6K, MBB; 1.2K, BID", "BID 1,200, MBB, 2,600",
             "Size by name: BID 1,200, MBB (2,600)", "BID 1,200, MBB [2,600]", "BID 1,200, MBB ~2,600",
             "MBB 3,000, BID ×1,500", "BID 1,200, MBB = 2,600", "BID 1,200, MBB ≈ 2,600",
             "MBB 3,000, BID x 1,500", "BID 1,200, MBB – 2,600", "BID 1,200, MBB about 2,600",
             "The book stood at BID 1,200, MBB roughly 2,600 at the close", "BID 1,200, MBB around 2,600",
             "MBB 3,000, BID some 1,500", "BID 1,200, MBB approximately 2,600", "BID 1,200, MBB approx. 2,600",
             "BID 1,200, MBB nearly 2,600", "BID 1,200, MBB almost 2,600",
             # (review of r4b, NIT 3: approx without its dot, and two dashes with a space on each side)
             "BID 1,200, MBB approx 2,600", "BID 1,200, MBB -- 2,600",
             # ... two connectors, and the forms of the triage ...
             "BID 1,200, MBB (~2,600)", "BID 1,200, MBB (about 2,600)", "BID 1,200, MBB: ~2,600",
             "BID 1,200, MBB: 2,600", "Book: MBB: 3,000, BID: 1,500", "MBB 3,000, BID: 1,500",
             "On the ledger, 2,600, MBB; 1,200, BID",
             # ... a comma glued to the ticker, and an unsigned count written first at the end of a clause
             "Lines of 2,600,MBB and 1,200,BID rolled over the weekend", "3,000, MBB", "1,200, MBB;", "1,200, MBB:",
             "(1,200, MBB)", "Is it 1,200, MBB?", "It was 1,200, MBB!", "The book has 3,000, MBB and 1,500, BID")
# Round r4 X3: the sentences of cold-check r3 (lens privacy finding 2 and its triage), withheld alone, in a carrier and
# inside a real thesis like REVIEW_LEAKS ...
COLDCHECK_R3_LEAKS = (
    "The October buyers are 12% under water.", "A 13.8% rise is needed to break even from 21,500.",
    "About 63,000,000 dong remains unused for this name.", "VND 120000000 is idle.",
    "Those positioned from the 21,450 cap are under water by 12%.",
    "The remaining 63000000 VND covers roughly 3,000 more at the ceiling.",
)
# ... and one sentence for every form the round added: each alternative and each separator of a changed pattern, in a
# sentence that only it withholds (review-fix/mutants.py takes each out in turn and needs a red case) ...
X3_PRIVATE = (
    "The October lot is 12.8% under water", "break even at 21,500", "breaks even at 21,500", "broke even at 21,500",
    "breaking even at 21,500", "a margin-call", "average-price 21,500", "average share-price 21,500",
    "net-asset-value", "off-the-table", "target-price 21,000", "price-target", "120,000,000 dong", "63000000 VND",
    "63000000 vnđ", "63000000đ", "63000000 ₫", "63000000 dong", "VND 120000000", "vnđ 120000000", "₫120000000",
    # review of r4 (review-W-exporter x3_probe.py, x4_probe.py): the money forms the docstring now names, one sentence
    # each: a dash before a scale word; after k or m "đ" as a word, "dong(s)" and a dash; after an amount of a million
    # or more a space between the thousands, decimals, a dash, "dongs", "Vietnamese"/"VN dong"; a colon and a space
    # separator after VND; USD before a number; k or m and a dash before USD or dollar(s)
    "a 63-million reserve", "About 63m dong remains idle", "A 900k dongs buffer is idle", "a 900k-dong buffer",
    "120m đ", "The idle balance is 63 000 000 VND", "The idle balance is 63,000,000.00 VND",
    "The idle balance is 63.000.000,00 đ", "A 63,000,000-dong reserve sits idle", "About 63,000,000 dongs remain idle",
    "Roughly 63,000,000 Vietnamese dong remain idle", "Roughly 63,000,000 VN dong remain idle",
    "Idle balance, VND: 63,000,000", "The idle balance stands at VND 63 000 000", "USD 50k", "2.5m dollars",
    "2.5k dollars", "a 2,500-dollar reserve",
    # ... and the cheap word forms of the review (NIT 4): margin call/calls/called/calling, margin balance, margin
    # financed/financing, net asset values, average prices, average per-share price, off of the table
    "Margin calls start below 18,550", "The October lot was margin-called at 18,600",
    "Margin calling starts below 18,550", "The margin balance on these shares is large",
    "Half of the October lot is margin-financed",
    "Margin financing is heavy for this name", "Net asset values across the account fell 12%",
    "Average prices for the two October lots were 21,500 and 21,800", "The average per-share price is 21,500",
    "The average per share price is 21,500", "Taking some money off of the table near 20,133",
    "Chips come off-of-the-table near 20,133",
    # round r4b B4: b (billion) before USD or dollar(s) (the VND forms are COLDCHECK_R4_LEAKS)
    "A 1.2B USD reserve is idle", "1.2b dollars sit idle",
    # review of r4b (MINOR 1): the optional space before k, m or b, in each of its three rules (the reviewer's K3, K4
    # and K10 took it out with the whole file green) ...
    "1.2 B VND", "63 m dong", "a 900 k-dong buffer", "VND 1.2 B", "VND 120 m", "₫ 63 m", "1.2 B USD", "2.5 m dollars",
    # ... and (MINOR 2) the country's name between k, m or b and the currency word, one form each, and Việt Nam after
    # an amount of a million or more
    "63m Vietnamese dong", "1.2B Vietnam dong", "900k Viet Nam đồng", "2B Việt Nam đồng", "63m VN dong",
    "63,000,000 Việt Nam đồng",
)
# ... while a VND price stays published: under a million with thousands separators, or at most 6 digits without them.
X3_TECHNICAL = (
    "MBB closed at 18,750 VND on 1.4x volume; FPT closed at 118500 VND and VCB at 61,000 dong.",
    "VND 18750 was the close; VND 118500 is the 20-session high of FPT; VIC closed at 242000 VND.",
    "The close is 18.600đ on a daily basis; 31700 VND was the 2026-11-03 close.",
    # review of r4: one thousands group with a space or after a colon, "dongs" and "VN dong" stay a price
    "VND 18 750 was the close; the close was 18 750 VND; VND: 18,750 was the 2026-11-03 close.",
    "61,000 dongs was the VCB close and 18,750 VN dong the MBB close.",
    # round r4b B4: a price with "Vietnam" or "Viet Nam dong" stays a price, and a b that starts a word or labels a
    # wave is not a billion
    "61,000 Vietnam dong was the VCB close and 18,750 Viet Nam dong the MBB close.",
    "VND 18750 by the end of the session was the close; wave 3b ended near 18,550 VND.",
    # review of r4b (MINOR 2): and so with Việt Nam
    "VCB closed at 61,000 Việt Nam đồng and MBB at 18,750 Việt Nam dong.",
)
# Round r4b B3 (cold-check r4, reviewer MINOR): the VND amount rules start at seven digits, a million; six digits are a
# price (a VN30 price has at most six). Relaxing either rule by one digit (\d{6,15} -> \d{7,15} after a number, VND
# \d{7,15} -> \d{8,15} before one: the reviewer's M9 and M10) publishes a seven-digit amount; tightening it withholds
# a six-digit price.
VND_EDGE = {"1200000 VND": True, "VND 1200000": True, "999999 VND": False, "VND 999999": False}
# Review of r4 (NIT 3): "break even" before lower, higher, further, more or deeper is chart prose (x3_probe.py).
BREAK_EVEN_PROSE = (
    "The close broke even lower on 2026-11-02, under the lower band.",
    "Price broke even further below SMA50 after the gap.", "The close broke even higher above the upper band.",
    "Price broke even more sharply below SMA50.", "Price broke even deeper into the gap.",
)
# Round r4b B1 (cold-check r4, privacy lens MINOR 1, triage: real): that exception is for the verb written with a
# space; "break-even" and "breakeven" (the noun, withheld unconditionally in r3) withhold before those words too.
BREAK_EVEN_NOUN = (
    "With the break-even higher than the 60-session high, a +5% move would still leave the October lot short of it.",
    "The break-even further up at 21,450 sits 11.6% above the close.",
    "A breakeven more than 11% above the close leaves the late-October lot exposed to the 18,550 low.",
    "The breakeven deeper in the old range, near 21,450, is out of reach in 20 sessions.",
    "The break-even lower down, near 20,100, applies to the second tranche.",
    "Break-even higher up the chart, at 21,450, is 11.6% above the close.",
)
# Round r4b: the sentences of cold-check r4 (privacy lens MINOR 1, 2 and 3, the triage's "Book: MBB: 3,000, BID:
# 1,500."), verbatim; each is withheld alone, in a carrier and inside a real thesis like REVIEW_LEAKS.
COLDCHECK_R4_LEAKS = BREAK_EVEN_NOUN + (
    "Carried into today's session: MBB: 2,600, BID: 1,200.", "BID 1,200, MBB: 2,600.",
    "The book stood at BID 1,200, MBB roughly 2,600 at the close.",
    "BID 1,200, MBB about 2,600 and VCB around 900 at the close.", "BID 1,200, MBB ~2,600 rolled into November.",
    "Size by name: BID 1,200, MBB (2,600).", "On the ledger, 2,600, MBB; 1,200, BID.",
    "Lines of 2,600,MBB and 1,200,BID rolled over the weekend.", "Book: MBB: 3,000, BID: 1,500.",
    "The 1.2B VND reserve set apart for this name is untouched.", "VND1.2B sits unused while the 18,550 low holds.",
    "A cushion of VND 1,2B is idle.", "₫1.2B is unused.",
    "About 63,000,000 Vietnam dong stays idle beside this chart.", "Roughly 63,000,000 Viet Nam dong is unused.",
)
# Review of round r4b (r4b/review-W-exporter: space_km_check.py, adversary.py): the reviewer's sentences that r4b
# published, verbatim; each is withheld alone, in a carrier and inside a real thesis like REVIEW_LEAKS.
REVIEW_R4B_LEAKS = (
    # MINOR 1: what the mutants K3, K4 and K10 published (a space before k, m or b)
    "A 1.2 B VND reserve is idle.", "63 m dong is idle.", "A 900 k-dong buffer is idle.", "VND 1.2 B is idle.",
    "VND 120 m is idle.", "₫ 63 m is idle.", "2.5 m dollars sit idle.", "1.2 B USD sits idle.",
    # MINOR 2 (b4-05, b4-06, b4-07, b4-15, b4-16): the country's name after k, m or b; Việt Nam after a million or more
    "A 2B Vietnam dong cushion is untouched.", "About 900k Vietnamese dong is untouched.",
    "63,000,000 Việt Nam đồng stays idle.", "2B Viet Nam dong is idle.", "A 63m Viet Nam dong reserve is untouched.",
    # NIT 3 (b2-13, b2-17, b2-18): approx without its dot, two dashes with a space on each side
    "Overnight carry: VCB 900, CTG approx 1,100.", "Overnight carry: VCB 900, CTG -- 1,100.",
    "Overnight carry: VCB 900, CTG (approx 1,100).",
)
# Round r4 X4: what the module docstring says about separators, phrase by phrase (r4/W-exporter/phrase_forms.py lists
# every multi-word phrase of the filter), as docstring text: an example. These match with a space and with a dash ...
SPACE_OR_DASH = {
    "stopped out": "stopped out", "scale in": "scale in", "stand aside": "stand aside", "take profit": "take profit",
    "new money": "new money", "falling knife": "falling knife", "risk reward": "risk reward",
    "break even": "break even at 21,500", "under water": "under water", "margin call": "margin call",
    "net asset value": "net asset value", "average price": "average price 21,500", "target price": "target price",
    "off the table": "off the table",
}
# ... and these only with spaces: their dash form is published, so the docstring may not claim it.
SPACE_ONLY = {
    "on margin": "held on margin", "at an average of": "at an average of 21,500", "on the books": "on the books",
    "basis of/is <number>": "basis of 21,500", "thousands of shares": "thousands of shares", "I am/have/own": "I am",
    "profits taken/booked": "profits taken", "cut the loss": "cut the loss", "an order": "an order",
    "place an order": "place an order", "order at <number>": "order at 18,600",
    "long above <number>": "long above 20,100", "a long at": "a long at 18,600", "long from": "long from 21,500",
    "already long": "already long", "already in the stock": "already in the stock",
    "short the bounce": "short the bounce", "fade the rally": "fade the rally", "lock in gains": "lock in gains",
    "book profits": "book profits", "fill at": "fill at 21,500", "pick up shares": "pick up shares",
    "close the trade": "close the trade", "keep the stock": "keep the stock",
    "existing investors/owners": "existing investors", "dot com": "dot com",
}
# The forms the docstring says are not listed (not every form of every listed word is).
FORMS_NOT_LISTED = ("stopping", "sidelining", "positioned", "going flat", "went flat")


def _sessions(n: int, start: date) -> list[str]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _ms(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1000)


def _target(day: str) -> str:
    return (date.fromisoformat(day) + timedelta(days=28)).isoformat()


def _thesis(symbol: str, i: int) -> str:
    return f"{symbol} thesis {i}: the 1d close sits above SMA50 and the 20-session support holds."


def _ret(i: int) -> float:
    return round(0.0005 * i - 0.02, 4)


def _bars(n: int, base: int, zero: tuple[int, ...] = ()) -> list[dict]:
    # open/high/low/close/volume all differ, so a swapped column shows
    return [{"openTimeMs": _ms(d), "timeIso": d + "T00:00:00Z", "open": base + 10 * i - 3, "high": base + 10 * i + 7,
             "low": base + 10 * i - 9, "close": base + 10 * i, "volume": 0 if i in zero else 1000 + i,
             "closeTimeMs": _ms(d) + 27_900_000, "quoteVolume": None} for i, d in enumerate(_sessions(n, date(2025, 8, 1)))]


def _technical(bars: list[dict], vnindex_status: str = "ok") -> dict:
    last = bars[-1]
    day_ind = {f: 100.0 + k + 0.5 for k, f in enumerate(INDICATOR_NAMES)}
    return {
        "contractVersion": "equity-technical-v1", "symbol": "X", "asOfUtc": "2026-11-02T08:00:00Z",
        "timeframes": [
            {"timeframe": "1h", "candles": [dict(last, openTimeMs=last["openTimeMs"] + 3_600_000, close=1)],
             "indicators": {"openTimeMs": last["openTimeMs"] + 3_600_000, **{f: -1.0 for f in INDICATOR_NAMES}}},
            {"timeframe": "1d", "candles": [{k: last[k] for k in ("openTimeMs", "open", "high", "low", "close", "volume")}],
             "indicators": {"openTimeMs": last["openTimeMs"], "isLatestCandle": True, **day_ind, "macdNorm": 0.25,
                            "obv": 123456, "vwap": 1.5, "methods": {"rsi14": "Wilder RSI(14)"}}},
        ],
        "relativeStrength": {"status": vnindex_status, "reason": None, "windows": [20, 60], "benchmarks": [
            {"symbol": "BID", "kind": "peer", "status": "ok", "returns": {
                "20": {"instrument": 0.011, "benchmark": 0.5, "excess": 0.9},
                "60": {"instrument": -0.07, "benchmark": 0.4, "excess": 0.8}}},
            {"symbol": "VNINDEX", "kind": "index", "status": vnindex_status, "returns": {
                "20": {"instrument": 0.011, "benchmark": -0.02, "excess": 0.031},
                "60": {"instrument": -0.07, "benchmark": -0.03, "excess": -0.04}}}]},
        "noteBinding": {"snapshotId": "snap-secret-id", "snapshotSha256": "b" * 64, "requestedAsOf": "1",
                        "signalBarOpenTimeMs": last["openTimeMs"]},
        "limitations": ["text"],
    }


def _entry(symbol: str, day: str, n: int, *, mode: str = "forward", direction: str = "up", available: int = 5_000,
           thesis: str = "", horizon: int = 20, status: str = "pending", pending: str | None = None,
           ret: float | None = None, hit: bool | None = None, target: str | None = None, realized: str | None = None,
           baseline: str | None = None, view: str = "wait", basis: str = "market-only") -> tuple[dict, dict]:
    """One journal note as GET research-notes serves it, and its GET direction-scores item. actionView/actionBasis
    default to what the daily task writes (equity-technical SKILL.md Step 3: market-only with wait)."""
    ms = _ms(day)
    note = {"noteId": f"session/{symbol}/{ms}/{n:08x}", "symbol": symbol, "mode": mode,
            "requestedAsOfUtc": day + "T08:00:00Z", "signalBarOpenTimeMs": ms, "signalBarCloseTimeMs": ms + 27_900_000,
            "horizonSessions": horizon, "snapshotId": f"snap/{symbol}/{ms}/snap-secret-id", "snapshotSha256": "a" * 64,
            "snapshotVerified": True, "skillVersion": "equity-technical-v6", "methodVersion": "m1",
            "direction": direction, "actionView": view, "actionBasis": basis,
            "body": {"schemaVersion": "equity-research-note-v1", "thesis": thesis,
                     "evidence": [{"claim": "c", "evidenceIds": ["e1"]}],
                     "counterEvidence": [{"claim": "x", "evidenceIds": ["e2"]}],
                     "scenarios": [{"name": "s", "horizonSessions": horizon, "description": "d", "invalidation": ["i"]}],
                     "limitations": ["l"], "missingData": ["m"],
                     "outcomeCheck": {"how": "h", "dueAfterSessions": horizon}, "probability": 0.7},
            "availableTimeMs": ms + available, "createdAtUtc": day + "T10:00:00Z", "forecastIds": ["fc-1"]}
    item = {"noteId": note["noteId"], "mode": mode, "signalBarOpenTimeMs": ms, "horizonSessions": horizon,
            "direction": direction, "availableTimeMs": note["availableTimeMs"], "status": status,
            "pendingReason": pending, "signalClose": 18750,
            "targetBarOpenTimeMs": None if target is None else _ms(target),
            "targetClose": None if target is None else 19000, "returnPct": ret, "realized": realized,
            "baselineCall": baseline, "hit": hit}
    return note, item


def _mbb_days() -> list[str]:
    return list(reversed(_sessions(100, date(2026, 6, 8))))  # [0] = the newest signal bar


def _mbb_journal() -> list[tuple[dict, dict]]:
    """One forward H=20 note on each of 100 signal bars (days[0] newest) plus a superseded one, 2 H=60 and 2
    retrospective notes. days[0..19]: not resolved yet, with every status and pendingReason of the contract and a
    thesis with private text (days[2]), a long one (days[11]) and one of exactly 1500 characters (days[12]); days[3]
    also has a later note on the taken bar (superseded). days[20..99]: 80 resolved notes, hit unless i % 3 == 0,
    the baseline right when i % 4 == 0."""
    days, pairs = _mbb_days(), []
    for i, d in enumerate(days):
        kw = {"direction": DIRS[i % 3], "thesis": _thesis("MBB", i), "status": "pending", "pending": "immature",
              "target": _target(d)}
        if i >= 20:
            hit = i % 3 != 0
            realized = DIRS[i % 3] if hit else DIRS[(i + 1) % 3]
            kw.update(status="resolved", pending=None, ret=_ret(i), hit=hit, realized=realized,
                      baseline=realized if i % 4 == 0 else DIRS[(DIRS.index(realized) + 1) % 3])
        elif i == 2:
            kw.update(thesis=PLANTED_THESIS)
        elif i == 3:
            kw.update(thesis="first note on the bar", available=1_000)
            pairs.append(_entry("MBB", d, 300, direction="down", thesis="late second note", available=9_000,
                                status="superseded-not-scored"))
        elif i == 5:
            kw.update(direction="no-view", status="no-view")
        elif i == 6:
            kw.update(pending="calendar-coverage-expired", target=None)
        elif i == 7:
            kw.update(pending="target-bar-missing")
        elif i == 8:
            kw.update(pending="signal-bar-missing")
        elif i in (9, 10):
            kw.update(status=("out-of-window-not-scored", "window-undetermined-not-scored")[i - 9], pending=None,
                      target=None)
        elif i == 11:
            kw.update(thesis=LONG_THESIS)
        elif i == 12:
            kw.update(thesis=EXACT_THESIS)
        pairs.append(_entry("MBB", d, i, **kw))
    pairs.append(_entry("MBB", days[0], 600, direction="down", thesis="h60 newest", available=9_000, horizon=60,
                        pending="immature", target=_target(days[0])))
    pairs.append(_entry("MBB", days[30], 630, direction="up", thesis="h60 resolved", horizon=60, status="resolved",
                        ret=0.12, hit=True, target=_target(days[30]), realized="up", baseline="up"))
    for i in (1, 4):
        pairs.append(_entry("MBB", days[i], 400 + i, mode="retrospective", thesis=f"retro {i}", available=7_000,
                            status="retrospective-not-scored"))
    return pairs


def _vcb_journal() -> list[tuple[dict, dict]]:
    """31 resolved notes 21 sessions apart (nEffective 31: "ok") and a newer pending one."""
    days = list(reversed(_sessions(31 * 21 + 1, date(2024, 1, 1))))
    pairs = [_entry("VCB", days[0], 0, thesis=_thesis("VCB", 0), pending="immature", target=_target(days[0]))]
    for k in range(1, 32):
        d, hit = days[21 * k], k % 5 != 0
        realized = DIRS[k % 3] if hit else DIRS[(k + 1) % 3]
        pairs.append(_entry("VCB", d, k, direction=DIRS[k % 3], thesis=_thesis("VCB", k), status="resolved",
                            ret=_ret(k), hit=hit, target=_target(d), realized=realized, baseline="sideways"))
    return pairs


def _vic_journal() -> list[tuple[dict, dict]]:
    days = list(reversed(_sessions(2, date(2026, 10, 29))))
    return [_entry("VIC", d, i, direction="down", thesis=_thesis("VIC", i), pending="immature", target=_target(d))
            for i, d in enumerate(days)]


def _acb_journal() -> list[tuple[dict, dict]]:
    d = "2026-09-01"
    return [_entry("ACB", d, 0, thesis=_thesis("ACB", 0), status="resolved", ret=0.06, hit=True, target=_target(d),
                   realized="up", baseline="sideways")]


def _crowded_journal() -> tuple[list[tuple[dict, dict]], list[str]]:
    """199 retrospective notes on the 199 newest bars; days[199] holds a retrospective note and, written later, the
    only forward note among the backend's newest 200 notes; days[200..279]: 80 older resolved forward notes."""
    days = list(reversed(_sessions(280, date(2025, 9, 1))))
    pairs = [_entry("MBB", days[i], i, mode="retrospective", thesis=f"retro {i}", available=1_000,
                    status="retrospective-not-scored") for i in range(200)]
    pairs.append(_entry("MBB", days[199], 199_000, direction="down", thesis="the only forward note in the take",
                        pending="target-bar-missing", target=_target(days[199])))
    for i in range(200, 280):
        hit = i % 2 == 0
        realized = DIRS[i % 3] if hit else DIRS[(i + 2) % 3]
        pairs.append(_entry("MBB", days[i], i, direction=DIRS[i % 3], thesis=_thesis("MBB", i), status="resolved",
                            ret=_ret(i), hit=hit, target=_target(days[i]), realized=realized,
                            baseline=realized if i % 5 == 0 else DIRS[(DIRS.index(realized) + 1) % 3]))
    return pairs, days


def _busday(ms: int) -> int:
    weeks, rest = divmod((datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date() - date(2024, 1, 1)).days, 7)
    return weeks * 5 + min(rest, 5)  # 2024-01-01 is a Monday


def _summary(h: int, items: list[dict]) -> dict:
    """One direction-scores horizons[] entry as EquityDirectionScoringService.Summarize builds it: over every item."""
    fwd = [i for i in items if i["mode"] == "forward" and i["horizonSessions"] == h]

    def count(status: str) -> int:
        return sum(i["status"] == status for i in fwd)

    def rate(k: int) -> float | None:
        return k / len(resolved) if resolved else None

    resolved = [i for i in fwd if i["status"] == "resolved"]
    pending = [i for i in fwd if i["status"] == "pending"]
    n_eff, last = 0, None
    for b in sorted(_busday(i["signalBarOpenTimeMs"]) for i in resolved):
        if last is None or b - last >= h:
            n_eff, last = n_eff + 1, b
    hits = sum(i["hit"] is True for i in resolved)
    prior = sum(i["baselineCall"] == i["realized"] for i in resolved)
    scored = count("no-view") + len(resolved) + len(pending)
    return {"horizonSessions": h, "configId": f"tb-h{h}", "upThresholdPct": 0.05 if h == 20 else 0.1,
            "downThresholdPct": -0.04 if h == 20 else -0.06,
            "status": "no-data" if not resolved else "insufficient-sample" if n_eff < 30 else "ok",
            "forwardNotes": len(fwd), "outOfWindow": count("out-of-window-not-scored"),
            "windowUndetermined": count("window-undetermined-not-scored"),
            "eligible": scored + count("superseded-not-scored"), "superseded": count("superseded-not-scored"),
            "scored": scored, "noView": count("no-view"), "directional": len(resolved) + len(pending),
            "resolved": len(resolved), "pending": len(pending),
            "pendingReasons": {k: sum(i["pendingReason"] == v for i in pending) for k, v in (
                ("immature", "immature"), ("targetBarMissing", "target-bar-missing"),
                ("calendarCoverageExpired", "calendar-coverage-expired"), ("signalBarMissing", "signal-bar-missing"))},
            "hits": hits, "hitRate": rate(hits), "coverage": (len(resolved) + len(pending)) / scored if scored else None,
            "nEffective": n_eff, "minEffectiveForScoring": 30, "baselinePrior": {"hits": prior, "hitRate": rate(prior)},
            "bestConstantInHindsight": {"label": None, "hits": 0, "hitRate": None}}


def _http(code: int, api_code: str | None = None) -> urllib.error.HTTPError:
    body = json.dumps({"code": api_code, "message": "m", "retryable": False, "requestId": "r1"}).encode()
    return urllib.error.HTTPError("http://fake.invalid/x", code, "error", {}, io.BytesIO(body))


def _query(path: str) -> dict[str, str]:
    q = path.split("?", 1)[1] if "?" in path else ""
    return dict(kv.split("=", 1) for kv in q.split("&") if kv)


class Clock:
    def __init__(self):
        self.t, self.sleeps = 1000.0, []

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


class FakeBackend:
    """GET-only fake of the endpoints the exporter reads, mimicking the backend. candles honour `limit` (the latest N
    bars). research-notes: take defaults to 50, clamped to 1..200, every mode and horizon counts, newest signal bar
    first, then availableTimeMs descending, then noteId. direction-scores: items take defaults to 200, clamped to
    1..500, newest signal bar first, then horizon, availableTimeMs and noteId; horizons[] (one entry per barrier
    config, ascending) summarise the WHOLE journal whatever take is. AdminGuard (401) runs before the capability
    check (400)."""

    def __init__(self, bars, technical, journals, unsupported=frozenset(), errors=None, clock=None):
        self.bars, self.technical, self.journals = bars, technical, journals
        self.unsupported, self.errors, self.clock, self.calls = set(unsupported), errors or {}, clock, []
        # per symbol: barrier configs (the backend has H=20 and H=60), a horizons[] rewrite, items left out
        self.configs, self.patch_horizons, self.hidden_items = {}, {}, set()

    def __call__(self, path: str, headers: dict):
        self.calls.append((path, dict(headers or {}), self.clock() if self.clock else None))  # record even bad calls
        if path == "/api/equities":
            return {"contractVersion": "equity-instruments-v1",
                    "items": [{"symbol": s, "label": f"{s} (Name {s})", "venue": "HOSE"} for s in self.bars]}
        parts = path.split("?")[0].split("/")
        symbol, kind = parts[3], "/".join(parts[4:])
        queue = self.errors.get((symbol, kind))
        if queue:
            error = queue.pop(0)  # a status, or (status, envelope code)
            raise _http(*error) if isinstance(error, tuple) else _http(error)
        if symbol not in self.bars:
            raise _http(404, "UNKNOWN_SYMBOL")
        if kind == "candles":
            return self.bars[symbol][-int(_query(path)["limit"]):]
        if kind.startswith("research-notes") and headers.get("X-Admin-Key") != KEY:
            raise _http(401)
        if symbol in self.unsupported:
            raise _http(400, "UNSUPPORTED_CAPABILITY")
        if kind == "technical":
            return self.technical[symbol]
        journal = self.journals.get(symbol, [])
        if kind == "research-notes":
            take = max(1, min(int(_query(path).get("take", 50)), 200))
            notes = sorted((n for n, _ in journal),
                           key=lambda n: (-n["signalBarOpenTimeMs"], -n["availableTimeMs"], n["noteId"]))
            return {"contractVersion": "equity-research-notes-v1", "symbol": symbol, "items": notes[:take]}
        if kind == "research-notes/direction-scores":
            take = max(1, min(int(_query(path).get("take", 200)), 500))
            every = [i for _, i in journal]
            items = sorted((i for i in every if i["noteId"] not in self.hidden_items),
                           key=lambda i: (-i["signalBarOpenTimeMs"], i["horizonSessions"], i["availableTimeMs"],
                                          i["noteId"]))
            horizons = [_summary(h, every) for h in sorted(self.configs.get(symbol, (20, 60)))]
            patch = self.patch_horizons.get(symbol)
            return {"contractVersion": "equity-direction-scores-v1", "symbol": symbol,
                    "ruleVersion": "direction-score-v1", "rule": "r", "generatedAtUtc": "2026-11-02T12:00:00Z",
                    "calendarCoverageTo": "2026-12-31", "totalNotes": len(every),
                    "retrospectiveNotes": sum(i["mode"] != "forward" for i in every), "noDeclaredThresholdNotes": 0,
                    "horizons": patch(horizons) if patch else horizons,
                    "items": items[:take]}
        raise AssertionError(path)


def _market(clock=None, errors=None, unsupported=frozenset({"ACB"})) -> FakeBackend:
    bars = {s: _bars(300, 20000 + 5000 * k, zero=(10, 250, 299)) for k, s in enumerate(SYMBOLS)}
    technical = {s: _technical(bars[s], vnindex_status="stale" if s == "VIC" else "ok") for s in SYMBOLS}
    journals = {"MBB": _mbb_journal(), "VCB": _vcb_journal(), "VIC": _vic_journal(), "ACB": _acb_journal()}
    return FakeBackend(bars, technical, journals, unsupported=unsupported, errors=errors, clock=clock)


def _screen(session: str = SESSION, schema: str = "vn30-screen/v1", symbols=SYMBOLS) -> dict:
    rows = [{"symbol": s, "close": 1000.0 * n, "returns": {"3": 0.01 * n, "6": 0.02 * n, "12": 0.03 * n},
             "excessVsIndex": {"3": -0.01 * n, "6": -0.02 * n, "12": -0.03 * n},
             "ranks": {"3": 10 + n, "6": n, "12": 20 + n}, "topByRankMonths": n <= 2} for n, s in enumerate(symbols, 1)]
    rows[-1]["ranks"]["6"] = None
    return {"schema": schema, "generatedAtUtc": "2026-11-02T09:20:00+00:00", "session": session,
            "asOfRequested": None, "caveat": "Quan sát", "basket": {"effectiveDate": "2026-08-03", "possiblyStale": False},
            "anchorSessions": {"3": "2026-08-01"}, "indexReturns": {"3": -0.014, "6": -0.088, "12": 0.035},
            "ranking": {"lookbackMonths": 6, "topK": 10}, "missingSessionBar": ["VIC"], "unknownToBackend": [],
            "notUpToDateNow": ["ACB"], "rows": rows}


@pytest.fixture
def screen_dir(tmp_path: Path) -> Path:
    d = tmp_path / "screen"
    d.mkdir()
    (d / f"{SESSION}.json").write_text(json.dumps(_screen()), encoding="utf-8")
    (d / "2026-10-30.json").write_text(json.dumps(_screen("2026-10-30", symbols=("BID", "FPT"))), encoding="utf-8")
    (d / "zz-notes.json").write_text("not a screen", encoding="utf-8")  # not <session>.json: ignored
    return d


def _run(fake, clock, screen_dir: Path, out: Path, monkeypatch, key: str | None = KEY) -> int:
    if key is None:
        monkeypatch.delenv("EQX_ADMIN_KEY", raising=False)
    else:
        monkeypatch.setenv("EQX_ADMIN_KEY", key)
    return web.main(["--out-dir", str(out), "--screen-dir", str(screen_dir)], fetch=fake, sleep=clock.sleep,
                    clock=clock)


def _summary_line(capsys) -> dict:
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line.isascii()
    return json.loads(line)


def _load(out: Path) -> dict[str, dict]:
    return {p.relative_to(out).as_posix(): json.loads(p.read_text(encoding="utf-8")) for p in out.rglob("*.json")}


def _meta(out: Path) -> dict:
    return json.loads((out / "meta.json").read_text(encoding="utf-8"))


def _symbol(out: Path, s: str) -> dict:
    return json.loads((out / "symbols" / f"{s}.json").read_text(encoding="utf-8"))


def _keys(x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield k
            yield from _keys(v)
    elif isinstance(x, list):
        for v in x:
            yield from _keys(v)


def _strings(x):
    """Every decoded key and string value (a Windows path is escaped in the raw JSON text)."""
    if isinstance(x, dict):
        for k, v in x.items():
            yield k
            yield from _strings(v)
    elif isinstance(x, list):
        for v in x:
            yield from _strings(v)
    elif isinstance(x, str):
        yield x


def _newest(fake: FakeBackend, symbol: str = "MBB") -> tuple[dict, dict]:
    """The newest forward H=20 note that holds its bar's slot, and its direction-score item."""
    return max((p for p in fake.journals[symbol] if p[0]["mode"] == "forward" and p[0]["horizonSessions"] == 20
                and p[1]["status"] != "superseded-not-scored"),
               key=lambda p: (p[0]["signalBarOpenTimeMs"], p[0]["availableTimeMs"]))


def _resolved(fake: FakeBackend) -> tuple[dict, dict]:
    """MBB's newest resolved H=20 note (days[20], inside the exported 60)."""
    day = _ms(_mbb_days()[20])
    return next(p for p in fake.journals["MBB"] if p[0]["signalBarOpenTimeMs"] == day and p[0]["horizonSessions"] == 20)


def _patch_h20(**changes):
    def patch(horizons):
        return [dict(h, **{k: (v(h) if callable(v) else v) for k, v in changes.items()})
                if h["horizonSessions"] == 20 else h for h in horizons]
    return patch


CORRUPTIONS = {
    "direction-buy": lambda f: [d.update(direction="buy") for d in _newest(f)],
    "status-reduce": lambda f: _newest(f)[1].update(status="reduce"),
    "status-retrospective-on-a-forward-note": lambda f: _newest(f)[1].update(status="retrospective-not-scored"),
    "status-no-declared-thresholds": lambda f: _newest(f)[1].update(status="no-declared-thresholds"),
    "status-capitalised": lambda f: _resolved(f)[1].update(status="Resolved"),
    "pending-reason-unknown": lambda f: _newest(f)[1].update(pendingReason="soon"),
    "hit-not-boolean": lambda f: _resolved(f)[1].update(hit="yes"),
    "score-of-another-note": lambda f: _newest(f)[1].update(direction="down"),
    "note-without-score": lambda f: f.hidden_items.add(_newest(f)[0]["noteId"]),
    "sample-status-unknown": lambda f: f.patch_horizons.update(MBB=_patch_h20(status="great")),
    "no-h20-entry": lambda f: f.configs.update(MBB=(60,)),
    "two-h20-entries": lambda f: f.patch_horizons.update(
        MBB=lambda hs: hs + [h for h in hs if h["horizonSessions"] == 20]),
    "hits-above-resolved": lambda f: f.patch_horizons.update(MBB=_patch_h20(hits=lambda h: h["resolved"] + 1)),
    "baseline-above-resolved": lambda f: f.patch_horizons.update(MBB=_patch_h20(
        baselinePrior=lambda h: {"hits": h["resolved"] + 1, "hitRate": None})),
    "resolved-not-a-count": lambda f: f.patch_horizons.update(MBB=_patch_h20(resolved=lambda h: float(h["resolved"]))),
}


def test_every_output_file_holds_only_contract_fields_and_no_planted_secret(screen_dir, tmp_path, monkeypatch):
    out = tmp_path / "data"
    assert _run(_market(Clock()), Clock(), screen_dir, out, monkeypatch) == 0
    files = _load(out)
    assert sorted(files) == ["meta.json", "screen.json"] + [f"symbols/{s}.json" for s in sorted(SYMBOLS)]
    for name, doc in files.items():
        leaked = FORBIDDEN & set(_keys(doc))
        assert not leaked, f"{name} leaks {leaked}"
        text, values = (out / name).read_text(encoding="utf-8"), list(_strings(doc))
        for secret in (KEY, "snap-secret-id", "session/MBB/", "fake.invalid", "http", "market-only", "a" * 64,
                       *THESIS_SECRETS):
            assert secret not in text and not any(secret in v for v in values), f"{name} contains {secret!r}"
    meta, screen = files["meta.json"], files["screen.json"]
    assert list(meta) == META_KEYS and meta["schema"] == "vn30-web-data/v2"
    assert all(list(u) == ["symbol", "part", "reason"] for u in meta["unavailable"])
    assert list(screen) == SCREEN_KEYS and all(list(r) == ROW_KEYS for r in screen["rows"])
    for s in SYMBOLS:
        doc = files[f"symbols/{s}.json"]
        assert list(doc) == SYMBOL_KEYS
        assert doc["score"] is None or list(doc["score"]) == SCORE_KEYS
        assert all(list(n) == NOTE_KEYS for n in doc["notes"])
        assert doc["indicators"] is None or list(doc["indicators"]) == INDICATOR_KEYS
        assert doc["relativeStrength"] is None or list(doc["relativeStrength"]) == ["20", "60"]
    assert len(files["symbols/MBB.json"]["notes"]) == 60  # the fake really served notes with forbidden fields
    assert list(files["symbols/MBB.json"]["score"]) == SCORE_KEYS and files["symbols/ACB.json"]["score"] is None


def test_candles_are_the_latest_260_bars_without_zero_volume(screen_dir, tmp_path, monkeypatch):
    fake = _market(Clock())
    out = tmp_path / "data"
    assert _run(fake, Clock(), screen_dir, out, monkeypatch) == 0
    bars = fake.bars["MBB"]
    expected = [[web.session_of(b["openTimeMs"]), b["open"], b["high"], b["low"], b["close"], b["volume"]]
                for b in bars[-260:] if b["volume"] > 0]
    got = _symbol(out, "MBB")["candles"]
    assert got == expected and len(got) == 258  # bars 250 and 299 had zero volume; bar 10 was outside the window
    assert got[-1][0] == web.session_of(bars[-2]["openTimeMs"])
    assert all(p.endswith("candles?timeframe=1d&limit=260") for p, _, _ in fake.calls if "/candles" in p)
    assert all(u["part"] != "candles" for u in _meta(out)["unavailable"])


def test_indicators_are_the_1d_row_and_strength_is_the_vnindex_excess(screen_dir, tmp_path, monkeypatch):
    fake = _market(Clock())
    out = tmp_path / "data"
    assert _run(fake, Clock(), screen_dir, out, monkeypatch) == 0
    mbb, last = _symbol(out, "MBB"), fake.bars["MBB"][-1]
    assert mbb["indicators"] == {"date": web.session_of(last["openTimeMs"]), "close": last["close"],
                                 **{f: 100.0 + k + 0.5 for k, f in enumerate(INDICATOR_NAMES)}}
    assert mbb["relativeStrength"] == {"20": 0.031, "60": -0.04}
    assert mbb["label"] == "MBB (Name MBB)" and mbb["isBank"] is True and _symbol(out, "VIC")["isBank"] is False
    vic = _symbol(out, "VIC")  # VNINDEX stale: its old window is not the latest session
    assert vic["relativeStrength"] is None and vic["indicators"] is not None
    assert [u for u in _meta(out)["unavailable"] if u["symbol"] == "VIC"] == [
        {"symbol": "VIC", "part": "technical", "reason": NO_STRENGTH}]


def test_a_truthful_empty_answer_is_never_replaced_by_the_previous_run(screen_dir, tmp_path, monkeypatch):
    out = tmp_path / "data"
    fake = _market(Clock())
    fake.technical["VIC"] = _technical(fake.bars["VIC"])  # VNINDEX ok on the first run
    assert _run(fake, Clock(), screen_dir, out, monkeypatch) == 0
    assert _symbol(out, "VIC")["relativeStrength"] is not None and _symbol(out, "MBB")["indicators"] is not None
    later = _market(Clock())  # VIC: VNINDEX stale; MBB: no 1d indicator row yet
    later.technical["MBB"]["timeframes"][1]["indicators"] = None
    assert _run(later, Clock(), screen_dir, out, monkeypatch) == 0
    vic, mbb = _symbol(out, "VIC"), _symbol(out, "MBB")
    assert vic["relativeStrength"] is None and vic["indicators"] is not None
    assert mbb["indicators"] is None and mbb["relativeStrength"] == {"20": 0.031, "60": -0.04}
    unavailable = _meta(out)["unavailable"]
    assert {"symbol": "VIC", "part": "technical", "reason": NO_STRENGTH} in unavailable
    assert {"symbol": "MBB", "part": "technical", "reason": NO_INDICATORS} in unavailable


def test_notes_are_the_newest_forward_20_session_notes_joined_by_note_id(screen_dir, tmp_path, monkeypatch):
    fake = _market(Clock())
    out = tmp_path / "data"
    assert _run(fake, Clock(), screen_dir, out, monkeypatch) == 0
    notes, days = _symbol(out, "MBB")["notes"], _mbb_days()
    # the newest 60 bars; the superseded second note of days[3], the H=60 notes (one written last on days[0]) and the
    # retrospective notes are not exported
    assert [n["signalDate"] for n in notes] == days[:60]
    assert {n["horizonSessions"] for n in notes} == {20}
    assert not {"late second note", "h60 newest", "h60 resolved", "retro 1", "retro 4"} & {n["thesis"] for n in notes}
    assert notes[0] == {"signalDate": days[0], "horizonSessions": 20, "direction": "up", "thesis": _thesis("MBB", 0),
                        "status": "pending", "pendingReason": "immature", "targetDate": _target(days[0]),
                        "returnPct": None, "hit": None}
    assert (notes[2]["thesis"], notes[2]["direction"], notes[2]["status"]) == (None, "sideways", "pending")
    assert (notes[3]["thesis"], notes[3]["direction"], notes[3]["status"]) == ("first note on the bar", "up", "pending")
    assert notes[5] == {"signalDate": days[5], "horizonSessions": 20, "direction": "no-view",
                        "thesis": _thesis("MBB", 5), "status": "no-view", "pendingReason": "immature",
                        "targetDate": _target(days[5]), "returnPct": None, "hit": None}
    assert (notes[6]["pendingReason"], notes[6]["targetDate"]) == ("calendar-coverage-expired", None)
    assert [n["pendingReason"] for n in notes[7:9]] == ["target-bar-missing", "signal-bar-missing"]
    assert notes[9] == {"signalDate": days[9], "horizonSessions": 20, "direction": "up", "thesis": _thesis("MBB", 9),
                        "status": "out-of-window-not-scored", "pendingReason": None, "targetDate": None,
                        "returnPct": None, "hit": None}
    assert (notes[10]["status"], notes[10]["direction"]) == ("window-undetermined-not-scored", "down")
    assert notes[11]["thesis"] == LONG_THESIS[:1499] + "\u2026" and len(notes[11]["thesis"]) == 1500
    assert notes[12]["thesis"] == EXACT_THESIS
    assert notes[20] == {"signalDate": days[20], "horizonSessions": 20, "direction": "sideways",
                         "thesis": _thesis("MBB", 20), "status": "scored", "pendingReason": None,
                         "targetDate": _target(days[20]), "returnPct": _ret(20), "hit": True}
    assert (notes[21]["direction"], notes[21]["status"], notes[21]["hit"], notes[21]["returnPct"]) == \
           ("up", "scored", False, _ret(21))
    assert [p for p, _, _ in fake.calls if "/MBB/research-notes" in p] == [
        "/api/equities/MBB/research-notes?take=200", "/api/equities/MBB/research-notes/direction-scores?take=500"]


def test_score_is_the_whole_journal_h20_total_from_direction_scores(screen_dir, tmp_path, monkeypatch):
    out = tmp_path / "data"
    assert _run(_market(Clock()), Clock(), screen_dir, out, monkeypatch) == 0
    mbb = _symbol(out, "MBB")
    # 80 resolved H=20 notes in the journal (53 hits, prior right on 20), only 40 of them among the exported 60 notes;
    # the resolved H=60 note counts nowhere
    assert mbb["score"] == {"scored": 80, "hits": 53, "baselineHits": 20, "sampleStatus": "insufficient-sample"}
    exported = [n for n in mbb["notes"] if n["status"] == "scored"]
    assert (len(exported), sum(n["hit"] for n in exported)) == (40, 27)
    assert _symbol(out, "VCB")["score"] == {"scored": 31, "hits": 25, "baselineHits": 10, "sampleStatus": "ok"}
    assert _symbol(out, "VIC")["score"] == {"scored": 0, "hits": 0, "baselineHits": 0, "sampleStatus": "no-data"}
    assert _symbol(out, "ACB")["score"] is None  # UNSUPPORTED_CAPABILITY: no score at all, never 0/0

    extra = _market(Clock())  # a further barrier config listed first: the entry is chosen by horizonSessions
    extra.configs["MBB"] = (10, 20, 60)
    assert _run(extra, Clock(), screen_dir, tmp_path / "extra", monkeypatch) == 0
    assert _symbol(tmp_path / "extra", "MBB")["score"] == mbb["score"]


def test_retrospective_notes_take_slots_and_the_scores_take_covers_the_notes_take(screen_dir, tmp_path, monkeypatch):
    fake = _market(Clock())
    fake.journals["MBB"], days = _crowded_journal()
    out = tmp_path / "data"
    assert _run(fake, Clock(), screen_dir, out, monkeypatch) == 0
    mbb = _symbol(out, "MBB")
    # the backend's newest 200 notes are 199 retrospective ones and the forward note of days[199], whose score item is
    # the 201st item (the bar's retrospective note was written first): it is exported with its status, while the score
    # still counts the 80 older resolved notes
    assert mbb["notes"] == [{"signalDate": days[199], "horizonSessions": 20, "direction": "down",
                             "thesis": "the only forward note in the take", "status": "pending",
                             "pendingReason": "target-bar-missing", "targetDate": _target(days[199]),
                             "returnPct": None, "hit": None}]
    assert mbb["score"] == {"scored": 80, "hits": 40, "baselineHits": 16, "sampleStatus": "insufficient-sample"}
    assert not any(u["symbol"] == "MBB" for u in _meta(out)["unavailable"])


@pytest.mark.parametrize("corruption", sorted(CORRUPTIONS))
def test_a_value_outside_the_contract_makes_the_notes_part_malformed(corruption, screen_dir, tmp_path, monkeypatch):
    out = tmp_path / "data"
    assert _run(_market(Clock()), Clock(), screen_dir, out, monkeypatch) == 0
    before = _symbol(out, "MBB")
    fake = _market(Clock())
    CORRUPTIONS[corruption](fake)
    assert _run(fake, Clock(), screen_dir, out, monkeypatch) == 0
    mbb = _symbol(out, "MBB")
    assert (mbb["notes"], mbb["score"]) == (before["notes"], before["score"])  # kept from the previous run
    reasons = [u["reason"] for u in _meta(out)["unavailable"] if u["symbol"] == "MBB" and u["part"] == "notes"]
    assert reasons == [MALFORMED + KEPT]  # no exception name on the public site (r3 E5)

    fresh = tmp_path / "fresh"  # no previous run: no notes and no score, never 0/0
    fake = _market(Clock())
    CORRUPTIONS[corruption](fake)
    assert _run(fake, Clock(), screen_dir, fresh, monkeypatch) == 0
    assert (_symbol(fresh, "MBB")["notes"], _symbol(fresh, "MBB")["score"]) == ([], None)
    assert _symbol(fresh, "VCB")["score"]["scored"] == 31  # one ticker's payload never touches another


# The end-to-end check (what the site reads): the cold-check r1 strings and one sentence per r3 rule group.
END_TO_END_PRIVATE = COLDCHECK_PRIVATE + (
    "Consider reducing on a close below 18600", "Take\u2011profit at 21000",
    "Build a small position within the 20-session range", "Of the 3,000 MBB, 1,000 settle on Wednesday",
    "\u202eyub the dip", "Đợi giá đóng cửa trên 20.100 rồi mới tham gia",
)


@pytest.mark.parametrize("private", END_TO_END_PRIVATE)
def test_a_thesis_naming_private_or_action_text_is_withheld(private, screen_dir, tmp_path, monkeypatch, capsys):
    fake = _market(Clock())
    _newest(fake)[0]["body"]["thesis"] = f"MBB 1d closed below SMA50, RSI14 34. {private}. View: sideways."
    out = tmp_path / "data"
    assert _run(fake, Clock(), screen_dir, out, monkeypatch) == 0
    summary = _summary_line(capsys)
    notes = _symbol(out, "MBB")["notes"]
    assert notes[0]["thesis"] is None and (notes[0]["direction"], notes[0]["status"]) == ("up", "pending")
    assert notes[1]["thesis"] == _thesis("MBB", 1)  # a clean thesis is still published
    assert summary["thesesWithheld"] == 2  # this one and the fixture's own (days[2])
    for p in out.rglob("*.json"):
        assert not any(private in v for v in _strings(json.loads(p.read_text(encoding="utf-8")))), p.name


@pytest.mark.parametrize("text", TECHNICAL_THESES)
def test_a_technical_thesis_is_published_unchanged(text):
    assert web.public_thesis(text) == (text, False)


@pytest.mark.parametrize("text", PRIVATE_THESES)
def test_private_or_action_text_anywhere_withholds_the_thesis(text):
    assert web.public_thesis(f"MBB 1d closes near SMA50. {text}. View: sideways, for reference only.") == (None, True)


@pytest.mark.parametrize("text", R2_LEAKS)
def test_every_cold_check_r2_leak_sentence_is_withheld(text):
    assert web.public_thesis(f"MBB 1d closes near SMA50. {text}. View: sideways, for reference only.") == (None, True)


@pytest.mark.parametrize("text", REVIEW_LEAKS + COLDCHECK_R3_LEAKS + COLDCHECK_R4_LEAKS + REVIEW_R4B_LEAKS)
def test_every_review_r3_probe_sentence_is_withheld_alone_in_a_carrier_and_in_a_real_thesis(text):
    real = REAL_THESES[0]
    half = real.find(". ", len(real) // 2) + 2  # a sentence boundary in the middle of a published thesis
    for thesis in (text, f"MBB 1d closes near SMA50. {text}. View: sideways, for reference only.",
                   f"{real[:half]}{text}. {real[half:]}"):
        assert web.public_thesis(thesis) == (None, True), thesis


@pytest.mark.parametrize("form", E1_FORMS + E1_VI)
def test_every_listed_action_word_is_withheld_in_every_form(form):
    # the bare form too: "near 18,600" alone withholds "long near 18,600" and "order near 18,600" (review r3)
    for text in (f"MBB 1d closes near SMA50; {form} near 18,600. View: sideways, for reference only.",
                 f"MBB 1d closes near SMA50; {form}. View: sideways, for reference only."):
        assert web.public_thesis(text) == (None, True), text


@pytest.mark.parametrize("text", ACTION_EXTRAS + E2_PRIVATE + E3_PRIVATE + ONE_RULE)
def test_each_rule_withholds_a_sentence_of_its_own(text):
    assert web.public_thesis(f"MBB 1d closes near SMA50. {text}. View: sideways, for reference only.") == (None, True)


@pytest.mark.parametrize("text", ALONE_PRIVATE)
def test_a_private_sentence_that_starts_the_thesis_is_withheld(text):
    assert web.public_thesis(text) == (None, True)


@pytest.mark.parametrize("text", E1_NOT_LISTED + E2_TECHNICAL + E3_TECHNICAL + (E4_READABLE,) + E4_READABLE_MORE)
def test_chart_language_the_filter_keeps_is_published_unchanged(text):
    assert web.public_thesis(text) == (text, False)


@pytest.mark.parametrize("dash", DASHES)
def test_every_dash_reads_as_a_hyphen(dash):
    for private in (f"take{dash}profit at 21,000", f"cut{dash}loss at 18,500", f"the 3,000{dash}share lot"):
        assert web.public_thesis(f"MBB 1d closes near SMA50. {private}. View: sideways.") == (None, True), private
    technical = f"Its sell{dash}off on 2.85x volume broke the range; position in the 60{dash}session range 0.26."
    assert web.public_thesis(technical) == (technical, False)


@pytest.mark.parametrize("text", E4_UNREADABLE + E4_UNREADABLE_MORE)
def test_a_character_the_lists_cannot_read_withholds_the_thesis(text):
    assert web.public_thesis(f"MBB 1d closes near SMA50. {text}. View: sideways, for reference only.") == (None, True)


def test_the_patterns_read_the_lowercased_probe_and_the_cased_rules_keep_their_case():
    # PRIVATE_TEXT has no IGNORECASE and reads the lowercased probe (sre then skips a branch on its first letter), so
    # an uppercase letter in one of its patterns could never match; the ticker rules read the probe as written.
    assert all(p == p.lower() for p in web.PRIVATE_PATTERNS) and not web.PRIVATE_TEXT.flags & re.IGNORECASE
    assert web.public_thesis("MBB 1d closes near SMA50. POSITION: 30%. View: sideways.") == (None, True)
    assert web.public_thesis("MBB 1d closes near SMA50. POS\u0130T\u0130ON: 30%. View: sideways.") == (None, True)
    bid = "BID 1d closes near SMA50; the BID 20 EMA is flat and 2026-11-02 BID volume was 2.1x its average."
    assert web.public_thesis(bid) == (bid, False)


# --- round r4 ---

CARRIER = "MBB 1d closes near SMA50. {}. View: sideways, for reference only."


@pytest.mark.parametrize("text", X2_LISTS + X3_TECHNICAL)
def test_a_list_after_a_number_and_a_vnd_price_are_published_unchanged(text):
    # X2: "+6.8, VCB" is a list (the comma ends the number), not a share count; X3: a VND price is not an amount
    for thesis in (text, CARRIER.format(text.rstrip("."))):
        assert web.public_thesis(thesis) == (thesis, False), thesis


@pytest.mark.parametrize("text", X2_COUNTS + X3_PRIVATE)
def test_a_number_before_a_ticker_and_every_new_amount_or_phrase_form_withholds_the_thesis(text):
    assert web.public_thesis(CARRIER.format(text)) == (None, True)


def test_an_unsigned_count_written_first_withholds_at_the_end_of_the_text_too():
    # round r4b B2: the end-of-clause rule reads the end of the text as well (the CARRIER always adds ". View: ...");
    # review of r4b: whitespace that ends the text is the end (NIT 3: the probe folds it into one space), and a double
    # quotation mark after the ticker is a character other than a space, a comma or an apostrophe (NIT 5, K31)
    for text in ("Book: 3,000, MBB", "The book has 3,000, MBB and 1,500, BID", "Overnight carry: 900, VCB ",
                 "Overnight carry: 900, VCB\n", "Overnight carry: 900, VCB\r\n", "Overnight carry: 900, VCB ",
                 'The log line read "900, VCB" at the close', "The log line read “900, VCB” at the close"):
        assert web.public_thesis(text) == (None, True), text
    for text in ("vs BID +6.8, VCB", "vs BID +6.8, VCB "):  # a signed number is a list, whatever ends the text
        assert web.public_thesis(text) == (text, False), text


def _dashed(text: str) -> str:
    """Every space between two words with a letter written as a dash ("average price 21,500": "average-price 21,500")."""
    words = text.split(" ")
    return "".join(w + ("" if i == len(words) - 1 else "-" if any(c.isalpha() for c in w)
                        and any(c.isalpha() for c in words[i + 1]) else " ") for i, w in enumerate(words))


def _flat(text: str) -> str:
    """A text as one line: whitespace folded, Markdown backticks dropped."""
    return " ".join(text.replace("`", "").split())


def _says_the_separators(text: str) -> None:
    """The text (the module docstring or data-contract.md) names every phrase of SPACE_OR_DASH in its list of phrases
    that take a dash and every phrase of SPACE_ONLY in its list of phrases that take only spaces, the forms that are
    not listed, the "sideline" rule and the break-even exception."""
    flat = _flat(text)
    low = flat.lower()
    head, tail = low.split("these match only with spaces:", 1)
    dash_list = head[head.index("the words of a listed phrasal verb or compound match with a space or a dash"):]
    space_list = tail[:tail.index('"sideline" matches only')]
    for phrase in SPACE_OR_DASH:
        assert phrase.lower() in dash_list and phrase.lower() not in space_list, phrase
    for phrase in SPACE_ONLY:
        assert phrase.lower() in space_list and phrase.lower() not in dash_list, phrase
    for form in FORMS_NOT_LISTED:
        assert f'"{form}"' in flat, form
    assert '"sideline" matches only as one word or with a dash' in flat
    assert ('"break even" right before lower, higher, further, more or deeper is chart prose ("broke even lower")'
            in flat)
    # round r4b B1: the exception is for the verb written with a space only
    assert 'written with a dash or as one word, the noun always matches ("the break-even higher", "breakeven higher")' \
        in flat


def _says_the_count_connectors(text: str) -> None:
    """Round r4b B2/B5: the text names every connector of _COUNT_AFTER_TICKER (its character class in order, " x ", the
    spaced dash and its words in order, up to two of them) and the two other exceptions to the list reading."""
    flat = _flat(text)
    source = web._COUNT_AFTER_TICKER
    chars = re.search(r"\[([^\]]+)\]", source).group(1)
    words = re.search(r"\(\?:(about\|[^)]*)\)", source).group(1).replace("\\.", ".").split("|")
    assert "one or two of " + ", ".join(f'"{c}"' for c in chars[:-1]) + f' and "{chars[-1]}"' in flat, chars
    assert ('" x ", one or two dashes with a space on each side, ' + ", ".join(words[:-1]) + f" and {words[-1]}"
            in flat), words
    assert "{1,2}" in source and " x ?| --? | ?(?:" in source  # what the two sentences above describe
    assert "the comma is glued to the ticker" in flat
    assert "the number has no sign and the ticker is followed by the end of the text or by a character other than a " \
           "space, a comma or an apostrophe" in flat
    assert "whitespace at the end of the text counting as the end" in flat  # review of r4b, NIT 3


def _says_the_limit_of_the_action_check(text: str) -> None:
    """Round r4b B5: a wait/market-only note of a session that knew the holding passes the action check, which is
    the skill's default (review of r4b, NIT 5: the reviewer's K34), and a kept thesis passed it only from r4 on."""
    flat = _flat(text)
    assert "in a session that knew the holding passes it and only the keyword filter below reads its thesis" in flat
    assert "writes hold only when the user says they hold the stock" in flat
    assert "wait with market-only is its default" in flat
    assert "r4 on, a kept thesis passed this check when it was first exported" in flat


# Review of r4 (MINOR 1 and NIT 5): the examples the docstring and data-contract.md give for amounts and for
# "<number> <ticker>": each example is in both texts, and the filter withholds or publishes it as the text says.
DOC_WITHHELD = ("63 million", "a 63-million reserve", "50k VND", "120m dong", "a 900k-dong buffer", "63,000,000.00 VND",
                "63 000 000 VND", "63000000-dong", "VND: 63,000,000", "VND 63 000 000", "VND 120000000", "VND 120m",
                "USD 50k", "2.5m dollars", "a 2,500-dollar reserve", "3,000 MBB", "3K MBB", "3 nghìn MBB",
                "MBB 3,000, BID 1,500", "MBB 18900, BID 31700", "at 18,750, BID 1d",
                # round r4b: B1, B2, B3 and B4
                "the break-even higher", "breakeven higher", "MBB: 3,000, BID: 1,500", "BID 1,200, MBB (2,600)",
                "BID 1,200, MBB: ~2,600", "2,600,MBB", "3,000, MBB.", "1200000 VND", "1.2B VND", "VND1.2B",
                "1.2B USD", "63,000,000 Vietnam dong",
                # review of r4b: a space before k, m or b in each of its rules, and the country's name
                "63 m dong", "VND 120 m", "1.2 B USD", "1.2B Vietnam dong", "63,000,000 Việt Nam đồng")
DOC_PUBLISHED = ("18,750 VND", "118500 VND", "61,000 dong", "63,000,000 in VND terms", "+6.8, VCB +5.1",
                 "broke even lower",
                 # round r4b: a six-digit price, what r4 opened on purpose, and the residual it names as not caught
                 "999999 VND", "+6.8, VCB", "18,550, MBB bounced", "In 2025, MBB, BID and VCB", "18900, MBB's lowest",
                 "MBB at 18900, BID at 31700", "3,000, MBB and 1,500, BID at the close",
                 # review of r4b (NIT 4): what the chart exception of the spaced verb lets through, named as such
                 "broke even more than a week ago")


def _says_the_examples(text: str) -> None:
    flat = _flat(text)
    for example in DOC_WITHHELD + DOC_PUBLISHED:
        assert example in flat, example
    names = re.fullmatch(r"\(\?!(.*)\)", web._NOT_TICKER).group(1).split("|")  # the exceptions of "<number> <ticker>"
    assert f"the indicator abbreviations {', '.join(names[:-1])}, nor {names[-1]}" in flat


def test_the_docstring_says_which_phrases_take_a_dash_and_which_forms_are_not_listed():
    # X4: the docstring names every phrase that matches only with spaces, and claims a dash only where the filter
    # withholds it; it also names forms of listed words that are not listed. A filter change that widens or narrows
    # one of these turns this test red until the docstring says so (and the next test: until data-contract.md does).
    _says_the_separators(web.__doc__)
    for example in SPACE_OR_DASH.values():
        for text in (example, _dashed(example)):
            assert web.public_thesis(CARRIER.format(text)) == (None, True), text
    for example in SPACE_ONLY.values():
        assert web.public_thesis(CARRIER.format(example)) == (None, True), example
        dashed = CARRIER.format(_dashed(example))
        assert web.public_thesis(dashed) == (dashed, False), dashed
    for form in FORMS_NOT_LISTED:
        assert web.public_thesis(CARRIER.format(form)) == (CARRIER.format(form), False), form
    for text, hidden in (("sideline", True), ("side-line", True), ("side line", False)):
        assert web.public_thesis(CARRIER.format(text))[1] is hidden, text


def test_the_docstring_examples_of_amounts_and_tickers_are_what_the_filter_does():
    _says_the_examples(web.__doc__)
    for example in DOC_WITHHELD:
        assert web.public_thesis(CARRIER.format(example)) == (None, True), example
    for example in DOC_PUBLISHED:
        assert web.public_thesis(CARRIER.format(example)) == (CARRIER.format(example), False), example
    for name in re.fullmatch(r"\(\?!(.*)\)", web._NOT_TICKER).group(1).split("|"):
        assert web.public_thesis(CARRIER.format(f"the 20 {name}"))[1] is False, name
    assert web.public_thesis(CARRIER.format("the 200 DMA")) == (None, True)  # not one of them: withheld


def test_the_docstring_names_the_connectors_of_a_count_and_the_limit_of_the_action_check():
    # round r4b B5: widening or narrowing _COUNT_AFTER_TICKER, or dropping the X1 caveat, turns this red until the
    # docstring says so (and the next test: until data-contract.md does)
    _says_the_count_connectors(web.__doc__)
    _says_the_limit_of_the_action_check(web.__doc__)
    # review of r4b (NIT 5, the reviewer's K33): where the theses exported before r4 came from
    assert "the theses exported before r4 all came from wait/market-only notes" in _flat(web.__doc__)


def test_the_data_contract_says_what_the_docstring_says():
    contract = AI_ROOT.parent / "evidence" / "vn30-web-20261103" / "data-contract.md"
    if not contract.is_file():
        pytest.skip("the data contract lives outside the ai repo (btc workspace only)")
    text = contract.read_text(encoding="utf-8")
    _says_the_separators(text)
    _says_the_examples(text)
    _says_the_count_connectors(text)
    _says_the_limit_of_the_action_check(text)
    # round r4b B5: "Kept parts" says the action check holds for a kept thesis from r4 on, not before
    kept = _flat(text.split("## Kept parts", 1)[1].split("\n## ", 1)[0])
    assert "From r4 on, a kept thesis is one that check let through when it was first exported" in kept
    # review of r4b (NIT 5): where the theses exported before r4 came from
    assert "all come from forward notes stored with actionView wait and actionBasis market-only" in _flat(text)


@pytest.mark.parametrize("text", BREAK_EVEN_PROSE)
def test_broke_even_lower_and_its_kin_are_chart_prose(text):
    # review of r4 (NIT 3): the break-even rule leaves "broke even lower" (higher, further, more, deeper) to the chart
    for thesis in (text, CARRIER.format(text.rstrip("."))):
        assert web.public_thesis(thesis) == (thesis, False), thesis
    assert web.public_thesis(CARRIER.format(text.replace("broke even", "broke even near 21,500, and"))) == (None, True)


@pytest.mark.parametrize("text", BREAK_EVEN_NOUN)
def test_break_even_with_a_dash_or_as_one_word_withholds_before_lower_higher_and_kin(text):
    # round r4b B1: the noun matches whatever follows it, as in r3; only the spaced verb has the chart exception
    # (BREAK_EVEN_PROSE is the guard the other way: widening this back over "broke even lower" turns it red)
    for thesis in (text, CARRIER.format(text.rstrip(".")), R4_THESES[0] + " " + text):
        assert web.public_thesis(thesis) == (None, True), thesis


@pytest.mark.parametrize("text", sorted(VND_EDGE))
def test_a_vnd_amount_starts_at_seven_digits(text):
    thesis = CARRIER.format(text)
    assert web.public_thesis(thesis) == ((None, True) if VND_EDGE[text] else (thesis, False))


def _one_note(change=None, thesis: str = REAL_THESES[0]) -> tuple[list[dict], dict, int]:
    """note_rows on a journal of one forward H=20 note with the daily task's actionView/actionBasis, after `change`."""
    note, item = _entry("MBB", "2026-11-02", 1, thesis=thesis, pending="immature", target=_target("2026-11-02"))
    if change is not None:
        change(note)
    return web.note_rows({"items": [note]}, {"horizons": [_summary(20, [item])], "items": [item]})


@pytest.mark.parametrize("basis", ACTION_BASES)
@pytest.mark.parametrize("view", ACTION_VIEWS)
def test_a_thesis_is_published_only_for_a_market_only_note_with_wait_or_none(view, basis):
    published, score, _ = _one_note()
    assert published[0]["thesis"] == REAL_THESES[0]
    rows, same_score, withheld = _one_note(lambda n: n.update(actionView=view, actionBasis=basis))
    assert same_score == score
    if (basis, view) in PUBLIC_ACTIONS:
        assert (rows, withheld) == (published, 0)
    else:  # the row stays (direction, status, dates); only the thesis is withheld, and it counts
        assert (rows, withheld) == ([dict(published[0], thesis=None)], 1)


@pytest.mark.parametrize("odd", sorted(ODD_ACTIONS))
def test_a_missing_or_odd_action_withholds_the_thesis_and_keeps_the_row(odd):
    published, _, _ = _one_note()
    rows, _, withheld = _one_note(ODD_ACTIONS[odd])  # never a malformed notes part
    assert (rows, withheld) == ([dict(published[0], thesis=None)], 1)


def test_a_note_without_a_thesis_counts_nothing_whatever_its_action():
    for thesis in ("", "  \n "):
        rows, _, withheld = _one_note(lambda n: n.update(actionView="hold", actionBasis="position-aware"), thesis)
        assert (rows[0]["thesis"], withheld) == (None, 0)


def test_unavailable_reasons_are_plain_vietnamese_and_the_details_go_to_stderr(screen_dir, tmp_path, monkeypatch,
                                                                               capsys):
    fake = _market(Clock(), errors={("MBB", "technical"): [(500, "INTERNAL")], ("VCB", "candles"): [(503, "UPSTREAM")]})
    fake.technical["VCB"]["timeframes"][1]["indicators"] = None  # no 1d indicator row; VIC: VNINDEX stale; ACB: 400
    _newest(fake, "VCB")[1].update(status="b\u00e1n")  # a malformed notes payload whose detail is not ASCII
    out = tmp_path / "data"
    capsys.readouterr()
    assert _run(fake, Clock(), screen_dir, out, monkeypatch) == 0
    captured = capsys.readouterr()
    meta = _meta(out)
    assert [(u["symbol"], u["part"], u["reason"]) for u in meta["unavailable"]] == [
        ("ACB", "technical", UNSUPPORTED), ("ACB", "notes", UNSUPPORTED),
        ("MBB", "technical", "lỗi HTTP 500" + NOTHING_KEPT), ("VCB", "candles", "lỗi HTTP 503" + NOTHING_KEPT),
        ("VCB", "technical", NO_INDICATORS), ("VCB", "notes", MALFORMED + NOTHING_KEPT),
        ("VIC", "technical", NO_STRENGTH)]
    _assert_public_reasons(meta)
    assert _stderr_lines(captured) == [  # one line per meta.unavailable entry, in the same order
        "unavailable ACB technical: HTTP 400 UNSUPPORTED_CAPABILITY",
        "unavailable ACB notes: HTTP 400 UNSUPPORTED_CAPABILITY",
        "unavailable MBB technical: HTTP 500 INTERNAL", "unavailable VCB candles: HTTP 503 UPSTREAM",
        "unavailable VCB technical: no 1d indicator row",
        "unavailable VCB notes: ValueError: not a contract value: 'b\\xe1n'",
        "unavailable VIC technical: VNINDEX benchmark status 'stale', reason None"]
    for p in out.rglob("*.json"):  # no internal code in any public file
        text = p.read_text(encoding="utf-8")
        codes = ("UNSUPPORTED_CAPABILITY", "INTERNAL", "UPSTREAM", "Error", "stale")
        assert not [c for c in codes if c in text], p.name
    summary = json.loads(captured.out.strip().splitlines()[-1])  # the stdout summary keeps its shape
    assert list(summary) == ["session", "symbols", "written", "changed", "unavailableByPart", "keptByPart",
                             "thesesWithheld"]
    assert summary["unavailableByPart"] == {"technical": 4, "notes": 2, "candles": 1} and summary["keptByPart"] == {}


def test_a_long_thesis_is_cut_to_1500_characters_with_a_marker():
    base = "Bollinger middle 18,900 caps the 1d close. "
    exact, longer = (base * 40)[:1500], (base * 40)[:1501]
    assert web.public_thesis(exact) == (exact, False)
    assert web.public_thesis(longer) == (longer[:1499] + "\u2026", False)
    assert web.public_thesis(exact + " Hold.") == (None, True)  # the filter reads the whole text, past the cut
    assert web.public_thesis("  \n ") == (None, False) and web.public_thesis(None) == (None, False)
    assert web.public_thesis(["not", "text"]) == (None, False)


FAILURES = {  # part, run-2 change, public reason before the suffix, the stderr detail
    "technical-500": ("technical", {"errors": {("MBB", "technical"): [(500, "INTERNAL")]}}, "lỗi HTTP 500",
                      "HTTP 500 INTERNAL"),
    "technical-429-twice": ("technical", {"errors": {("MBB", "technical"): [429, 429]}}, "lỗi HTTP 429", "HTTP 429"),
    "technical-malformed": ("technical", {"technical": lambda t: t.pop("relativeStrength")}, MALFORMED,
                            "KeyError: 'relativeStrength'"),
    "notes-500": ("notes", {"errors": {("MBB", "research-notes"): [500]}}, "lỗi HTTP 500", "HTTP 500"),
    "scores-401": ("notes", {"errors": {("MBB", "research-notes/direction-scores"): [(401, "ADMIN_KEY_INVALID")]}},
                   "lỗi HTTP 401", "HTTP 401 ADMIN_KEY_INVALID"),
    "notes-without-admin-key": ("notes", {"key": None}, NO_KEY, "EQX_ADMIN_KEY is not set"),
}


def _stderr_lines(captured) -> list[str]:
    assert captured.err.isascii()  # a redirected stderr is cp1252
    return [line for line in captured.err.splitlines() if line.startswith("unavailable ")]


def _assert_public_reasons(meta: dict) -> None:
    """Every meta.unavailable reason is one of the plain sentences of contract r3 E5: no envelope code, no exception
    name, no benchmark status (the site shows the reason as it is)."""
    for u in meta["unavailable"]:
        reason = u["reason"]
        if reason in (UNSUPPORTED, NO_INDICATORS, NO_STRENGTH):
            continue
        head = reason.removesuffix(KEPT).removesuffix(NOTHING_KEPT)
        assert head != reason and (head in (MALFORMED, NO_KEY) or (
            head.startswith("lỗi HTTP ") and head[9:].isdigit() and len(head) == 12)), reason


@pytest.mark.parametrize("failure", sorted(FAILURES))
def test_a_failed_part_keeps_the_previous_v2_run(failure, screen_dir, tmp_path, monkeypatch, capsys):
    part, change, reason, detail = FAILURES[failure]
    out = tmp_path / "data"
    assert _run(_market(Clock()), Clock(), screen_dir, out, monkeypatch) == 0
    before = _symbol(out, "MBB")
    fake = _market(Clock(), errors=change.get("errors"))
    if "technical" in change:
        change["technical"](fake.technical["MBB"])
    last = fake.bars["MBB"][-1]  # a newer bar: candles are read fresh, the file is not kept whole
    fake.bars["MBB"].append(dict(last, openTimeMs=last["openTimeMs"] + 3 * 86_400_000, volume=5))
    capsys.readouterr()
    assert _run(fake, Clock(), screen_dir, out, monkeypatch, key=change.get("key", KEY)) == 0
    captured = capsys.readouterr()
    summary = json.loads(captured.out.strip().splitlines()[-1])
    mbb = _symbol(out, "MBB")
    kept = ("indicators", "relativeStrength") if part == "technical" else ("notes", "score")
    fresh = ("notes", "score") if part == "technical" else ("indicators", "relativeStrength")
    assert all(before[f] and mbb[f] == before[f] for f in kept + fresh)  # the backend data did not change
    assert mbb["candles"][-1][0] == web.session_of(fake.bars["MBB"][-1]["openTimeMs"]) != before["candles"][-1][0]
    assert {"symbol": "MBB", "part": part, "reason": reason + KEPT} in _meta(out)["unavailable"]
    assert summary["keptByPart"][part] >= 1 and summary["written"] == len(SYMBOLS)
    assert f"unavailable MBB {part}: {detail}" in _stderr_lines(captured)  # the detail only on stderr
    _assert_public_reasons(_meta(out))


def test_a_kept_thesis_passes_the_privacy_filter_again(screen_dir, tmp_path, monkeypatch, capsys):
    out = tmp_path / "data"
    assert _run(_market(Clock()), Clock(), screen_dir, out, monkeypatch) == 0
    for s in ("MBB", "VCB"):  # as if an older filter had let them through
        doc = _symbol(out, s)
        doc["notes"][1]["thesis"] = "Hold above 19,500; cash 50m."
        (out / "symbols" / f"{s}.json").write_text(json.dumps(doc), encoding="utf-8")
    capsys.readouterr()
    errors = {("MBB", "research-notes"): [500], ("VCB", "candles"): [500]}  # MBB notes kept, VCB kept whole
    assert _run(_market(Clock(), errors=errors), Clock(), screen_dir, out, monkeypatch) == 0
    summary = _summary_line(capsys)
    mbb, vcb = _symbol(out, "MBB"), _symbol(out, "VCB")
    assert mbb["notes"][1]["thesis"] is None and vcb["notes"][1]["thesis"] is None
    assert mbb["notes"][0]["thesis"] == _thesis("MBB", 0) and vcb["notes"][0]["thesis"] == _thesis("VCB", 0)
    assert summary["thesesWithheld"] == 2 and summary["keptByPart"] == {"notes": 1, "candles": 1}
    assert summary["written"] == len(SYMBOLS) - 1
    assert not any("cash" in v for p in out.rglob("*.json") for v in _strings(json.loads(p.read_text("utf-8"))))


def _forward_pairs(fake: FakeBackend, symbol: str = "MBB") -> dict[str, tuple[dict, dict]]:
    """signal date -> the forward H=20 note that holds its bar's slot, and its direction-score item."""
    return {web.session_of(n["signalBarOpenTimeMs"]): (n, i) for n, i in fake.journals[symbol]
            if n["mode"] == "forward" and n["horizonSessions"] == 20 and i["status"] != "superseded-not-scored"}


def test_a_note_that_may_know_the_users_holding_keeps_its_row_but_never_its_thesis(screen_dir, tmp_path, monkeypatch,
                                                                                   capsys):
    # X1, end to end: a clean technical thesis (one the filter publishes) under hold (SKILL.md Step 3: the user said
    # they hold the stock) or under position-aware/reduce reaches no public file; the rows and the score stay.
    fake = _market(Clock())
    days, pairs = _mbb_days(), _forward_pairs(fake)
    pairs[days[0]][0].update(actionView="hold")
    pairs[days[0]][0]["body"]["thesis"] = R4_THESES[0]
    pairs[days[1]][0].update(actionView="reduce", actionBasis="position-aware")
    pairs[days[1]][0]["body"]["thesis"] = R4_THESES[1]
    assert all(web.public_thesis(t) == (t, False) for t in R4_THESES[:2])
    out = tmp_path / "data"
    assert _run(fake, Clock(), screen_dir, out, monkeypatch) == 0
    summary = _summary_line(capsys)
    mbb = _symbol(out, "MBB")
    assert [(n["signalDate"], n["direction"], n["status"], n["thesis"]) for n in mbb["notes"][:4]] == [
        (days[0], "up", "pending", None), (days[1], "down", "pending", None), (days[2], "sideways", "pending", None),
        (days[3], "up", "pending", "first note on the bar")]
    assert len(mbb["notes"]) == 60 and mbb["score"]["scored"] == 80
    assert summary["thesesWithheld"] == 3  # these two and the fixture's planted one (days[2])
    for p in out.rglob("*.json"):
        values = list(_strings(json.loads(p.read_text(encoding="utf-8"))))
        assert not any(t[:100] in v for t in R4_THESES[:2] for v in values), p.name


def test_a_thesis_withheld_for_its_action_stays_withheld_when_the_notes_part_is_kept(screen_dir, tmp_path, monkeypatch,
                                                                                    capsys):
    # X1 on the kept path: the files carry no actionView/actionBasis, so a kept thesis is the one the check let
    # through when it was first exported; a thesis it withheld is null in the file and stays null.
    fake = _market(Clock())
    _newest(fake)[0].update(actionView="hold")
    _newest(fake)[0]["body"]["thesis"] = R4_THESES[0]
    out = tmp_path / "data"
    assert _run(fake, Clock(), screen_dir, out, monkeypatch) == 0
    first = _symbol(out, "MBB")
    assert first["notes"][0]["thesis"] is None and first["notes"][1]["thesis"] == _thesis("MBB", 1)
    capsys.readouterr()
    assert _run(_market(Clock(), errors={("MBB", "research-notes"): [500]}), Clock(), screen_dir, out, monkeypatch) == 0
    summary = _summary_line(capsys)
    kept = _symbol(out, "MBB")
    assert summary["keptByPart"] == {"notes": 1} and (kept["notes"], kept["score"]) == (first["notes"], first["score"])
    assert not any(R4_THESES[0][:100] in v for p in out.rglob("*.json")
                   for v in _strings(json.loads(p.read_text(encoding="utf-8"))))


def _errors() -> dict:
    return {("MBB", "technical"): [500], ("MBB", "research-notes"): [500], ("VCB", "candles"): [500],
            ("ACB", "technical"): [500]}


def test_without_a_previous_v2_run_a_failed_part_is_written_empty(screen_dir, tmp_path, monkeypatch):
    fresh = tmp_path / "fresh"
    assert _run(_market(Clock(), errors=_errors()), Clock(), screen_dir, fresh, monkeypatch) == 0
    mbb, vcb = _symbol(fresh, "MBB"), _symbol(fresh, "VCB")
    assert (mbb["indicators"], mbb["relativeStrength"], mbb["notes"], mbb["score"]) == (None, None, [], None)
    assert vcb["candles"] == [] and vcb["indicators"] is not None and vcb["score"]["scored"] == 31
    unavailable = _meta(fresh)["unavailable"]
    for u in ({"symbol": "MBB", "part": "technical", "reason": "lỗi HTTP 500" + NOTHING_KEPT},
              {"symbol": "MBB", "part": "notes", "reason": "lỗi HTTP 500" + NOTHING_KEPT},
              {"symbol": "VCB", "part": "candles", "reason": "lỗi HTTP 500" + NOTHING_KEPT},
              {"symbol": "ACB", "part": "technical", "reason": "lỗi HTTP 500" + NOTHING_KEPT}):
        assert u in unavailable

    v1 = tmp_path / "v1"  # files from a v1 data set (other meaning of notes/score/thesis) are never a source
    assert _run(_market(Clock()), Clock(), screen_dir, v1, monkeypatch) == 0
    meta = _meta(v1)
    (v1 / "meta.json").write_text(json.dumps(dict(meta, schema="vn30-web-data/v1")), encoding="utf-8")
    assert _run(_market(Clock(), errors=_errors()), Clock(), screen_dir, v1, monkeypatch) == 0
    assert {s: _symbol(v1, s) for s in SYMBOLS} == {s: _symbol(fresh, s) for s in SYMBOLS}
    assert _meta(v1)["unavailable"] == unavailable

    gone = tmp_path / "gone"  # a previous v2 run whose meta.json cannot be read
    assert _run(_market(Clock()), Clock(), screen_dir, gone, monkeypatch) == 0
    (gone / "meta.json").write_text("{not json", encoding="utf-8")
    assert _run(_market(Clock(), errors=_errors()), Clock(), screen_dir, gone, monkeypatch) == 0
    assert _symbol(gone, "MBB") == mbb and _meta(gone)["unavailable"] == unavailable


def test_a_previous_part_is_kept_only_when_it_validates_and_was_read(screen_dir, tmp_path, monkeypatch):
    out = tmp_path / "data"
    assert _run(_market(Clock()), Clock(), screen_dir, out, monkeypatch) == 0  # ACB: UNSUPPORTED, nothing read
    before = _symbol(out, "MBB")
    (out / "symbols" / "MBB.json").write_text(json.dumps(dict(before, score={"scored": 2, "hits": 1})),
                                              encoding="utf-8")  # a v1-shaped score
    assert _run(_market(Clock(), errors=_errors(), unsupported=frozenset()), Clock(), screen_dir, out,
                monkeypatch) == 0
    mbb, acb = _symbol(out, "MBB"), _symbol(out, "ACB")
    assert (mbb["notes"], mbb["score"]) == ([], None) and mbb["indicators"] == before["indicators"]
    assert (acb["indicators"], acb["relativeStrength"]) == (None, None) and acb["score"]["scored"] == 1
    unavailable = _meta(out)["unavailable"]
    assert {"symbol": "MBB", "part": "notes", "reason": "lỗi HTTP 500" + NOTHING_KEPT} in unavailable
    assert {"symbol": "MBB", "part": "technical", "reason": "lỗi HTTP 500" + KEPT} in unavailable
    assert {"symbol": "ACB", "part": "technical", "reason": "lỗi HTTP 500" + NOTHING_KEPT} in unavailable


def test_unsupported_capability_is_listed_without_a_suffix_and_never_kept(screen_dir, tmp_path, monkeypatch):
    out = tmp_path / "data"
    assert _run(_market(Clock(), unsupported=frozenset()), Clock(), screen_dir, out, monkeypatch) == 0
    acb = _symbol(out, "ACB")
    assert acb["indicators"] is not None and acb["notes"] and acb["score"]["scored"] == 1  # served before
    assert _run(_market(Clock()), Clock(), screen_dir, out, monkeypatch) == 0  # now 400 UNSUPPORTED_CAPABILITY
    acb = _symbol(out, "ACB")
    assert (acb["indicators"], acb["relativeStrength"], acb["notes"], acb["score"]) == (None, None, [], None)
    assert len(acb["candles"]) == 258  # candles need only ingestion
    assert [u for u in _meta(out)["unavailable"] if u["symbol"] == "ACB"] == [
        {"symbol": "ACB", "part": "technical", "reason": UNSUPPORTED},
        {"symbol": "ACB", "part": "notes", "reason": UNSUPPORTED}]


def test_technical_calls_are_spaced_15_s_and_a_429_waits_a_minute_and_retries_once(screen_dir, tmp_path, monkeypatch):
    clock = Clock()
    fake = _market(clock, errors={("MBB", "technical"): [429], ("VCB", "technical"): [429, 429],
                                  ("VIC", "candles"): [429]})
    out = tmp_path / "data"
    assert _run(fake, clock, screen_dir, out, monkeypatch) == 0
    assert clock.sleeps.count(60.0) == 3  # MBB once, VCB once (then gives up), VIC candles once
    times = [t for p, _, t in fake.calls if p.endswith("/technical?bars=1")]
    assert len(times) == 6  # ACB, MBB x2, VCB x2, VIC
    assert all(b - a >= 15.0 for a, b in zip(times, times[1:]))
    assert all(sum(1 for u in times if t <= u < t + 60) <= 4 for t in times)
    assert _symbol(out, "MBB")["indicators"] is not None and len(_symbol(out, "VIC")["candles"]) == 258
    assert {"symbol": "VCB", "part": "technical", "reason": "lỗi HTTP 429" + NOTHING_KEPT} in _meta(out)["unavailable"]


def test_without_the_admin_key_notes_are_never_requested(screen_dir, tmp_path, monkeypatch):
    fake = _market(Clock())
    out = tmp_path / "data"
    assert _run(fake, Clock(), screen_dir, out, monkeypatch, key=None) == 0
    assert not any("research-notes" in p for p, _, _ in fake.calls)
    notes_parts = [u for u in _meta(out)["unavailable"] if u["part"] == "notes"]
    assert sorted(u["symbol"] for u in notes_parts) == sorted(SYMBOLS)
    assert all(u["reason"] == NO_KEY + NOTHING_KEPT for u in notes_parts)
    assert (_symbol(out, "MBB")["notes"], _symbol(out, "MBB")["score"]) == ([], None)

    with_key = _market(Clock())
    assert _run(with_key, Clock(), screen_dir, tmp_path / "keyed", monkeypatch) == 0
    for path, headers, _ in with_key.calls:  # the key goes only to the two admin endpoints
        assert ("X-Admin-Key" in headers) == ("research-notes" in path)
        assert headers.get("X-Admin-Key") in (None, KEY)


def test_unchanged_files_are_not_rewritten(screen_dir, tmp_path, monkeypatch):
    out = tmp_path / "data"
    fake = _market(Clock())
    assert _run(fake, Clock(), screen_dir, out, monkeypatch) == 0
    meta = _meta(out)
    (out / "meta.json").write_bytes(web._dump(dict(meta, generatedAtUtc="2026-01-01T00:00:00Z")))
    files = sorted(out.rglob("*.json"))
    before = {p: p.read_bytes() for p in files}
    for p in files:
        os.utime(p, ns=(1_000_000_000, 1_000_000_000))
    assert _run(_market(Clock()), Clock(), screen_dir, out, monkeypatch) == 0
    assert all(p.stat().st_mtime_ns == 1_000_000_000 and p.read_bytes() == before[p] for p in files)

    newer = _market(Clock())
    last = newer.bars["MBB"][-1]
    newer.bars["MBB"].append(dict(last, openTimeMs=last["openTimeMs"] + 3 * 86_400_000, volume=5))
    assert _run(newer, Clock(), screen_dir, out, monkeypatch) == 0
    rewritten = {p.relative_to(out).as_posix() for p in files if p.stat().st_mtime_ns != 1_000_000_000}
    assert rewritten == {"symbols/MBB.json", "meta.json"}
    assert _meta(out)["generatedAtUtc"] != "2026-01-01T00:00:00Z"
    assert not list(out.rglob("*.tmp"))
    assert (out / "meta.json").read_bytes().endswith(b"\n") and b"\r\n" not in (out / "meta.json").read_bytes()


def test_a_file_is_replaced_whole_and_no_tmp_file_is_left(screen_dir, tmp_path, monkeypatch):
    out = tmp_path / "data"
    assert _run(_market(Clock()), Clock(), screen_dir, out, monkeypatch) == 0
    before = (out / "symbols" / "MBB.json").read_bytes()
    (out / "symbols" / "ZZZ.json.tmp").write_bytes(b"left by an interrupted run")
    real_replace = os.replace

    def replace(src, dst):
        if str(dst).endswith("MBB.json"):
            raise PermissionError("the file is held by another process")
        real_replace(src, dst)

    monkeypatch.setattr(web.os, "replace", replace)
    newer = _market(Clock())
    last = newer.bars["MBB"][-1]
    newer.bars["MBB"].append(dict(last, openTimeMs=last["openTimeMs"] + 3 * 86_400_000, volume=5))
    with pytest.raises(PermissionError):
        _run(newer, Clock(), screen_dir, out, monkeypatch)
    assert (out / "symbols" / "MBB.json").read_bytes() == before  # never half-written
    assert not list(out.rglob("*.tmp"))


def test_a_ticker_whose_candles_fail_keeps_its_previous_file(screen_dir, tmp_path, monkeypatch):
    out = tmp_path / "data"
    assert _run(_market(Clock()), Clock(), screen_dir, out, monkeypatch) == 0
    previous = (out / "symbols" / "VCB.json").read_bytes()
    fake = _market(Clock(), errors={("VCB", "candles"): [500]})
    fake.technical["VCB"]["timeframes"][1]["indicators"]["rsi14"] = 1.0  # would change the file if it were rebuilt
    assert _run(fake, Clock(), screen_dir, out, monkeypatch) == 0
    assert (out / "symbols" / "VCB.json").read_bytes() == previous
    assert not any(p.startswith("/api/equities/VCB/") and "/candles" not in p for p, _, _ in fake.calls)
    assert {"symbol": "VCB", "part": "candles", "reason": "lỗi HTTP 500" + KEPT} in _meta(out)["unavailable"]

    fresh = tmp_path / "fresh"
    assert _run(_market(Clock(), errors={("VCB", "candles"): [500]}), Clock(), screen_dir, fresh, monkeypatch) == 0
    vcb = _symbol(fresh, "VCB")
    assert vcb["candles"] == [] and vcb["indicators"] is not None and vcb["notes"]
    assert {"symbol": "VCB", "part": "candles", "reason": "lỗi HTTP 500" + NOTHING_KEPT} in _meta(fresh)["unavailable"]

    again = _market(Clock(), errors={("VCB", "candles"): [500]})  # the previous file has no candles either: nothing
    assert _run(again, Clock(), screen_dir, fresh, monkeypatch) == 0  # to keep, so the other parts are read again
    assert any(p.startswith("/api/equities/VCB/technical") for p, _, _ in again.calls)
    assert {"symbol": "VCB", "part": "candles", "reason": "lỗi HTTP 500" + NOTHING_KEPT} in _meta(fresh)["unavailable"]


def test_screen_json_follows_the_newest_screen_file(screen_dir, tmp_path, monkeypatch):
    out = tmp_path / "data"
    assert _run(_market(Clock()), Clock(), screen_dir, out, monkeypatch) == 0
    files = _load(out)
    screen, meta = files["screen.json"], files["meta.json"]
    assert screen["session"] == SESSION and meta["session"] == SESSION
    assert [r["symbol"] for r in screen["rows"]] == list(SYMBOLS)  # the screen's order is kept
    assert meta["symbols"] == sorted(SYMBOLS) and meta["schema"] == "vn30-web-data/v2"
    assert screen["rows"][1] == {"symbol": "MBB", "close": 2000.0, "returns": {"3": 0.02, "6": 0.04, "12": 0.06},
                                 "excessVsIndex": {"3": -0.02, "6": -0.04, "12": -0.06}, "rank": 2, "top": True}
    assert screen["rows"][2]["top"] is False and screen["rows"][3]["rank"] is None
    assert (screen["basket"], screen["rankingMonths"], screen["topK"]) == \
           ({"effectiveDate": "2026-08-03", "possiblyStale": False}, 6, 10)
    assert screen["indexReturns"] == {"3": -0.014, "6": -0.088, "12": 0.035}
    assert (screen["missingSessionBar"], screen["notUpToDate"]) == (["VIC"], ["ACB"])


def test_exit_codes(screen_dir, tmp_path, monkeypatch):
    def unreachable(path, headers):
        raise urllib.error.URLError("connection refused")
    assert _run(unreachable, Clock(), screen_dir, tmp_path / "a", monkeypatch) == 1

    def broken_list(path, headers):
        raise _http(500)
    assert _run(broken_list, Clock(), screen_dir, tmp_path / "b", monkeypatch) == 1

    fake = _market(Clock())
    real = fake.__call__

    def dies_midway(path, headers):  # the backend stops after the first tickers: nothing is written
        if path.startswith("/api/equities/VCB/"):
            raise urllib.error.URLError("connection refused")
        return real(path, headers)
    assert _run(dies_midway, Clock(), screen_dir, tmp_path / "c", monkeypatch) == 1

    assert _run(_market(Clock()), Clock(), tmp_path / "missing", tmp_path / "d", monkeypatch) == 2
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "2026-11-03.json").write_text(json.dumps(_screen(schema="vn30-screen/v0")), encoding="utf-8")
    assert _run(_market(Clock()), Clock(), bad, tmp_path / "e", monkeypatch) == 2
    (bad / "2026-11-04.json").write_text(json.dumps(_screen(symbols=("MBB", "../X"))), encoding="utf-8")
    assert _run(_market(Clock()), Clock(), bad, tmp_path / "f", monkeypatch) == 2
    for d in "abcdef":
        assert not (tmp_path / d).exists()


def test_http_fetch_sends_the_headers_it_is_given(monkeypatch):
    seen = {}

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def urlopen(request, timeout):
        seen["url"], seen["key"], seen["timeout"] = request.full_url, request.get_header("X-admin-key"), timeout
        return Response(b'{"items": []}')
    monkeypatch.setattr(web.urllib.request, "urlopen", urlopen)
    assert web.http_fetch("http://127.0.0.1:1/")("/api/x?y=1", {"X-Admin-Key": KEY}) == {"items": []}
    assert seen == {"url": "http://127.0.0.1:1/api/x?y=1", "key": KEY, "timeout": 120.0}
