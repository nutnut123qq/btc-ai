"""VN30-2 portfolio backtester: multi-asset, long-only, HOSE rules (roadmap item 2.1).

Pure computation (no network, no DB). Given a session calendar, per-symbol adjusted daily bars, the
point-in-time VN30 membership and a scoring function ``score(date, view) -> {symbol: score}``, it simulates
the portfolio rules of docs/research/protocols/vn30-2.json and returns daily NAV, trades, rejected orders,
rebalances, turnover and costs.

Rules (``PortfolioRules.from_protocol`` reads them from the frozen protocol; defaults equal it):

- Decisions every ``rebalance_every`` sessions (20), from a first decision session (by default the protocol's
  "first session with >= 260 prior sessions for >= 20 members", see ``first_decision_index``). The signal is
  the close of decision session d: ``score`` receives a ``PointInTimeView`` that exposes bars dated <= d only.
- Orders are sent for the open of d+1. Long-only, equal weight 1/top_k per name (top 10); hysteresis: a
  holding (shares > 0 at the close of d) is kept while its rank <= keep_rank (15). Ranks are by descending
  score, ties by symbol. Only members in force at d with a finite score are ranked, so a member that left
  VN30 is sold at the next rebalance. ``top_k=None`` is the equal-weight benchmark (all ranked members).
- Universe (protocol universe.minimumHistorySessions): only members with >= ``min_history_sessions`` (260)
  bars strictly before d are ranked. The filter is applied here, before ranking, identically for every model
  and for the equal-weight benchmark, so the scorer does not decide the universe.
- Sizing at the open of d+1 (the backtester knows that open when it sizes, which an ATO order would not;
  declared simplification): target value = (NAV at the open - estimated slippage, fee and tax of selling the
  dropped names and of trimming the kept ones) / top_k; target shares = whole lots of
  target value / (buy price x (1 + buy fee)), the all-in buy cost. Each target name, kept ones included, is
  traded to its target when the difference is at least one lot; names not targeted are sold in full. A name
  with no bar at the open is sized from its last close.
- Every session sells run first (by symbol), then buys (by rank). Buy price = open + slippage, sell price =
  open - slippage; slippage = ``slippage_ticks`` HOSE ticks of the adjusted open (10 / 50 / 100 VND below
  10,000 / 50,000 / above), the frozen portfolio.costs.tickRule. ``raw_tick_slippage=True`` (off by default;
  using it is a NEW registered trial) uses instead, where raw limit info exists for the bar, the tick of the
  raw open as a fraction applied to the adjusted open. Fees 0.15% of gross each side, sell tax 0.1% of gross.
- T+2 shares: shares bought at session i become sellable from the open of session i + settlement_trading_days
  (as in vn_execution_model). Known simplification: on HOSE shares bought on T arrive around midday of T+2,
  so selling at the T+2 open is slightly optimistic; ``shares_sellable_after_sessions=3`` makes them sellable
  from the open of i+3 (the first open after the T+2 afternoon). A sell fills only the settled, lot-aligned
  shares; the rest is retried.
- Cash: sale proceeds are spendable after ``cash_settlement_sessions`` sessions. The default 0 (same-session
  buying power from matched sales) departs from vn_execution_model (cash usable after T+2): on HOSE it is a
  broker cash advance on sale proceeds, whose interest is NOT charged here. The protocol is silent on it, so
  the value is recorded in every result (rules and summary) and must be confirmed per run; 2 is strict cash
  T+2.
- Limit-locked opens do not fill (protocol data.limitFillProxy): a buy is not filled when
  open >= 1.065 x previous close, a sell when open <= 0.935 x previous close (previous close = the symbol's
  last close before the session, seeded from the history before the first decision). When ``limits`` gives
  raw info for that bar, it replaces the proxy for the fill check only: a buy is blocked when
  raw open >= ceiling, a sell when raw open <= floor.
- An order that does not fill (no bar that session, limit-locked, shares not settled) is retried every next
  session until it fills or the next rebalance replaces it ("the miss is carried to the next session"). A buy
  short of cash is cut to what spendable cash affords; the shortfall (or the whole buy) is kept and retried
  while any sell order is still open or sale proceeds are pending, and dropped only when neither remains (or
  at the next rebalance).
- Optional risk exits (``risk_exit``, off by default; using it is a NEW registered trial, e.g. stops-1): after
  each session's fills the hook gets a ``RiskContext`` (positions, the session's fills, open orders) and returns
  {symbol: shares to sell}. Each becomes a sell order for the next open (origin "risk-exit") with the usual fill
  rules, replacing an open buy of that symbol ("superseded-by-risk-exit"); a symbol with an open sell order is
  skipped. Exits returned at a decision session's close are ignored: the rebalance at the next open supersedes
  every order.
- Optional exposure overlay (``exposure``, off by default = 1; using it is a NEW registered trial, e.g.
  overlay-1): at each decision close, after the targets are selected, ``exposure(date, view, targets)`` returns a
  gross exposure in [0, 1] that multiplies the rebalance budget at the next open (0 sells everything); selection
  is unchanged and the value is recorded in the rebalance record.
- Unused weight stays in cash at 0%. Positions are marked at the close (last close when a bar is missing). A
  held position without a bar for more than ``stale_after`` sessions (delisted or suspended) is counted in
  the daily rows and the summary (count and value) instead of being silently marked at a stale price.

All prices are the adjusted basis (``PRICE_BASIS``); records carry it so they are not mistaken for matched
prices. Prices are VND per share unless ``price_unit_vnd`` says otherwise (e.g. 1000 for thousand-VND data).
"""
from __future__ import annotations

import json
import math
import re
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple, Sequence

from vn30_rotation import Periods, members_on
from vn_execution_model import HOSE_LOT_SIZE, PRICE_BASIS, VnEquityCostSpec, buy_shares_for_budget

SCHEMA = "vn-portfolio-backtest/v1"
DEFAULT_INITIAL_CASH_VND = 10_000_000_000.0

Bar = tuple  # (open, high, low, close[, volume]) on the adjusted basis
Bars = Mapping[str, Mapping[str, Bar]]  # symbol -> ISO session date -> bar


class LimitInfo(NamedTuple):
    """Raw (unadjusted) open and the day's price band, when such data exists for a bar.

    All three are VND per share (as vn_raw_prices returns them), whatever ``price_unit_vnd`` the adjusted bars
    use; ``run_backtest`` rejects values that do not look like VND (raw open below 1,000)."""

    raw_open: float
    floor: float
    ceiling: float


Limits = Mapping[str, Mapping[str, LimitInfo]]
MIN_RAW_PRICE_VND = 1_000.0
DEFAULT_STALE_AFTER_SESSIONS = 5


def validate_limits(limits: Limits) -> None:
    """Fail loudly on raw limit info that is not VND or not a band (e.g. thousand-VND data)."""
    for s, rows in limits.items():
        for day, info in rows.items():
            raw_open, floor, ceiling = (float(x) for x in info)
            if not (math.isfinite(raw_open) and math.isfinite(floor) and math.isfinite(ceiling)):
                raise ValueError(f"limit info {s} {day}: non-finite value")
            if raw_open < MIN_RAW_PRICE_VND:
                raise ValueError(f"limit info {s} {day}: raw open {raw_open} < {MIN_RAW_PRICE_VND:.0f}; "
                                 "LimitInfo must be in VND, not thousand VND")
            if not 0 < floor < ceiling:
                raise ValueError(f"limit info {s} {day}: floor {floor} / ceiling {ceiling} is not a band")


def hose_tick(price_vnd: float) -> float:
    """HOSE tick size (VND) for a price in VND: 10 below 10,000, 50 below 50,000, else 100."""
    if price_vnd <= 0:
        raise ValueError("price must be positive")
    if price_vnd < 10_000:
        return 10.0
    if price_vnd < 50_000:
        return 50.0
    return 100.0


def lot_floor(shares: float, lot_size: int = HOSE_LOT_SIZE) -> int:
    """Whole lots (rounded down) of a share count; never negative."""
    if shares <= 0:
        return 0
    n = int(math.floor(shares + 1e-9))
    return n - n % lot_size


@dataclass(frozen=True)
class PortfolioRules:
    rebalance_every: int = 20
    top_k: int | None = 10
    keep_rank: int | None = 15
    slippage_ticks: int = 1
    limit_up_proxy: float = 1.065
    limit_down_proxy: float = 0.935
    cash_settlement_sessions: int = 0
    initial_cash_vnd: float = DEFAULT_INITIAL_CASH_VND
    price_unit_vnd: float = 1.0
    costs: VnEquityCostSpec = field(default_factory=VnEquityCostSpec)
    min_history_sessions: int = 260
    raw_tick_slippage: bool = False
    shares_sellable_after_sessions: int | None = None  # None = costs.settlement_trading_days (open of T+2)

    @property
    def share_lag(self) -> int:
        """Sessions after a buy before its shares can be sold (at that session's open)."""
        if self.shares_sellable_after_sessions is None:
            return self.costs.settlement_trading_days
        return self.shares_sellable_after_sessions

    def __post_init__(self) -> None:
        if self.rebalance_every < 1:
            raise ValueError("rebalance_every must be >= 1")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be >= 1 or None")
        if self.top_k is not None and self.keep_rank is not None and self.keep_rank < self.top_k:
            raise ValueError("keep_rank must be >= top_k")
        if self.slippage_ticks < 0 or self.cash_settlement_sessions < 0:
            raise ValueError("slippage_ticks and cash_settlement_sessions cannot be negative")
        if not (self.limit_down_proxy < 1.0 < self.limit_up_proxy):
            raise ValueError("limit proxies must bracket 1.0")
        if self.initial_cash_vnd <= 0 or self.price_unit_vnd <= 0:
            raise ValueError("initial cash and price unit must be positive")
        if self.min_history_sessions < 0:
            raise ValueError("min_history_sessions cannot be negative")
        if (self.shares_sellable_after_sessions is not None
                and self.shares_sellable_after_sessions < self.costs.settlement_trading_days):
            raise ValueError("shares_sellable_after_sessions cannot be earlier than settlement_trading_days")

    @classmethod
    def from_protocol(cls, protocol: Mapping[str, Any] | Path, **overrides: Any) -> "PortfolioRules":
        """Rules of a research-protocol/v1 file (vn30-2.json); fails loudly on anything it cannot read."""
        if isinstance(protocol, Path):
            protocol = json.loads(protocol.read_text(encoding="utf-8"))
        port = protocol["portfolio"]
        construction = port["construction"]
        top = re.search(r"top (\d+)", construction)
        keep = re.search(r"rank <= (\d+)", construction)
        proxy = re.findall(r"open ([<>]=) ([0-9.]+) x previous close", protocol["data"]["limitFillProxy"])
        costs = port["costs"]
        if not top or not keep or len(proxy) != 2 or costs["buyFee"] != costs["sellFee"]:
            raise ValueError("protocol portfolio rules not in the expected form")
        bounds = {op: float(v) for op, v in proxy}
        schedule = protocol["decisionSchedule"]
        values: dict[str, Any] = {
            "rebalance_every": int(schedule["rebalanceEverySessions"]),
            "top_k": int(top.group(1)),
            "keep_rank": int(keep.group(1)),
            "slippage_ticks": int(costs["slippageTicks"]),
            "min_history_sessions": int(protocol["universe"]["minimumHistorySessions"]),
            "limit_up_proxy": bounds[">="],
            "limit_down_proxy": bounds["<="],
            "costs": VnEquityCostSpec(broker_fee_fraction=float(costs["buyFee"]),
                                      sell_tax_fraction=float(costs["sellTax"]),
                                      lot_size=int(port["lotSize"])),
        }
        values.update(overrides)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {"rebalanceEverySessions": self.rebalance_every, "topK": self.top_k, "keepRank": self.keep_rank,
                "slippageTicks": self.slippage_ticks, "limitUpProxy": self.limit_up_proxy,
                "limitDownProxy": self.limit_down_proxy, "cashSettlementSessions": self.cash_settlement_sessions,
                "initialCashVnd": self.initial_cash_vnd, "priceUnitVnd": self.price_unit_vnd,
                "minHistorySessions": self.min_history_sessions, "rawTickSlippage": self.raw_tick_slippage,
                "sharesSellableAfterSessions": self.share_lag, "costs": self.costs.to_dict()}


class PointInTimeView:
    """What a scoring function may see at the close of a decision session: nothing dated after it."""

    def __init__(self, calendar: Sequence[str], index: int, bars: Bars, dates: Mapping[str, list[str]],
                 members: frozenset[str]):
        self._calendar = calendar
        self._index = index
        self._bars = bars
        self._dates = dates
        self.date = calendar[index]
        self.members = members

    @property
    def calendar(self) -> tuple[str, ...]:
        return tuple(self._calendar[: self._index + 1])

    @property
    def symbols(self) -> list[str]:
        return sorted(s for s, ds in self._dates.items() if ds and ds[0] <= self.date)

    def history(self, symbol: str) -> list[tuple[str, Bar]]:
        """(date, bar) of the symbol, oldest first, dated <= the decision session."""
        ds = self._dates.get(symbol, [])
        bars = self._bars[symbol] if ds else {}
        return [(d, tuple(bars[d])) for d in ds[: bisect_right(ds, self.date)]]

    def closes(self, symbol: str) -> list[float]:
        return [float(b[3]) for _, b in self.history(symbol)]


ScoreFn = Callable[[str, PointInTimeView], Mapping[str, float]]


def rank_members(scores: Mapping[str, float], members: frozenset[str]) -> list[str]:
    """Members with a finite score, best first (descending score, ties by symbol)."""
    ok = {s: float(v) for s, v in scores.items()
          if s in members and v is not None and math.isfinite(float(v))}
    return sorted(ok, key=lambda s: (-ok[s], s))


def select_targets(ranked: list[str], held: set[str], top_k: int | None, keep_rank: int | None) -> list[str]:
    """Top-k with hysteresis: holdings ranked <= keep_rank stay, the best others fill to k. Rank order."""
    if top_k is None:
        return list(ranked)
    keep_limit = keep_rank if keep_rank is not None else top_k
    kept = [s for s in ranked[:keep_limit] if s in held][:top_k]
    fill = [s for s in ranked if s not in kept][: top_k - len(kept)]
    chosen = set(kept) | set(fill)
    return [s for s in ranked if s in chosen]


def first_decision_index(calendar: Sequence[str], bars: Bars, periods: Periods, min_history: int = 260,
                         min_members: int = 20) -> int | None:
    """First session with >= min_history prior sessions (bars strictly before it) for >= min_members of the
    members then in force (vn30-2.json decisionSchedule.firstDecision)."""
    dates = {s: sorted(b) for s, b in bars.items()}
    for i, day in enumerate(calendar):
        members = members_on(periods, day)
        if not members:
            continue
        enough = sum(1 for s in members if bisect_left(dates.get(s, []), day) >= min_history)
        if enough >= min_members:
            return i
    return None


@dataclass
class _Order:
    symbol: str
    side: str
    shares: int
    decision_date: str
    rank: int  # buys execute in rank order; sells use 0
    origin: str = "rebalance"  # or "risk-exit"


@dataclass(frozen=True)
class RiskContext:
    """What a risk-exit hook sees at the close of a session (after that session's fills)."""

    date: str
    index: int
    shares: Mapping[str, int]  # symbol -> shares held (settled or not)
    fills: Sequence[Mapping[str, Any]]  # trade records of this session, in fill order
    open_orders: Mapping[str, str]  # symbol -> "buy" / "sell" still open after this session
    rebalance_entry: bool  # this session was a rebalance's entry session
    decision_session: bool  # a rebalance decision was made at this close (exits returned now are ignored)


RiskExitFn = Callable[[RiskContext], Mapping[str, int]]
ExposureFn = Callable[[str, "PointInTimeView", Sequence[str]], float]


@dataclass
class BacktestResult:
    rules: PortfolioRules
    daily: list[dict[str, Any]]
    trades: list[dict[str, Any]]
    rejections: list[dict[str, Any]]
    rebalances: list[dict[str, Any]]
    summary: dict[str, Any]

    def nav_by_date(self) -> dict[str, float]:
        return {r["date"]: r["nav"] for r in self.daily}

    def to_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "priceBasis": PRICE_BASIS, "rules": self.rules.to_dict(),
                "summary": self.summary, "rebalances": self.rebalances, "daily": self.daily,
                "trades": self.trades, "rejections": self.rejections}


def run_backtest(calendar: Sequence[str], bars: Bars, periods: Periods, score: ScoreFn,
                 rules: PortfolioRules | None = None, *, first_decision: int | None = None,
                 end: str | None = None, limits: Limits | None = None,
                 stale_after: int = DEFAULT_STALE_AFTER_SESSIONS,
                 risk_exit: RiskExitFn | None = None,
                 exposure: ExposureFn | None = None) -> BacktestResult:
    """Simulate the portfolio from the first decision session to ``end`` (default: last session).

    ``limits`` (raw VND open and band per symbol and session) replaces the limit-lock proxy where present.
    ``stale_after``: a held position with no bar for more than this many sessions is reported as stale.
    ``risk_exit``: optional hook returning risk-exit sells at each close (see the module docstring).
    ``exposure``: optional hook returning the gross exposure in [0, 1] of each decision (module docstring)."""
    rules = rules or PortfolioRules()
    costs = rules.costs
    lot = costs.lot_size
    calendar = list(calendar)
    if calendar != sorted(set(calendar)):
        raise ValueError("calendar must be unique, chronological ISO dates")
    if first_decision is None:
        first_decision = first_decision_index(calendar, bars, periods)
        if first_decision is None:
            raise ValueError("no session satisfies the first-decision history rule")
    last = len(calendar) - 1 if end is None else bisect_right(calendar, end) - 1
    if not 0 <= first_decision <= last:
        raise ValueError("first decision outside the simulated range")
    limits = limits or {}
    validate_limits(limits)
    if stale_after < 0:
        raise ValueError("stale_after cannot be negative")
    dates = {s: sorted(b) for s, b in bars.items()}

    cash = rules.initial_cash_vnd / rules.price_unit_vnd  # internal money is in price units
    pending_cash: list[tuple[int, float]] = []  # (session index it becomes spendable, amount)
    lots: dict[str, list[tuple[int, int]]] = {}  # symbol -> [(session index sellable from, shares)]
    last_close: dict[str, float] = {}
    for s, ds in dates.items():  # the symbol's last close before the first simulated session
        k = bisect_left(ds, calendar[first_decision])
        if k:
            last_close[s] = float(bars[s][ds[k - 1]][3])
    missing: dict[str, int] = {}  # held symbol -> consecutive sessions without a bar
    orders: dict[str, _Order] = {}
    decision: dict[str, Any] | None = None
    daily, trades, rejections, rebalances = [], [], [], []

    def shares_of(s: str) -> int:
        return sum(n for _, n in lots.get(s, []))

    def settled_of(s: str, i: int) -> int:
        return sum(n for k, n in lots.get(s, []) if k <= i)

    def bar_of(s: str, day: str) -> Bar | None:
        return bars.get(s, {}).get(day)

    def mark(day: str, use_open: bool) -> float:
        total = cash + sum(a for _, a in pending_cash)
        for s in lots:
            b = bar_of(s, day)
            px = float(b[0] if use_open else b[3]) if b is not None else last_close[s]
            total += shares_of(s) * px
        return total

    def slip(s: str, day: str, open_px: float) -> float:
        info = limits.get(s, {}).get(day) if rules.raw_tick_slippage else None
        if info is not None:
            return open_px * rules.slippage_ticks * hose_tick(info.raw_open) / info.raw_open
        return rules.slippage_ticks * hose_tick(open_px * rules.price_unit_vnd) / rules.price_unit_vnd

    def locked(s: str, day: str, side: str, open_px: float) -> bool:
        info = limits.get(s, {}).get(day)
        if info is not None:
            return info.raw_open >= info.ceiling if side == "buy" else info.raw_open <= info.floor
        prev = last_close.get(s)
        if prev is None:
            return False
        return open_px >= rules.limit_up_proxy * prev if side == "buy" else open_px <= rules.limit_down_proxy * prev

    def funding_pending() -> bool:
        """Cash may still come: a sell order is open or sale proceeds are not yet spendable."""
        return bool(pending_cash) or any(x.side == "sell" for x in orders.values())

    def exit_cost_of(s: str, day: str, shares: int) -> float:
        """Estimated slippage, fee and tax of selling ``shares`` at the open (0 without a bar)."""
        b = bar_of(s, day)
        if b is None or shares <= 0:
            return 0.0
        px = float(b[0])
        sp = slip(s, day, px)
        return shares * (sp + (px - sp) * (costs.broker_fee_fraction + costs.sell_tax_fraction))

    def size(targets: list[str], day: str, budget: float) -> dict[str, int]:
        per_name = budget / (rules.top_k if rules.top_k is not None else max(len(targets), 1))
        out: dict[str, int] = {}
        for s in targets:
            b = bar_of(s, day)
            ref = float(b[0]) if b is not None else last_close.get(s)
            if ref is None:
                continue  # no price ever seen: nothing to size from
            all_in = (ref + slip(s, day, ref)) * (1.0 + costs.broker_fee_fraction)
            out[s] = lot_floor(per_name / all_in, lot)
        return out

    def reject(day: str, o: _Order, reason: str, retried: bool) -> None:
        rejections.append({"date": day, "symbol": o.symbol, "side": o.side, "shares": o.shares,
                           "decisionDate": o.decision_date, "reason": reason, "retried": retried})

    for i in range(first_decision, last + 1):
        day = calendar[i]
        ready = [a for k, a in pending_cash if k <= i]
        pending_cash = [(k, a) for k, a in pending_cash if k > i]
        cash += sum(ready)
        nav_start = mark(calendar[i - 1], use_open=False) if i > first_decision else cash
        day_traded = day_fee = day_tax = day_slip = 0.0
        fills_from = len(trades)
        rebalance_entry = decision is not None

        if decision is not None:  # entry session of the previous decision: replace every open order
            for o in orders.values():
                reject(day, o, "superseded-by-rebalance", retried=False)
            orders = {}
            nav_open = mark(day, use_open=True)
            targets = decision["targets"]
            # estimated slippage, fee and tax of selling the dropped names, then of trimming the kept ones
            exit_cost = sum(exit_cost_of(s, day, shares_of(s)) for s in lots if s not in targets)
            scale = decision.get("exposure", 1.0)
            first = size(targets, day, (nav_open - exit_cost) * scale)
            trim_cost = sum(exit_cost_of(s, day, shares_of(s) - n) for s, n in first.items()
                            if shares_of(s) - n >= lot)
            target_shares = size(targets, day, (nav_open - exit_cost - trim_cost) * scale)
            for rank, s in enumerate(targets, start=1):
                if s not in target_shares:
                    continue
                diff = target_shares[s] - shares_of(s)
                if diff >= lot:
                    orders[s] = _Order(s, "buy", diff, decision["decisionDate"], rank)
                elif diff <= -lot:
                    orders[s] = _Order(s, "sell", -diff, decision["decisionDate"], 0)
            for s in sorted(lots):
                if s not in target_shares and shares_of(s) > 0:
                    orders[s] = _Order(s, "sell", shares_of(s), decision["decisionDate"], 0)
            rebalances.append({**decision, "entryDate": day, "navAtOpen": nav_open,
                               "targetShares": target_shares})
            decision = None

        sells = sorted((o for o in orders.values() if o.side == "sell"), key=lambda o: o.symbol)
        buys = sorted((o for o in orders.values() if o.side == "buy"), key=lambda o: (o.rank, o.symbol))
        for o in sells + buys:
            b = bar_of(o.symbol, day)
            if b is None:
                reject(day, o, "no-bar", retried=True)
                continue
            open_px = float(b[0])
            if locked(o.symbol, day, o.side, open_px):
                reject(day, o, "limit-locked", retried=True)
                continue
            s_px = slip(o.symbol, day, open_px)
            if o.side == "sell":
                qty = min(o.shares, lot_floor(settled_of(o.symbol, i), lot))
                if qty <= 0:
                    reject(day, o, "shares-not-settled", retried=True)
                    continue
                price = open_px - s_px
                gross = qty * price
                fee, tax = gross * costs.broker_fee_fraction, gross * costs.sell_tax_fraction
                remaining = qty
                kept_lots = []
                for k, n in sorted(lots[o.symbol]):  # consume settled lots, oldest first
                    take = min(n, remaining) if k <= i else 0
                    remaining -= take
                    if n - take:
                        kept_lots.append((k, n - take))
                lots[o.symbol] = kept_lots
                if not kept_lots:
                    del lots[o.symbol]
                proceeds = gross - fee - tax
                if rules.cash_settlement_sessions == 0:
                    cash += proceeds
                else:
                    pending_cash.append((i + rules.cash_settlement_sessions, proceeds))
            else:
                price = open_px + s_px
                affordable = buy_shares_for_budget(cash, price, costs)
                qty = min(o.shares, affordable)
                if qty < lot:
                    keep = funding_pending()
                    reject(day, o, "insufficient-cash", retried=keep)
                    if not keep:
                        del orders[o.symbol]
                    continue
                gross = qty * price
                fee, tax = gross * costs.broker_fee_fraction, 0.0
                cash -= gross + fee
                lots.setdefault(o.symbol, []).append((i + rules.share_lag, qty))
            trades.append({"date": day, "symbol": o.symbol, "side": o.side, "shares": qty, "open": open_px,
                           "price": price, "grossVnd": gross * rules.price_unit_vnd,
                           "feeVnd": fee * rules.price_unit_vnd, "taxVnd": tax * rules.price_unit_vnd,
                           "slippageVnd": qty * s_px * rules.price_unit_vnd, "decisionDate": o.decision_date,
                           "origin": o.origin, "priceBasis": PRICE_BASIS})
            day_traded += gross
            day_fee += fee
            day_tax += tax
            day_slip += qty * s_px
            o.shares -= qty
            if o.shares <= 0:
                del orders[o.symbol]
            elif o.side == "buy":
                keep = funding_pending()
                reject(day, o, "cash-capped", retried=keep)  # shares = the unfilled remainder
                if not keep:
                    del orders[o.symbol]

        for s, b in bars.items():
            if day in b:
                last_close[s] = float(b[day][3])
        for s in [s for s in missing if s not in lots]:
            del missing[s]
        for s in lots:
            missing[s] = 0 if bar_of(s, day) is not None else missing.get(s, 0) + 1
        stale = sorted(s for s, n in missing.items() if n > stale_after)
        nav = mark(day, use_open=False)
        daily.append({"date": day, "nav": nav * rules.price_unit_vnd,
                      "cash": (cash + sum(a for _, a in pending_cash)) * rules.price_unit_vnd,
                      "positions": len(lots), "tradedVnd": day_traded * rules.price_unit_vnd,
                      "feeVnd": day_fee * rules.price_unit_vnd, "taxVnd": day_tax * rules.price_unit_vnd,
                      "slippageVnd": day_slip * rules.price_unit_vnd,
                      "turnover": day_traded / nav_start if nav_start > 0 else 0.0,
                      "stalePositions": len(stale),
                      "staleValueVnd": sum(shares_of(s) * last_close[s] for s in stale) * rules.price_unit_vnd})

        if (i - first_decision) % rules.rebalance_every == 0 and i < len(calendar) - 1 and i < last:
            in_force = members_on(periods, day) or frozenset()
            members = frozenset(s for s in in_force
                                if bisect_left(dates.get(s, []), day) >= rules.min_history_sessions)
            view = PointInTimeView(calendar, i, bars, dates, members)
            ranked = rank_members(score(day, view), members)
            held = {s for s in lots if shares_of(s) > 0}
            targets = select_targets(ranked, held, rules.top_k, rules.keep_rank)
            decision = {"decisionDate": day, "ranks": {s: r for r, s in enumerate(ranked, start=1)},
                        "targets": targets, "heldBefore": sorted(held),
                        "excludedShortHistory": sorted(in_force - members)}
            if exposure is not None:
                scale = float(exposure(day, view, tuple(targets)))
                if not (math.isfinite(scale) and 0.0 <= scale <= 1.0):
                    raise ValueError(f"{day}: exposure {scale} outside [0, 1]")
                decision["exposure"] = scale

        if risk_exit is not None:
            ctx = RiskContext(day, i, {s: shares_of(s) for s in lots}, tuple(trades[fills_from:]),
                              {s: o.side for s, o in orders.items()}, rebalance_entry, decision is not None)
            exits = risk_exit(ctx)
            if decision is None:
                for s in sorted(exits):
                    n = min(int(exits[s]), shares_of(s))
                    if n <= 0 or (s in orders and orders[s].side == "sell"):
                        continue
                    if s in orders:
                        reject(day, orders[s], "superseded-by-risk-exit", retried=False)
                    orders[s] = _Order(s, "sell", n, day, 0, "risk-exit")

    summary = _summary(daily, trades, rules)
    end_stale = [{"symbol": s, "sessionsWithoutBar": missing[s], "shares": shares_of(s),
                  "markedAt": last_close[s], "valueVnd": shares_of(s) * last_close[s] * rules.price_unit_vnd}
                 for s in sorted(missing) if missing[s] > stale_after]
    summary.update({"staleAfterSessions": stale_after,
                    "stalePositionSessions": sum(r["stalePositions"] for r in daily),
                    "maxStalePositions": max((r["stalePositions"] for r in daily), default=0),
                    "maxStaleValueVnd": max((r["staleValueVnd"] for r in daily), default=0.0),
                    "stalePositionsAtEnd": end_stale,
                    "stalePositionsAtEndValueVnd": sum(x["valueVnd"] for x in end_stale)})
    return BacktestResult(rules, daily, trades, rejections, rebalances, summary)


def _summary(daily: list[dict[str, Any]], trades: list[dict[str, Any]], rules: PortfolioRules) -> dict[str, Any]:
    start_nav = rules.initial_cash_vnd * rules.price_unit_vnd
    end_nav = daily[-1]["nav"] if daily else start_nav
    years = ((date.fromisoformat(daily[-1]["date"]) - date.fromisoformat(daily[0]["date"])).days / 365.25
             if len(daily) > 1 else 0.0)
    fee = sum(t["feeVnd"] for t in trades)
    tax = sum(t["taxVnd"] for t in trades)
    slippage = sum(t["slippageVnd"] for t in trades)
    turnover = sum(r["turnover"] for r in daily)
    return {"from": daily[0]["date"] if daily else None, "to": daily[-1]["date"] if daily else None,
            "sessions": len(daily), "years": years, "startNavVnd": start_nav, "endNavVnd": end_nav,
            "totalReturn": end_nav / start_nav - 1.0, "trades": len(trades),
            "tradedVnd": sum(t["grossVnd"] for t in trades), "feeVnd": fee, "taxVnd": tax,
            "slippageVnd": slippage, "costVnd": fee + tax + slippage,
            "turnoverTwoSided": turnover, "annualTurnoverTwoSided": turnover / years if years > 0 else None,
            "cashSettlementSessions": rules.cash_settlement_sessions,
            "sharesSellableAfterSessions": rules.share_lag, "rawTickSlippage": rules.raw_tick_slippage,
            "minHistorySessions": rules.min_history_sessions}


__all__ = ["BacktestResult", "LimitInfo", "PointInTimeView", "PortfolioRules", "RiskContext", "RiskExitFn", "ExposureFn", "SCHEMA",
           "first_decision_index", "hose_tick","lot_floor", "rank_members", "run_backtest", "select_targets", "validate_limits"]
