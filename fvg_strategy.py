"""
Opening-range Fair-Value-Gap (FVG) strategy -- clean backtest, local CSV data.

Each session it fades the opening gap: on a gap-up open it only looks for shorts,
on a gap-down only longs. It takes the FIRST fair-value gap in the allowed direction
inside the 09:30 -> open_end window, enters on a limit at the gap's near edge when
price trades through it, stops at the most recent swing, and targets a reward_risk
multiple of the stop. One trade per market per day.

Data: local CSV exports in cfg.csv_dir -- {SYMBOL}_1m.csv / {SYMBOL}_5m.csv
(1-minute CME futures; other timeframes are resampled from the 1-minute file).

Run:  python fvg_strategy.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from datetime import time

import matplotlib.pyplot as plt
import pandas as pd

SYMBOLS = ["ES", "NQ"]   # CME futures roots (also the CSV file names)


@dataclass
class Config:
    timeframe: str = "5min"      # 1min, 5min, 15min, 30min, 1hour
    reward_risk: float = 1.0     # take-profit distance as a multiple of risk (1:1)
    swing_lookback: int = 2      # bars on each side that define a swing pivot
    start_date: str = "2016-1-1"
    end_date: str = "2026-10-16"
    open_start: time = time(9, 30)
    open_end: time = time(12, 10)   # only look for the FVG inside this window
    session_close: time = time(16, 0)
    entry_by: time = time(12, 10)   # retest must fill by this time, else skip the trade
    gap_bias: bool = True           # True: only short after a gap-up open, long after gap-down
    csv_dir: str = r"/volumes/t9/futures_data"  # folder of {SYMBOL}_1m.csv / {SYMBOL}_5m.csv
    account_start: float = 1_000_000  # starting cash for the compounding account sim
    risk_per_trade: float = 0.01    # fraction risked per trade (1%)
    compound_risk: bool = True    # True: 1% of the live balance (grows). False: flat 1% of the start
    currency: str = "£"             # symbol used when printing account balances
    slippage_ticks: float = 1.0     # round-turn slippage in ticks (limit entry ~0, ~1 on the stop exit)
    fill_through_ticks: float = 0.0  # ticks price must trade BEYOND a limit (entry AND target) to fill
    slip_on_exits: tuple = ("stop", "close")  # exits that cross the spread and so pay slippage
    entry_offset_ticks: float = 0.0  # ticks to move the limit TOWARDS the market (worse fill price)
    is_start: str = "2024-01-01"    # start of the window used to pick the parameters
    is_end: str = "2024-06-30"      # end of it -- every other session is out-of-sample
    exclude_roll_days: bool = True  # drop sessions whose gap looks like a contract roll


# --------------------------------------------------------------------------- #
# Data -- local CSV (1-minute futures exports, resampled to any timeframe)
# --------------------------------------------------------------------------- #
_RESAMPLE = {"1min": "1min", "5min": "5min", "15min": "15min", "30min": "30min", "1hour": "1h"}


def fetch_csv(symbol: str, cfg: Config) -> pd.DataFrame:
    """Load OHLC bars for `symbol` from a local CSV export in cfg.csv_dir.

    Reads {symbol}_1m.csv / {symbol}_5m.csv directly when one matches the
    timeframe, else resamples the 1-minute file. No API call, no cost.
    """
    tag = {"1min": "1m", "5min": "5m"}.get(cfg.timeframe)
    direct = os.path.join(cfg.csv_dir, f"{symbol}_{tag}.csv") if tag else ""
    path = direct if direct and os.path.exists(direct) else os.path.join(cfg.csv_dir, f"{symbol}_1m.csv")

    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    df = df.set_index("timestamp").sort_index()
    if path.endswith("_1m.csv") and cfg.timeframe != "1min":
        df = (df.resample(_RESAMPLE.get(cfg.timeframe, "5min"), label="left", closed="left")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna())
    df.index.name = "date"
    return df.loc[cfg.start_date:cfg.end_date, ["open", "high", "low", "close"]]


# --------------------------------------------------------------------------- #
# Strategy
# --------------------------------------------------------------------------- #
def last_swing(values, idx, lookback, want_high):
    """Most recent confirmed swing high/low strictly before bar `idx` (no look-ahead)."""
    for p in range(idx - lookback - 1, lookback - 1, -1):
        window = values[p - lookback : p + lookback + 1]
        if values[p] == (window.max() if want_high else window.min()):
            return values[p]
    return None


def find_trade(day: pd.DataFrame, cfg: Config, bias: int = 0, tick: float = 0.25) -> dict | None:
    """First opening-range FVG of the session in the allowed direction, simulated
    to its exit. `bias`: +1 longs only, -1 shorts only, 0 either. `tick` is the
    market's minimum price increment -- _simulate needs it to price the
    fill-through requirement in cfg.fill_through_ticks."""
    h, l = day["high"].to_numpy(), day["low"].to_numpy()
    times = day.index.time

    for i in range(2, len(day)):
        if times[i] > cfg.open_end:        # past the entry window without a gap
            return None

        long_fvg = l[i] > h[i - 2]         # gap up: bar i's low above bar i-2's high
        short_fvg = h[i] < l[i - 2]        # gap down: bar i's high below bar i-2's low
        if not (long_fvg or short_fvg):
            continue

        direction = 1 if long_fvg else -1
        if bias and direction != bias:     # gap bias: skip the disallowed side
            continue
        # Entry is a limit at the gap's near edge; it only fills if price later
        # retraces into the gap (long -> the gap high, short -> the gap low).
        # cfg.entry_offset_ticks steps that limit TOWARDS the market -- up for a long,
        # down for a short -- buying queue position with a worse price. The stop level
        # is untouched, so the offset widens risk and pushes the 1:1 target twice as far
        # again; it is a real order-placement decision, not a costing assumption.
        entry = l[i] if direction == 1 else h[i]
        entry += direction * cfg.entry_offset_ticks * tick

        if direction == 1:
            stop = last_swing(l, i, cfg.swing_lookback, want_high=False)
            if stop is None or stop >= entry:          # fall back to the gap's own low
                stop = l[i - 2 : i + 1].min()
        else:
            stop = last_swing(h, i, cfg.swing_lookback, want_high=True)
            if stop is None or stop <= entry:          # fall back to the gap's own high
                stop = h[i - 2 : i + 1].max()

        risk = abs(entry - stop)
        if risk == 0:
            return None
        target = entry + direction * cfg.reward_risk * risk
        return _simulate(day, i, direction, entry, stop, target, risk, cfg, tick)

    return None


def _simulate(day, i, direction, entry, stop, target, risk, cfg, tick=0.25) -> dict | None:
    """Wait for price to trade through the gap edge, then run the trade to its exit.

    A resting limit is NOT filled by a touch. Price has to trade cfg.fill_through_ticks
    ticks beyond the limit before the order is treated as done -- 0 ticks reproduces the
    old touch-is-a-fill assumption, which flatters the results because the touch-and-
    reverse cases are exactly the winners you would not have been filled on. The fill
    price is still the limit itself.

    This applies to BOTH limits. The entry and the take-profit are the same kind of
    order and face the same queue, so exempting the target would mean claiming passive
    pricing with aggressive certainty. The stop is different: it becomes a market order
    the moment it is touched, so a touch triggers it and it pays slippage instead.
    Returns None if the first FVG is never filled by cfg.entry_by.
    """
    h, l, c = (day[col].to_numpy() for col in ("high", "low", "close"))
    times = day.index.time

    through = cfg.fill_through_ticks * tick     # how far past a limit price must trade
    entry_idx = None
    for j in range(i + 1, len(day)):
        if entry_idx is None:                      # still waiting for the retest
            if times[j] > cfg.entry_by:            # no fill within the entry window
                return None
            filled = (l[j] <= entry - through) if direction == 1 else (h[j] >= entry + through)
            if not filled:
                continue
            entry_idx = j                          # limit fills at the gap edge

        # Touching the stop triggers it (market order); the target has to trade through.
        hit_stop = l[j] <= stop if direction == 1 else h[j] >= stop
        hit_target = ((h[j] >= target + through) if direction == 1
                      else (l[j] <= target - through))
        if hit_stop:                               # conservative: stop wins ties
            exit_price, reason, exit_idx = stop, "stop", j
            break
        if hit_target:
            exit_price, reason, exit_idx = target, "target", j
            break
    else:
        if entry_idx is None:
            return None                            # gap edge never traded through -> no trade
        exit_price, reason, exit_idx = c[-1], "close", len(day) - 1  # ran to the close

    pnl = direction * (exit_price - entry)
    return {
        "fvg_time": day.index[i],
        "entry_time": day.index[entry_idx],
        "exit_time": day.index[exit_idx],
        "direction": "long" if direction == 1 else "short",
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "exit": round(exit_price, 2),
        "reason": reason,
        "pnl": round(pnl, 2),
        "R": round(pnl / risk, 2),
    }


# --------------------------------------------------------------------------- #
# Backtest + report
# --------------------------------------------------------------------------- #
def session_gaps(sessions: dict) -> pd.Series:
    """Overnight gap per session: the RTH open minus the prior session's RTH close."""
    opens = pd.Series({d: day["open"].iloc[0] for d, day in sessions.items()}).sort_index()
    closes = pd.Series({d: day["close"].iloc[-1] for d, day in sessions.items()}).sort_index()
    return (opens - closes.shift(1)).dropna()


def flag_roll_days(gaps: pd.Series, window: int = 60, n_sd: float = 4.0) -> pd.Series:
    """True for sessions whose gap sits more than `n_sd` trailing SDs off the trailing
    mean of the last `window` gaps.

    gap_bias reads the overnight gap, but a futures contract roll moves the quoted price
    by the calendar spread. If the CSV is spliced rather than back-adjusted that jump is
    an artefact of the data, not an overnight move, so the setup it creates is fake. The
    trailing window ends on the PRIOR session, so a roll never inflates its own yardstick.
    """
    prior = gaps.shift(1)
    mu = prior.rolling(window, min_periods=window // 2).mean()
    sd = prior.rolling(window, min_periods=window // 2).std()
    return ((gaps - mu).abs() > n_sd * sd).fillna(False)


def report_roll_days(roll_days: dict, cfg: Config, max_show: int = 12) -> None:
    """Print the sessions flagged as contract rolls and whether they were dropped."""
    total = sum(len(v) for v in roll_days.values())
    action = "EXCLUDED" if cfg.exclude_roll_days else "kept (cfg.exclude_roll_days is False)"
    print("\n=== Roll-day check (gap > 4 trailing SDs of the last 60 sessions) ===")
    print(f"{total} suspect sessions flagged -- {action}")
    for sym, dates in roll_days.items():
        if not len(dates):
            print(f"  {sym}: none")
            continue
        shown = ", ".join(str(d) for d in list(dates)[:max_show])
        more = f"  (+{len(dates) - max_show} more)" if len(dates) > max_show else ""
        print(f"  {sym}: {len(dates)} -- {shown}{more}")


def backtest(cfg: Config, verbose: bool = True) -> pd.DataFrame:
    trades = []
    roll_days = {}
    for future in SYMBOLS:
        bars = fetch_csv(future, cfg).between_time(cfg.open_start, cfg.session_close)
        sessions = dict(tuple(bars.groupby(bars.index.date)))
        gaps = session_gaps(sessions)
        flagged = gaps.index[flag_roll_days(gaps)]
        roll_days[future] = flagged
        skip = set(flagged) if cfg.exclude_roll_days else set()
        tick = TICK_SIZES.get(future, 0.25)

        prev_close = None
        for d, day in sessions.items():
            bias = 0
            if cfg.gap_bias and prev_close is not None:
                gap = day["open"].iloc[0] - prev_close          # RTH open vs prior RTH close
                bias = -1 if gap > 0 else 1 if gap < 0 else 0   # gap up -> shorts, down -> longs
            if d in skip:                                       # suspected contract roll
                prev_close = day["close"].iloc[-1]
                continue
            trade = find_trade(day, cfg, bias, tick)
            if trade:
                trades.append({"future": future, **trade})
            prev_close = day["close"].iloc[-1]

    if verbose:
        report_roll_days(roll_days, cfg)
    return pd.DataFrame(trades)


def report(trades: pd.DataFrame) -> None:
    if trades.empty:
        print("No trades found in the selected period.")
        return

    wins = trades["R"] > 0
    print("\n=== Summary ===")
    print(f"Trades       : {len(trades)}")
    print(f"Win rate     : {wins.mean():.1%}")
    print(f"Total R      : {trades['R'].sum():+.2f}")
    print(f"Avg R / trade: {trades['R'].mean():+.2f}")
    print(f"Total points : {trades['pnl'].sum():+.2f}")
    print("\nBy market:")
    print(trades.groupby("future")["R"].agg(["count", "sum", "mean"]).round(2))


# --------------------------------------------------------------------------- #
# Analytics
# --------------------------------------------------------------------------- #
def _ann_sharpe(returns: pd.Series) -> float:
    """Annualised Sharpe of a time-indexed per-trade R series (0 risk-free).

    Annualises by sqrt(trades per year), which treats every trade as an
    INDEPENDENT bet. When ES and NQ both trade the same morning off the same gap
    that assumption is false, so this figure is optimistic -- see _daily_sharpe.
    """
    sd = returns.std()
    if not sd:
        return float("nan")
    years = max((returns.index[-1] - returns.index[0]).days / 365.25, 1e-9)
    per_year = len(returns) / years                           # trades per year
    return returns.mean() / sd * per_year ** 0.5


def _daily_sharpe(returns: pd.Series) -> float:
    """Correlation-aware annualised Sharpe. `returns` is a time-indexed per-trade
    series; same-day legs (e.g. ES + NQ off one gap) are SUMMED into a daily P&L
    first, so their correlation is baked into the volatility, then annualised by
    sqrt(active days per year). This is the honest headline: it does not assume
    two 0.9-correlated sleeves are two independent bets."""
    daily = returns.groupby(returns.index.date).sum().sort_index()
    sd = daily.std()
    if not sd or len(daily) < 2:
        return float("nan")
    span_days = (pd.Timestamp(daily.index[-1]) - pd.Timestamp(daily.index[0])).days
    per_year = len(daily) / max(span_days / 365.25, 1e-9)     # active days per year
    return daily.mean() / sd * per_year ** 0.5


def _metrics(returns: pd.Series) -> dict:
    """Risk / return stats from a time-indexed series of per-trade R results."""
    equity = returns.cumsum()
    drawdown = equity.cummax() - equity                       # R below the peak
    years = max((returns.index[-1] - returns.index[0]).days / 365.25, 1e-9)
    per_year = len(returns) / years
    sd = returns.std()
    gross_win = returns[returns > 0].sum()
    gross_loss = -returns[returns < 0].sum()
    return {
        "Trades": len(returns),
        "Total R": round(equity.iloc[-1], 2),
        "Win %": round((returns > 0).mean() * 100, 1),
        "Expectancy R": round(returns.mean(), 3),
        "Max DD R": round(drawdown.max(), 2),
        "Vol R/trade": round(sd, 3),
        "Ann vol R": round(sd * per_year ** 0.5, 2),
        "Sharpe/day": round(_daily_sharpe(returns), 2),      # headline, correlation-aware
        "Sharpe/trade": round(_ann_sharpe(returns), 2),      # optimistic, not the headline
        "Profit factor": round(gross_win / gross_loss, 2) if gross_loss else float("inf"),
    }


def performance(trades: pd.DataFrame, cfg: Config) -> None:
    """Print per-asset and combined risk stats, plus the active risk settings."""
    curves = {sym: g.set_index("entry_time").sort_index()["R"]
              for sym, g in trades.groupby("future")}
    curves["COMBINED"] = trades.set_index("entry_time").sort_index()["R"]
    table = pd.DataFrame({name: _metrics(r) for name, r in curves.items()}).T

    print("\n=== Performance / risk ===")
    print(table.to_string())
    print("\n=== Risk parameters ===")
    print(f"Timeframe    : {cfg.timeframe}")
    print(f"Reward:risk  : {cfg.reward_risk:g} : 1   (risk = 1.00 R per trade)")
    print(f"Stop         : most recent swing, {cfg.swing_lookback}-bar pivot")
    print(f"Entry window : {cfg.open_start:%H:%M}-{cfg.entry_by:%H:%M} ET")
    print(f"Fill rule    : entry and target limits must trade through by "
          f"{cfg.fill_through_ticks:g} tick(s)")
    print(f"Slippage on  : {', '.join(cfg.slip_on_exits)} exits only "
          f"({cfg.slippage_ticks:g} tick round turn)")
    print(f"Limit offset : {cfg.entry_offset_ticks:g} tick(s) towards the market")
    print("Sharpe/day is the headline (same-day ES+NQ legs summed first, so their "
          "correlation is priced in); Sharpe/trade assumes independent trades and flatters.")


# Per market: (point value $ per 1.00 of price, round-turn commission $, one-tick value $).
# Dollar risk is |entry - stop| * point value. It does NOT reference the price level, so
# it stays correct in 2016 and in 2026 -- the old notional-based version overstated the
# risk (and so understated cost as a share of R) in the early years by up to 3x.
COST_SPECS = {
    "ES":  (50.0, 4.0, 12.50),
    "NQ":  (20.0, 4.0,  5.00),
    "YM":  (5.0,  4.0,  5.00),
    "RTY": (50.0, 4.0,  5.00),
}

# Minimum price increment per market -- the unit cfg.fill_through_ticks is measured in.
TICK_SIZES = {"ES": 0.25, "NQ": 0.25, "YM": 1.0, "RTY": 0.10}


def dollar_risk(trades: pd.DataFrame, specs=None) -> pd.Series:
    """Cash at risk on ONE contract: |entry - stop| * the market's point value."""
    specs = specs or COST_SPECS
    point_value = trades["future"].map(lambda s: specs[s][0])
    return (trades["entry"] - trades["stop"]).abs() * point_value


def round_turn_cost(trades: pd.DataFrame, cfg: Config, specs=None) -> pd.Series:
    """Round-turn cost of ONE contract: commission always, slippage only where the order
    crossed the spread.

    A stop becomes a market order the moment it triggers, and a run to the session close
    is a market order outright, so both pay slippage -- those are cfg.slip_on_exits. A
    take-profit is a resting limit: it fills at its price or not at all, so it pays none.
    What a passive exit pays instead is fill UNCERTAINTY, which _simulate charges through
    cfg.fill_through_ticks. Charging both would be taking the cost of crossing the spread
    on a trade that never crossed it, and the old flat charge did exactly that on the
    ~54% of trades that exited at target. Slippage is a round-turn total, so the limit
    entry is already assumed to fill at its price.
    """
    specs = specs or COST_SPECS
    commission = trades["future"].map(lambda s: specs[s][1])
    tick_value = trades["future"].map(lambda s: specs[s][2])
    crosses = trades["reason"].isin(cfg.slip_on_exits).astype(float)
    return commission + tick_value * cfg.slippage_ticks * crosses


def size_positions(trades: pd.DataFrame, cfg: Config, specs=None) -> pd.DataFrame:
    """Size every trade in WHOLE contracts off the running account balance.

    Taken in time order: the account is willing to risk balance * cfg.risk_per_trade
    (a flat stake off the starting balance when cfg.compound_risk is off), one contract
    risks |entry - stop| * point value, so contracts = floor(budget / risk per contract).
    Futures do not come in fractions, so a trade the budget cannot cover even once is
    UNTRADEABLE: it is skipped, leaves the balance alone, and is counted so the report
    can say how much of the sample the account was simply too small to take.

    Cash P&L is contracts * direction * (exit - entry) * point value minus
    contracts * (commission + slippage). Cost as a fraction of R is measured against the
    risk of the position actually held, contracts * risk per contract -- the same
    denominator the account compounds on, which is what the old one-contract version
    got wrong. Returns the trades in time order with sizing, costs and balance attached.
    """
    specs = specs or COST_SPECS
    sized = trades.sort_values("entry_time").copy()
    sized["point_value"] = sized["future"].map(lambda s: specs[s][0])
    sized["risk_per_contract"] = dollar_risk(sized, specs)
    sized["cost_per_contract"] = round_turn_cost(sized, cfg, specs)

    flat_stake = cfg.account_start * cfg.risk_per_trade
    balance = cfg.account_start
    contracts, cost_d, cost_r, pnl_d, net_r, balances = [], [], [], [], [], []
    for _, t in sized.iterrows():
        budget = balance * cfg.risk_per_trade if cfg.compound_risk else flat_stake
        risk1 = t["risk_per_contract"]
        n = int(budget // risk1) if risk1 > 0 else 0        # whole contracts only
        cost = n * t["cost_per_contract"]
        pnl = n * t["pnl"] * t["point_value"] - cost        # direction is already in pnl
        balance += pnl                                      # untradeable => n = 0 => no change
        contracts.append(n)
        cost_d.append(cost)
        cost_r.append(cost / (n * risk1) if n else float("nan"))
        pnl_d.append(pnl)
        net_r.append(t["R"] - cost / (n * risk1) if n else float("nan"))
        balances.append(balance)

    sized["contracts"] = contracts
    sized["cost_$"] = cost_d
    sized["cost_R"] = cost_r
    sized["pnl_$"] = pnl_d
    sized["net_R"] = net_r
    sized["balance"] = balances
    return sized


def cost_per_trade_r(trades: pd.DataFrame, cfg: Config, specs=None) -> pd.Series:
    """All-in round-turn cost as a fraction of R, on the position size the account
    actually carried. Untradeable trades have no position, so they are dropped -- see
    size_positions() for the sizing rules."""
    sized = size_positions(trades, cfg, specs)
    return sized.loc[sized["contracts"] > 0, "cost_R"]


def net_returns(trades: pd.DataFrame, cfg: Config, specs=None) -> pd.Series:
    """After-cost per-trade R for the trades the account could actually take:
    gross R minus the sized position's round-turn cost."""
    sized = size_positions(trades, cfg, specs)
    return sized.loc[sized["contracts"] > 0, "net_R"]


def _net_stats(sized: pd.DataFrame) -> dict:
    """Headline after-cost figures for a block of already-sized trades. `sized` is the
    tradeable rows of size_positions(), indexed by entry_time."""
    net = sized["net_R"]
    if net.empty:
        return {"n": 0, "exp": float("nan"), "total": float("nan"),
                "win": float("nan"), "sharpe": float("nan")}
    return {"n": len(net), "exp": net.mean(), "total": net.sum(),
            "win": (net > 0).mean() * 100, "sharpe": _daily_sharpe(net)}


def _tradeable(trades: pd.DataFrame, cfg: Config, specs=None) -> tuple:
    """Sized trades split into (every sized row, the tradeable ones indexed by
    entry_time, how many were skipped because the account could not afford a contract)."""
    sized = size_positions(trades, cfg, specs)
    taken = sized[sized["contracts"] > 0].set_index("entry_time").sort_index()
    return sized, taken, int((sized["contracts"] == 0).sum())


def after_costs(trades: pd.DataFrame, cfg: Config, specs=None) -> None:
    """Re-score the edge after realistic per-trade futures costs (commission plus
    cfg.slippage_ticks of round-turn slippage) on the integer-contract position the
    account can actually carry. See size_positions() for the sizing rules and
    cost_per_trade_r() for the dollar-cost -> fraction-of-R conversion."""
    sized, taken, skipped = _tradeable(trades, cfg, specs)
    stop_pct = (taken["entry"] - taken["stop"]).abs() / taken["entry"]

    def stats(r):
        sd = r.std()
        t = r.mean() / (sd / len(r) ** 0.5) if sd else float("nan")
        return r.mean(), r.sum(), (r > 0).mean() * 100, t, _ann_sharpe(r), _daily_sharpe(r)

    print(f"\n=== After-cost edge (commission + {cfg.slippage_ticks:g}-tick round-turn slippage) ===")
    print(f"{'':6}{'expectancy':>12}{'total R':>10}{'win %':>8}{'t-stat':>9}"
          f"{'Shp/trade':>11}{'Shp/day':>10}")
    for label, r in (("gross", taken["R"]), ("net", taken["net_R"])):
        m, tot, win, t, shp_tr, shp_day = stats(r)
        print(f"{label:6}{m:>+12.3f}{tot:>10.1f}{win:>8.1f}{t:>9.2f}{shp_tr:>11.2f}{shp_day:>10.2f}")
    print(f"HEADLINE Sharpe (daily, correlation-aware, net of costs): "
          f"{_daily_sharpe(taken['net_R']):.2f}")
    print(f"Cost {taken['cost_R'].mean():.3f} R/trade avg (median {taken['cost_R'].median():.3f}) | "
          f"avg stop {stop_pct.mean() * 100:.3f}% of price | "
          f"break-even at {taken['R'].mean():.3f} R")
    print(f"Position size : median {taken['contracts'].median():.0f} contracts "
          f"(min {taken['contracts'].min():.0f}, max {taken['contracts'].max():.0f})")
    print(f"Untradeable   : {skipped} of {len(sized)} trades ({skipped / len(sized):.1%}) skipped -- "
          f"the risk budget could not cover one contract")
    print("Shp/trade annualises by sqrt(trades) (assumes each trade independent) and is NOT the "
          "headline; Shp/day sums same-day ES+NQ legs first, so their correlation is priced in.")


def _is_mask(taken: pd.DataFrame, cfg: Config):
    """Boolean mask picking the in-sample rows out of an entry_time-indexed frame."""
    days = pd.Index([t.date() for t in taken.index])
    lo, hi = pd.Timestamp(cfg.is_start).date(), pd.Timestamp(cfg.is_end).date()
    return (days >= lo) & (days <= hi)


def in_out_sample(trades: pd.DataFrame, cfg: Config, specs=None) -> None:
    """Split the after-cost results into the parameter-selection window and everything else.

    Only cfg.is_start..cfg.is_end was ever used to choose the settings; every other session
    in the file is untouched out-of-sample. If the edge is real the two blocks look alike.
    An in-sample figure towering over the out-of-sample one is the signature of a fitted
    parameter set, and the out-of-sample column is the one to believe.
    """
    _, taken, _ = _tradeable(trades, cfg, specs)
    is_mask = _is_mask(taken, cfg)

    print(f"\n=== In-sample vs out-of-sample (IS = {cfg.is_start} to {cfg.is_end}) ===")
    print(f"{'window':>16}{'trades':>9}{'net exp R':>12}{'win %':>8}"
          f"{'net total R':>13}{'Sharpe/day':>12}")
    for label, block in (("in-sample", taken[is_mask]), ("out-of-sample", taken[~is_mask])):
        m = _net_stats(block)
        print(f"{label:>16}{m['n']:>9}{m['exp']:>+12.3f}{m['win']:>8.1f}"
              f"{m['total']:>+13.1f}{m['sharpe']:>12.2f}")
    print("Sharpe/day is the correlation-aware headline figure. The in-sample block is about "
          "six months, so its Sharpe carries a wide error bar -- read it for sign and rough "
          "size, not for precision.")


def fill_sensitivity(cfg: Config, levels=(0, 1, 2), specs=None) -> None:
    """Re-run the WHOLE backtest at each cfg.fill_through_ticks setting.

    0 ticks is the old touch-is-a-fill assumption; 1 and 2 make price trade that far past
    the limit before the order counts as done. The trades that disappear as the requirement
    tightens are the touch-and-reverse cases, which are disproportionately winners, so the
    drop from row to row is the share of the edge that rests on the fill assumption rather
    than on the strategy.
    """
    print("\n=== Fill-through sensitivity (whole backtest re-run at each level) ===")
    print(f"{'ticks':>6}{'trades':>9}{'untradeable':>13}{'net exp R':>12}"
          f"{'net total R':>13}{'win %':>8}{'Sharpe/day':>12}")
    for level in levels:
        run_cfg = replace(cfg, fill_through_ticks=float(level))
        t = backtest(run_cfg, verbose=False)
        if t.empty:
            print(f"{level:>6}{0:>9}{'-':>13}{'-':>12}{'-':>13}{'-':>8}{'-':>12}")
            continue
        _, taken, skipped = _tradeable(t, run_cfg, specs)
        m = _net_stats(taken)
        print(f"{level:>6}{m['n']:>9}{skipped:>13}{m['exp']:>+12.3f}"
              f"{m['total']:>+13.1f}{m['win']:>8.1f}{m['sharpe']:>12.2f}")
    print("Sharpe/day is the correlation-aware headline figure.")


def calibrate_entry_offset(cfg: Config, offsets=(0, 1, 2, 3, 4), fill_levels=(0, 1, 2),
                           slips=(0, 1, 2), flat_sizing: bool = True, specs=None) -> None:
    """Grid the limit offset against the fill and slippage assumptions, CALIBRATING ON
    THE IN-SAMPLE WINDOW ONLY and printing the same grid out-of-sample beside it.

    The offset is a real order-placement choice, so it costs a degree of freedom and has
    to be picked the way every other parameter was -- on cfg.is_start..cfg.is_end, with
    the out-of-sample block kept back as confirmation rather than as part of the search.
    Read the IS grid for a PLATEAU, not a peak: a smooth trade-off (fill rate up, edge per
    trade down) is a real effect, whereas one cell towering over its neighbours is noise.
    The offset exists to buy queue position, so its falsifiable prediction is that it
    FLATTENS the fill-through columns -- if it only helps at fill_through 0, it is an
    artefact of the touch-is-a-fill assumption rather than a fix for it.

    Each (offset, fill) pair needs its own backtest because both change which trades
    exist; slippage only re-prices them, so it is swept for free inside each run.

    flat_sizing forces a constant risk budget across the whole grid. Under compounding a
    weak cell shrinks the account, a smaller account makes more trades unaffordable, and
    the statistics then get computed on a different (and quietly self-selected) subset --
    so cells stop being comparable exactly where they matter most. A fixed stake breaks
    that feedback loop. Calibrate on this grid, then read the account curve separately.
    """
    if flat_sizing:
        cfg = replace(cfg, compound_risk=False)
    runs = {}
    for off in offsets:
        for ft in fill_levels:
            run_cfg = replace(cfg, entry_offset_ticks=float(off), fill_through_ticks=float(ft))
            runs[(off, ft)] = (run_cfg, backtest(run_cfg, verbose=False))

    header = (f"{'off':>4}{'fill':>6}{'trades':>8}"
              + "".join(f"{'exp s' + str(sl):>10}{'Shp s' + str(sl):>9}" for sl in slips))
    for window in ("IN-SAMPLE", "OUT-OF-SAMPLE"):
        print(f"\n=== Entry-offset calibration -- {window} "
              f"({cfg.is_start} to {cfg.is_end}) ===")
        print(header)
        for off in offsets:
            for ft in fill_levels:
                run_cfg, t = runs[(off, ft)]
                # Fills in this window, before any affordability test -- unlike the sized
                # count it does not move with the slippage level, so the column compares.
                fills = t.set_index("entry_time").sort_index()
                in_win = _is_mask(fills, cfg)
                n = int((in_win if window == "IN-SAMPLE" else ~in_win).sum())
                cells = []
                for sl in slips:
                    _, taken, _ = _tradeable(t, replace(run_cfg, slippage_ticks=float(sl)), specs)
                    mask = _is_mask(taken, cfg)
                    block = taken[mask] if window == "IN-SAMPLE" else taken[~mask]
                    stat = _net_stats(block)
                    cells.append(f"{stat['exp']:>+10.3f}{stat['sharpe']:>9.2f}")
                print(f"{off:>4}{ft:>6}{n:>8}" + "".join(cells))
    print("exp = net expectancy in R, Shp = Sharpe/day (correlation-aware headline); "
          "s0/s1/s2 = round-turn slippage ticks; trades = fills in the window.")
    if flat_sizing:
        print("Sizing held FLAT across the grid so a weak cell cannot shrink its own sample.")
    print("The in-sample block is about six months, so single cells carry wide error bars "
          "-- judge the SHAPE of the grid, and treat the out-of-sample grid as the test, "
          "not as more search space.")


def market_cost_grid(cfg: Config, fill_levels=(0, 1, 2), slips=(0, 1, 2),
                     window: str = "OOS", flat_sizing: bool = True, specs=None) -> None:
    """ES against NQ across the fill and slippage assumptions, side by side.

    The two sleeves are not equally exposed to costs and the combined figures hide it.
    A fixed dollar cost is charged against each trade's own risk, so what matters is the
    tick value relative to the typical stop -- and ES pairs a $12.50 tick with stops about
    half NQ's dollar size, while NQ pairs a $5.00 tick with much wider ones. The same
    assumption therefore bites several times harder on ES.

    `window` is OOS (default), IS or FULL. Per-market Sharpe/day is that sleeve ALONE, so
    it carries none of the diversification in the combined number -- one market rarely
    trades twice in a day, which makes it a per-trade figure annualised by active days.
    Sizing is held flat by default so a weak cell cannot shrink its own sample.
    """
    if flat_sizing:
        cfg = replace(cfg, compound_risk=False)
    markets = sorted(SYMBOLS)
    head = f"{'fill':>5}{'slip':>5}"
    for sym in markets:
        head += f"{'| ' + sym + ' n':>10}{'exp':>9}{'Shp':>7}{'cost R':>9}"
    print(f"\n=== {' vs '.join(markets)} across fill and slippage "
          f"({window}, offset {cfg.entry_offset_ticks:g} ticks) ===")
    print(head)

    for ft in fill_levels:
        run_cfg = replace(cfg, fill_through_ticks=float(ft))
        trades = backtest(run_cfg, verbose=False)
        for sl in slips:
            _, taken, _ = _tradeable(trades, replace(run_cfg, slippage_ticks=float(sl)), specs)
            mask = _is_mask(taken, cfg)
            block = taken if window == "FULL" else (taken[mask] if window == "IS" else taken[~mask])
            row = f"{ft:>5}{sl:>5}"
            for sym in markets:
                leg = block[block["future"] == sym]
                stat = _net_stats(leg)
                cost = leg["cost_R"].mean() if len(leg) else float("nan")
                row += (f"{stat['n']:>10}{stat['exp']:>+9.3f}"
                        f"{stat['sharpe']:>7.2f}{cost:>9.4f}")
            print(row)
    print("exp = net expectancy in R, Shp = Sharpe/day for that sleeve alone, "
          "cost R = mean round-turn cost as a share of R.")


def slippage_sensitivity(trades: pd.DataFrame, cfg: Config, levels=(1, 2, 3), specs=None) -> None:
    """Re-cost the same trades at 1, 2 and 3 ticks of round-turn slippage.

    A stop order in fast futures does not get the tick it was resting on. Slippage changes
    what the trades cost, not WHICH ones happen (the entry is a limit), so the one trade
    list is simply re-priced at each level -- no re-run needed.
    """
    print("\n=== Slippage sensitivity (round-turn ticks) ===")
    print(f"{'ticks':>6}{'trades':>9}{'cost R':>9}{'net exp R':>12}"
          f"{'net total R':>13}{'win %':>8}{'Sharpe/day':>12}")
    for level in levels:
        run_cfg = replace(cfg, slippage_ticks=float(level))
        _, taken, _ = _tradeable(trades, run_cfg, specs)
        m = _net_stats(taken)
        print(f"{level:>6}{m['n']:>9}{taken['cost_R'].mean():>9.3f}{m['exp']:>+12.3f}"
              f"{m['total']:>+13.1f}{m['win']:>8.1f}{m['sharpe']:>12.2f}")
    print("Sharpe/day is the correlation-aware headline figure.")

def one_candle_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Trades that opened and closed inside a single bar -- the stop or target was
    hit on the very candle the limit filled (entry_time == exit_time). Returns the
    matching rows and prints a short summary."""
    same_bar = trades[trades["entry_time"] == trades["exit_time"]]

    print("\n=== One-candle trades (entered and exited on the same bar) ===")
    if same_bar.empty:
        print("None -- every trade spanned at least two bars.")
        return same_bar

    print(f"{len(same_bar)} of {len(trades)} trades ({len(same_bar) / len(trades):.1%}) "
          f"| win rate {(same_bar['R'] > 0).mean():.1%} | total {same_bar['R'].sum():+.2f} R")
    return same_bar


def simulate_account(trades: pd.DataFrame, cfg: Config, specs=None) -> pd.DataFrame:
    """Compound a real cash account over the whole timeline, risking a fixed % of
    the *current* balance on every trade.

    The sizing and the cash P&L both come from size_positions(), so the account curve
    and the cost figures share one denominator: the risk budget buys a WHOLE number of
    contracts, and each trade returns contracts * (exit - entry) * point value less
    commission and slippage. With cfg.compound_risk the budget is a fixed % of the LIVE
    balance, so it grows with the account (1% of £1m budgets £10k, 1% of £2m budgets
    £20k); with it off the budget stays flat at 1% of the starting balance. Trades the
    budget cannot cover even once are untradeable and are reported, not silently taken.
    Returns the trade-by-trade balance curve.
    """
    cur = cfg.currency
    mode = "compounding" if cfg.compound_risk else "flat stake"
    print(f"\n=== Account simulation ({mode}, after costs) ===")
    if trades.empty:
        print("No trades to simulate.")
        return pd.DataFrame()

    sized = size_positions(trades, cfg, specs)    # integer contracts off the live balance
    start = cfg.account_start
    flat_stake = start * cfg.risk_per_trade       # constant stake when not compounding
    balance = sized["balance"].iloc[-1]
    skipped = int((sized["contracts"] == 0).sum())
    taken = sized[sized["contracts"] > 0]

    curve = (sized.rename(columns={"entry_time": "time"})
                  .set_index("time")[["balance"]])
    running_peak = curve["balance"].cummax()
    max_dd_pct = ((running_peak - curve["balance"]) / running_peak).max()

    years = max((sized["entry_time"].iloc[-1] - sized["entry_time"].iloc[0]).days / 365.25, 1e-9)
    cagr = (balance / start) ** (1 / years) - 1

    print(f"Start balance   : {cur}{start:,.0f}")
    if cfg.compound_risk:
        print(f"Risk per trade  : {cfg.risk_per_trade:.1%} of live balance "
              f"({cur}{flat_stake:,.0f} at start -> {cur}{balance * cfg.risk_per_trade:,.0f} now)")
    else:
        print(f"Risk per trade  : flat {cur}{flat_stake:,.0f} "
              f"({cfg.risk_per_trade:.1%} of the {cur}{start:,.0f} start, never compounds)")
    print(f"Trades taken    : {len(taken)} of {len(sized)} "
          f"({skipped} untradeable -- budget short of one contract)")
    if len(taken):
        print(f"Contracts       : median {taken['contracts'].median():.0f} "
              f"(min {taken['contracts'].min():.0f}, max {taken['contracts'].max():.0f})")
    print(f"Final balance   : {cur}{balance:,.0f}")
    print(f"Total return    : {balance / start - 1:+.1%}")
    print(f"CAGR            : {cagr:+.1%}  over {years:.1f} years")
    print(f"Max drawdown    : -{max_dd_pct:.1%} from peak")

    year_end = curve["balance"].groupby(curve.index.year).last()
    print("\nYear-end balance:")
    for yr, bal in year_end.items():
        print(f"  {yr}: {cur}{bal:,.0f}")
    return curve


def plot_equity(trades: pd.DataFrame) -> None:
    """Display cumulative-R equity curves on a dark theme: one colour per market,
    with the combined total in white."""
    colors = {"ES": "#FF3B3B", "NQ": "#2E9BFF", "YM": "#3BFF6E", "RTY": "#FFB23B"}
    ordered = trades.sort_values("entry_time")

    fig, ax = plt.subplots(figsize=(11, 6), facecolor="black")
    ax.set_facecolor("black")

    def line(rows, color, width, label):
        equity = rows.set_index("entry_time")["R"].cumsum()
        equity.index = equity.index.tz_localize(None)
        ax.plot(equity.index, equity.values, color=color, lw=width,
                label=f"{label}  {equity.iloc[-1]:+.1f}R")

    for sym, grp in ordered.groupby("future"):
        line(grp, colors.get(sym, "white"), 1.6, sym)
    line(ordered, "white", 2.4, "Combined")

    ax.axhline(0, color="grey", lw=0.8, alpha=0.6)
    ax.set_title("FVG strategy — equity curves (cumulative R)", color="white", fontsize=13)
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Cumulative R", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("grey")
    ax.grid(True, color="grey", alpha=0.2)
    ax.legend(facecolor="black", edgecolor="grey", labelcolor="white", loc="upper left")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    cfg = Config()
    trades = backtest(cfg)
    report(trades)
    if not trades.empty:
        performance(trades, cfg)
        after_costs(trades, cfg)
        in_out_sample(trades, cfg)
        slippage_sensitivity(trades, cfg)
        one_candle_trades(trades)
        simulate_account(trades, cfg)
        fill_sensitivity(cfg)
        plot_equity(trades)
