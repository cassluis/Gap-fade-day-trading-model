# Gap-Fade Day-Trading Model

This project tests whether a popular discretionary "Smart Money Concepts" strategy has a measurable edge. The strategy fades the opening gap by trading the session's first fair-value gap.

It's backtested on ES and NQ futures on 5-minute bars: 3,997 trades over 10.8 years (Jan 2016 – Oct 2026).

**Finding: the edge doesn't survive realistic fill assumptions.** It depends on resting limit orders filling as soon as price touches them.

## Results

Out-of-sample daily Sharpe, net of costs:

| Fill-through and slippage | Sharpe | ES | NQ |
|---|---|---|---|
| 0 ticks | 1.65 | 1.45 | 1.11 |
| 1 tick | 0.36 | −0.12 | 0.68 |
| 2 ticks | −1.16 | −1.57 | −0.09 |

- **Fill-through alone:** with no slippage, it takes the Sharpe from 1.65 to −0.16 at 2 ticks.
- **In-sample vs out-of-sample:** gross daily Sharpe is 3.59 in-sample against 1.92 out-of-sample.

That collapse is the finding.
- **Why fill-on-touch is wrong:** a limit order doesn't fill because price touched the level. It fills when price trades *through* it.
- **Why it matters here:** the trades where price touches and reverses are exactly the winners.
- **The point of the exercise** was to test that assumption.

## Method

- **Signal:** the first fair-value gap in the 09:30–12:10 ET window.
  - entry on a limit order at the gap's near edge
  - stop at the most recent swing
  - target at 1:1 reward to risk
- **Fill rule:** the stop fills as soon as it's touched. Entry and target limits must trade `fill_through_ticks` beyond their price.
- **Stop and target on the same bar:** the stop wins.
- **Costs:** commission on every trade. Slippage only on exits that cross the spread (stops, and the market close).
- **Sizing:** 1% of the account at risk per trade, in whole contracts.
- **Sharpe:** daily and correlation-aware. Same-day ES and NQ trades are summed before annualising.
- **Fitting:** parameters were selected on Q1–Q2 2024 only. The rest of 2016–2026 is untouched out-of-sample: about 40 quarters against 2 in-sample.
- **Roll days:** sessions with abnormal opening gaps are excluded.

## Running

```bash
python fvg_strategy.py
```

The script reads 1-minute (or 5-minute) OHLC CSVs for ES and NQ from `Config.csv_dir`. The data isn't included. Settings such as the timeframe, fill-through, slippage and the date range are in `Config`.

## Attribution

The implementation is largely AI-assisted (Claude Code). The strategy specification, test design and error analysis are mine.
