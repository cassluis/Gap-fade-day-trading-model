Gap-Fade Day Trading Model

Quantifies a discretionary "Smart Money Concepts" retail strategy — fading the opening gap via the first fair-value gap of the session — to test whether it has measurable edge. Backtested on ES and NQ futures on 5-minute bars, 3,997 trades over 10.8 years.

The edge does not survive realistic fill assumptions. Out-of-sample daily Sharpe decays from 1.65 to 0.36 to -1.16 at 0, 1 and 2 ticks of fill-through and slippage, which establishes the edge relies on the assumption that resting limits fill on touch. ES runs at -1.57 against NQ's -0.09 at the pessimistic assumption. Fill-through alone, with no slippage, takes it from 1.65 to -0.16 at 2 ticks.

That collapse is the finding. A limit order does not fill because price touched the level — it fills when price trades through it, and the trades where price touches and reverses are precisely the winners. Testing that assumption was the point of the exercise.

Parameters were selected on Q1–Q2 2024 only; 2016–2026 outside that window is untouched out-of-sample — roughly 40 out-of-sample quarters against 2 in-sample. Gross in-sample daily Sharpe 3.59 against 1.92 out-of-sample.

Implementation largely AI-assisted (Claude Code). Strategy specification, test design, and error analysis are mine.
