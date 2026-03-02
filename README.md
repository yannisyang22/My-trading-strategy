
# My Trading Strategy (BTC / Stocks Backtest)

This repo contains a low-frequency trend + range (grid) hybrid strategy framework.

- **Trend engine:** multi-timeframe synthetic bars scoring (1D / 2D / 3D / 5D / Weekly / Monthly)
- **Range engine:** weekly Bollinger “box” pseudo-grid (continuous position via tanh)
- **Overlays:** Wyckoff bottom/top (aux), Fibonacci tp1/tp2 (aux), mild risk controls
- **Execution:** next-day execution (`pos.shift(1)`) + position step threshold to reduce churn

---

## 1) Quick Start

### Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run backtest
```bash
python run_backtest.py
```
Outputs:
	•	Console stats (CAGR / Sharpe / Max DD / turnover / trades)
	•	trade_reasons.csv for trade-by-trade diagnostics
	•	Equity / drawdown plots

---

## 2) Strategy Overview

A) Trend Module (MTF Scoring)

Timeframes:
	•	1D, 2D, 3D, 5D, W (weekly), ME (month-end)

Per timeframe indicators:
	•	Trend alignment: close above/below EMA56 and MA116
	•	BB mid filter: close above/below Bollinger mid
	•	Momentum filter: MACD histogram > 0 (long) / < 0 (short)

Scoring:
	•	Each timeframe contributes up to 2 * weight (trend + momentum).
	•	Weighted sum produces:
	•	long_score, short_score
	•	A gap constraint forces “directional dominance”:
	•	gap = long_score_adj - short_score_adj

Entry conditions (after confirmation days):
	•	Long: long_score_adj >= long_th AND gap >= gap_th AND not banned by Wyckoff top
	•	Short: short_score_adj >= short_th AND -gap >= gap_th AND not banned by Wyckoff bottom

B) Wyckoff Overlay
	•	wyckoff.py outputs bottom_score / top_score (pattern + score smoothing)
	•	Used only as small bonus/penalty, so it helps timing without breaking baseline behavior:
	•	Bottom score can slightly boost long score / ban shorts
	•	Top score can ban longs / slightly boost shorts

C) Fibonacci Overlay 
	•	fib.py produces tp signals:
	•	tp1_long/tp2_long reduce long exposure to 0.6 / 0.3
	•	tp1_short/tp2_short reduce short exposure to -0.6 / -0.3
	•	Purpose: lock profits and reduce tail drawdowns.

---

## 3) Range Regime + Grid Module

Range regime detection (grid_mode)

Daily conditions:
	•	Bollinger bandwidth is low vs rolling quantile
	•	MA116 slope is small
	•	Confirmed over confirm_days to reduce false triggers

When grid_mode=True:
	•	Use weekly Bollinger “box” grid position pos_grid_raw instead of trend position.

Weekly-box pseudo grid
	•	Build weekly Bollinger (mid/up/dn/bw) with shift(1) to avoid look-ahead.
	•	Normalize distance from mid: z
	•	Continuous position: pos = -tanh(kappa*z) * leverage_scale
	•	deadzone suppresses trades near mid
	•	breakout_decay reduces position when price breaks out of the box
	•	smooth_days smooths position series

Directional restriction (in range mode):
	•	Above weekly mid → grid biased to long only
	•	Below weekly mid → grid biased to short only

---

## 4) Risk Controls (Mild / Tail-focused)
	•	High-volatility soft cap using rolling return volatility quantile
	•	Bear-trend long cap (max_long_in_bear) to reduce bear market tail risk
	•	ATR trailing “soft stop” (two-tier de-risk, no forced flat)
	•	Vol targeting (downscale only, only when position is large)
	•	Optional left-side short: bear-trend retest near weekly mid; reduce on reclaim

--

## 5) Execution / Backtest Assumptions
	•	Next-day execution: pos_exec = pos_daily_raw.shift(1)
	•	Returns: close-to-close (works well for BTC; for stocks use trading-day resampling)
	•	Costs: fee_bps + slippage_bps
	•	Position step threshold: skip small rebalances (but always allow risk reductions)

---

## 6) Config

Main params are in config.yaml (or config.example.yaml if you keep local config ignored).

Key tuning knobs:
	•	mtf_score.long_th / short_th / gap_th / exit_gap
	•	mtf_score.weights (W/ME typically higher)
	•	regime.confirm_days (grid activation stability)
	•	grid.deadzone / smooth_days / leverage / grid_leverage_cap
	•	risk.max_long_in_bear (bear tail risk)
	•	costs.pos_step_threshold (reduce churn)

---

## 7) Notes for US Stocks
	•	Use trading-day synthetic bars (2B/3B/5B) and weekly resample W-FRI.
	•	Prefer adjusted prices (auto_adjust=True) to handle splits/dividends.

---

## Disclaimer

This code is for research/education only. It is not financial advice.

