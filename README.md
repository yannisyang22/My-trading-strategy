
# My Trading Strategy (BTC / Stocks Backtest)

This repo contains a low-frequency trend + range (grid) hybrid strategy framework.
Expected Sharpe > 1
- **Trend Module (MTF Scoring)**: EMA56 and MA116 in multiple timeframes (1D, 2D, 3D, 5D, W (weekly), ME (month-end)), weekly Bollinger middle band for main support/resistance
- **Range engine:** Use Bollinger “box” grid when market is moving sideways
- **Additional** MACD as momentum filter, Wyckoff bottom/top score, Fibonacci as aid in determining target, mild risk controls
- **Execution:** next-day execution (`pos.shift(1)`) + position step threshold to reduce churn

---

## Quick Start

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
- Console stats (CAGR / Sharpe / Max DD / turnover / trades)
- trade_reasons.csv for trade-by-trade diagnostics
- Equity / drawdown plots


