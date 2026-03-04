
# My Trading Strategy (BTC / Stocks Backtest)

This repo contains a low-frequency trend + range (grid) hybrid strategy framework.
Expected Sharpe > 1
- **Trend Module (MTF Scoring)**
  Timeframes:
  - 1D, 2D, 3D, 5D, W (weekly), ME (month-end)
  Per timeframe indicators:
  - Trend alignment: close above/below EMA56 and MA116
  - BB mid filter: close above/below Bollinger mid
  - Momentum filter: MACD
- **Range engine:** When the market is in a period of fluctuation, use Bollinger “box” grid 
- **Additional** Wyckoff bottom/top, Fibonacci, mild risk controls
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


