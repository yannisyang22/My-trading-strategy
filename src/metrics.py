import numpy as np
import pandas as pd

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())

def sharpe(daily_ret: pd.Series) -> float:
    x = daily_ret.dropna()
    if x.std() == 0 or len(x) < 10:
        return 0.0
    return float(np.sqrt(365) * x.mean() / x.std())

def cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    start = equity.index[0]
    end = equity.index[-1]
    years = (end - start).days / 365.0
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1)

def summary_stats(bt: pd.DataFrame) -> dict:
    eq = bt["equity"]
    r = bt["ret"]
    return {
        "final_equity": float(eq.iloc[-1]),
        "cagr": cagr(eq),
        "max_drawdown": max_drawdown(eq),
        "sharpe": sharpe(r),
        "avg_turnover": float(bt["turnover"].mean()),
        "num_trades": int(bt["trade"].sum()),
        "win_rate": float((r[r != 0] > 0).mean()) if (r != 0).any() else 0.0,
    }