import numpy as np
import pandas as pd


def backtest(
    df: pd.DataFrame,
    position: pd.Series,
    fee_bps: float,
    slippage_bps: float,
    initial_equity: float = 1.0,
    max_drawdown_stop: float = 0.20,
    cooldown_days: int = 21,
    pos_step_threshold: float = 0.0,
):
    idx = df.index
    asset_ret = df["close"].pct_change().fillna(0.0).values

    target_pos = position.reindex(idx).fillna(0.0).values
    cost_rate = (fee_bps + slippage_bps) / 10000.0

    equity = np.zeros(len(idx))
    eff_pos = np.zeros(len(idx))
    turnover = np.zeros(len(idx))
    cost = np.zeros(len(idx))
    ret = np.zeros(len(idx))
    dd = np.zeros(len(idx))
    cb_active = np.zeros(len(idx), dtype=bool)
    trade = np.zeros(len(idx), dtype=bool)

    equity[0] = initial_equity
    peak = initial_equity
    cooldown = 0
    prev_pos = 0.0

    for i in range(len(idx)):
        if cooldown > 0:
            desired = 0.0
            cb_active[i] = True
            cooldown -= 1
        else:
            desired = float(target_pos[i])

        th = float(pos_step_threshold)
        reducing_risk = abs(desired) < abs(prev_pos)
        if th and (abs(desired - prev_pos) < th) and (not reducing_risk):
            p = prev_pos
            t = 0.0
        else:
            p = desired
            t = abs(p - prev_pos)

        c = t * cost_rate
        r = p * asset_ret[i] - c

        if i > 0:
            equity[i] = equity[i - 1] * (1.0 + r)

        peak = max(peak, equity[i])
        dd_i = equity[i] / peak - 1.0
        dd[i] = dd_i

        if cooldown == 0 and p != 0.0 and dd_i <= -max_drawdown_stop:
            cooldown = int(cooldown_days)
            peak = equity[i]

        eff_pos[i] = p
        turnover[i] = t
        cost[i] = c
        ret[i] = r
        trade[i] = t > 0.0
        prev_pos = p

    out = pd.DataFrame(index=idx)
    out["close"] = df["close"]
    out["asset_ret"] = df["close"].pct_change().fillna(0.0)
    out["pos"] = eff_pos
    out["turnover"] = turnover
    out["cost"] = cost
    out["ret"] = ret
    out["equity"] = equity
    out["drawdown"] = dd
    out["cb_active"] = cb_active
    out["trade"] = trade
    return out