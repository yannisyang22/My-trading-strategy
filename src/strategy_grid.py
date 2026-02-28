import numpy as np
import pandas as pd
from .indicators import bollinger


def build_grid_position(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    c = cfg.get("indicators", {})
    g = cfg.get("grid", {})
    filt = cfg.get("filters", {})

    w_freq = filt.get("weekly_resample", "W-SUN")
    bb_n = int(g.get("bb_n", c.get("bb_n", 20)))
    bb_k = float(g.get("bb_k", c.get("bb_k", 2.0)))

    w_close = out["close"].resample(w_freq).last()
    w_mid, w_up, w_dn, w_bw = bollinger(w_close, bb_n, bb_k)

    # FIX: shift(1) to avoid using current week's final value during the week
    mid = w_mid.shift(1).reindex(out.index, method="ffill")
    up  = w_up.shift(1).reindex(out.index, method="ffill")
    dn  = w_dn.shift(1).reindex(out.index, method="ffill")
    bw  = w_bw.shift(1).reindex(out.index, method="ffill")

    out["g_bb_mid"], out["g_bb_up"], out["g_bb_dn"], out["g_bb_bw"] = mid, up, dn, bw

    denom = (up - mid).replace(0, np.nan)
    z = ((out["close"] - mid) / denom).clip(-2.0, 2.0)
    out["grid_z"] = z

    L_base = float(g.get("leverage", 0.60))
    bw_ref_window = int(g.get("bw_ref_window", 252))
    bw_ref = out["g_bb_bw"].rolling(bw_ref_window).median()

    scale_lo = float(g.get("lev_scale_lo", 0.60))
    scale_hi = float(g.get("lev_scale_hi", 1.20))
    scale = (bw_ref / out["g_bb_bw"].replace(0, np.nan)).clip(lower=scale_lo, upper=scale_hi).fillna(1.0)
    L = (L_base * scale).fillna(L_base)

    kappa = float(g.get("kappa", 1.20))
    pos = (-np.tanh(kappa * z) * L).astype(float)

    dead = float(g.get("deadzone", 0.40))
    pos = pd.Series(pos, index=out.index).where(z.abs() > dead, 0.0)

    margin = float(g.get("breakout_margin", 0.006))
    above = out["close"] > (up * (1.0 + margin))
    below = out["close"] < (dn * (1.0 - margin))
    br = (above | below).fillna(False)

    decay = float(g.get("breakout_decay", 0.35))
    if decay > 0:
        pos = pos.where(~br, pos * (1.0 - decay))
    else:
        pos = pos.where(~br, 0.0)

    smooth = int(g.get("smooth_days", 7))
    if smooth > 1:
        pos = pos.rolling(smooth).mean()

    out["pos_grid_raw"] = pos.fillna(0.0)
    return out