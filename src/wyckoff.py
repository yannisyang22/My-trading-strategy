import numpy as np
import pandas as pd


def atr_percent(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(n).mean()
    return (atr / close).replace([np.inf, -np.inf], np.nan)


# ============================================================
# Wyckoff Bottom (Accumulation)
# outputs:
#   bottom_score, sc, ar, st, spring, sos
# ============================================================
def compute_wyckoff_bottom(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    wcfg = cfg.get("wyckoff", {})

    n_atr = int(wcfg.get("atr_n", 14))
    vol_ma_n = int(wcfg.get("vol_ma", 20))

    sc_ret_th = float(wcfg.get("sc_ret_th", 0.08))
    sc_atr_mult = float(wcfg.get("sc_atr_mult", 2.0))
    sc_vol_mult = float(wcfg.get("sc_vol_mult", 1.8))
    sc_range_mult = float(wcfg.get("sc_range_mult", 1.8))

    ar_lookahead = int(wcfg.get("ar_lookahead", 20))
    ar_rebound = float(wcfg.get("ar_rebound", 0.12))

    st_lookahead = int(wcfg.get("st_lookahead", 60))
    st_near_low = float(wcfg.get("st_near_low", 0.06))
    st_vol_frac = float(wcfg.get("st_vol_frac", 0.75))

    spring_window = int(wcfg.get("spring_window", 120))
    spring_break = float(wcfg.get("spring_break", 0.01))
    sos_break = float(wcfg.get("sos_break", 0.00))

    score_smooth = int(wcfg.get("score_smooth", 5))

    out = pd.DataFrame(index=df.index)
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]

    atrp = atr_percent(df, n_atr)
    vol_ma = vol.rolling(vol_ma_n).mean()
    day_ret = close.pct_change()
    day_range = (high - low).abs() / close

    out["atrp"] = atrp
    out["vol_ma"] = vol_ma
    out["ret"] = day_ret
    out["range_pct"] = day_range

    # Selling Climax (SC): large down day + volume spike + range spike
    sc_dn = (day_ret <= -sc_ret_th) | (day_ret <= -(sc_atr_mult * atrp))
    sc_vol = vol >= (sc_vol_mult * vol_ma)
    sc_rng = day_range >= (sc_range_mult * atrp)
    out["sc"] = (sc_dn & sc_vol & sc_rng).fillna(False)

    # Anchor most recent SC
    sc_idx = out.index[out["sc"]]
    sc_anchor = pd.Series(pd.NaT, index=out.index)
    last = pd.NaT
    sc_set = set(sc_idx)
    for i, dt in enumerate(out.index):
        if dt in sc_set:
            last = dt
        sc_anchor.iloc[i] = last
    out["sc_anchor"] = sc_anchor

    sc_low = pd.Series(np.nan, index=out.index)
    sc_vol_val = pd.Series(np.nan, index=out.index)
    for i, dt in enumerate(out.index):
        a = sc_anchor.iloc[i]
        if pd.isna(a):
            continue
        sc_low.iloc[i] = float(low.loc[a])
        sc_vol_val.iloc[i] = float(vol.loc[a])
    out["sc_low"] = sc_low
    out["sc_vol"] = sc_vol_val

    days_since_sc = (out.index.to_series() - out["sc_anchor"]).dt.days
    out["days_since_sc"] = days_since_sc

    rebound_from_sc = (close / sc_low) - 1.0
    out["rebound_from_sc"] = rebound_from_sc

    # AR: rebound enough from SC low within window
    out["ar"] = (
        (days_since_sc >= 1) &
        (days_since_sc <= ar_lookahead) &
        (rebound_from_sc >= ar_rebound)
    ).fillna(False)

    # Range low/high approximation
    range_low = sc_low.copy()
    range_high = pd.Series(np.nan, index=out.index)
    for i, dt in enumerate(out.index):
        a = sc_anchor.iloc[i]
        if pd.isna(a):
            continue
        window_end = a + pd.Timedelta(days=ar_lookahead)
        end2 = min(dt, window_end)
        seg = close.loc[a:end2]
        if len(seg) > 0:
            range_high.iloc[i] = float(seg.max())
    out["range_low"] = range_low
    out["range_high"] = range_high

    # ST: revisit near SC low later with weaker volume
    near_low = close <= (sc_low * (1.0 + st_near_low))
    weaker_vol = vol <= (st_vol_frac * sc_vol_val)
    out["st"] = (
        (days_since_sc > ar_lookahead) &
        (days_since_sc <= st_lookahead) &
        near_low &
        weaker_vol
    ).fillna(False)

    # Spring: poke below range low then close back above it
    rl = out["range_low"].copy()
    rl_fallback = low.rolling(spring_window).min()
    rl = rl.fillna(rl_fallback)
    out["range_low_eff"] = rl

    spring = (low < rl * (1.0 - spring_break)) & (close > rl)
    out["spring"] = spring.fillna(False)

    # SOS: break above range high (simple)
    rh = out["range_high"].copy()
    rh_fallback = high.rolling(spring_window).max()
    rh = rh.fillna(rh_fallback)
    out["range_high_eff"] = rh

    out["sos"] = (close > rh * (1.0 + sos_break)).fillna(False)

    # Bottom score
    score = (
        40 * out["sc"].astype(int) +
        20 * out["ar"].astype(int) +
        20 * out["st"].astype(int) +
        20 * out["spring"].astype(int)
    ).clip(0, 100)

    if score_smooth > 1:
        score = score.rolling(score_smooth).max()

    out["bottom_score"] = score.fillna(0.0)
    return out


# ============================================================
# Wyckoff Top (Distribution)
# outputs:
#   top_score, top_bc, top_ar, top_st, top_utad
# ============================================================
def compute_wyckoff_top(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    tcfg = cfg.get("wyckoff_top", {})

    n_atr = int(tcfg.get("atr_n", 14))
    vol_ma_n = int(tcfg.get("vol_ma", 20))

    bc_ret_th = float(tcfg.get("bc_ret_th", 0.08))
    bc_atr_mult = float(tcfg.get("bc_atr_mult", 2.0))
    bc_vol_mult = float(tcfg.get("bc_vol_mult", 1.8))
    bc_range_mult = float(tcfg.get("bc_range_mult", 1.8))

    ar_lookahead = int(tcfg.get("ar_lookahead", 20))
    ar_drop = float(tcfg.get("ar_drop", 0.10))

    st_lookahead = int(tcfg.get("st_lookahead", 60))
    st_near_high = float(tcfg.get("st_near_high", 0.06))
    st_vol_frac = float(tcfg.get("st_vol_frac", 0.75))

    utad_window = int(tcfg.get("utad_window", 120))
    utad_break = float(tcfg.get("utad_break", 0.01))
    score_smooth = int(tcfg.get("score_smooth", 5))

    out = pd.DataFrame(index=df.index)
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]

    atrp = atr_percent(df, n_atr)
    vol_ma = vol.rolling(vol_ma_n).mean()
    day_ret = close.pct_change()
    day_range = (high - low).abs() / close

    out["atrp"] = atrp
    out["vol_ma"] = vol_ma
    out["ret"] = day_ret
    out["range_pct"] = day_range

    # Buying Climax (BC): big up + vol spike + range spike
    bc_up = (day_ret >= bc_ret_th) | (day_ret >= (bc_atr_mult * atrp))
    bc_vol = vol >= (bc_vol_mult * vol_ma)
    bc_rng = day_range >= (bc_range_mult * atrp)
    out["top_bc"] = (bc_up & bc_vol & bc_rng).fillna(False)

    # Anchor most recent BC
    bc_idx = out.index[out["top_bc"]]
    bc_anchor = pd.Series(pd.NaT, index=out.index)
    last = pd.NaT
    bc_set = set(bc_idx)
    for i, dt in enumerate(out.index):
        if dt in bc_set:
            last = dt
        bc_anchor.iloc[i] = last
    out["top_bc_anchor"] = bc_anchor

    bc_high = pd.Series(np.nan, index=out.index)
    bc_vol_val = pd.Series(np.nan, index=out.index)
    for i, dt in enumerate(out.index):
        a = bc_anchor.iloc[i]
        if pd.isna(a):
            continue
        bc_high.iloc[i] = float(high.loc[a])
        bc_vol_val.iloc[i] = float(vol.loc[a])
    out["top_bc_high"] = bc_high
    out["top_bc_vol"] = bc_vol_val

    days_since_bc = (out.index.to_series() - out["top_bc_anchor"]).dt.days
    out["top_days_since_bc"] = days_since_bc

    drop_from_bc = 1.0 - (close / bc_high)
    out["top_drop_from_bc_high"] = drop_from_bc

    # AR: drop enough from BC high within window
    out["top_ar"] = (
        (days_since_bc >= 1) &
        (days_since_bc <= ar_lookahead) &
        (drop_from_bc >= ar_drop)
    ).fillna(False)

    # Range approximation
    range_high = bc_high.copy()
    range_low = pd.Series(np.nan, index=out.index)
    for i, dt in enumerate(out.index):
        a = bc_anchor.iloc[i]
        if pd.isna(a):
            continue
        window_end = a + pd.Timedelta(days=ar_lookahead)
        end2 = min(dt, window_end)
        seg = close.loc[a:end2]
        if len(seg) > 0:
            range_low.iloc[i] = float(seg.min())
    out["top_range_high"] = range_high
    out["top_range_low"] = range_low

    # ST: revisit near BC high later with weaker vol
    near_high = close >= (bc_high * (1.0 - st_near_high))
    weaker_vol = vol <= (st_vol_frac * bc_vol_val)
    out["top_st"] = (
        (days_since_bc > ar_lookahead) &
        (days_since_bc <= st_lookahead) &
        near_high &
        weaker_vol
    ).fillna(False)

    # UTAD: poke above range high then close back below
    rh = out["top_range_high"].copy()
    rh_fallback = high.rolling(utad_window).max()
    rh = rh.fillna(rh_fallback)
    out["top_range_high_eff"] = rh

    out["top_utad"] = ((high > rh * (1.0 + utad_break)) & (close < rh)).fillna(False)

    # Top score
    score = (
        40 * out["top_bc"].astype(int) +
        20 * out["top_ar"].astype(int) +
        20 * out["top_st"].astype(int) +
        20 * out["top_utad"].astype(int)
    ).clip(0, 100)

    if score_smooth > 1:
        score = score.rolling(score_smooth).max()

    out["top_score"] = score.fillna(0.0)
    return out