import numpy as np
import pandas as pd

from .indicators import sma, bollinger
from .strategy_trend import build_trend_signals
from .strategy_grid import build_grid_position


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def build_signals(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()

    r = cfg.get("regime", {})
    c = cfg["indicators"]
    g = cfg.get("grid", {})
    risk = cfg.get("risk", {})
    lb = cfg.get("leverage_boost", {})
    bbreg = cfg.get("bb_regime", {})
    filt = cfg.get("filters", {})
    stop_cfg = cfg.get("stop", {})
    vt_cfg = cfg.get("vol_target", {})

    # --- components ---
    trend_df = build_trend_signals(out, cfg)
    grid_df = build_grid_position(out, cfg)

    merged = out.copy()
    for col in trend_df.columns:
        if col not in merged.columns:
            merged[col] = trend_df[col]
    for col in grid_df.columns:
        if col not in merged.columns:
            merged[col] = grid_df[col]

    if "pos_trend_raw" not in merged.columns:
        raise ValueError("pos_trend_raw not found (strategy_trend.py)")
    if "pos_grid_raw" not in merged.columns:
        raise ValueError("pos_grid_raw not found (strategy_grid.py)")

    # ---------------------------
    # A) Daily RANGE regime
    # ---------------------------
    bb_n = int(r.get("bb_n", c["bb_n"]))
    bb_k = float(r.get("bb_k", c["bb_k"]))

    _, _, _, bw = bollinger(merged["close"], bb_n, bb_k)
    merged["reg_bb_bw"] = bw

    w_win = int(r.get("bw_window", 252))
    q = float(r.get("bw_quantile", 0.40))
    merged["reg_bw_q"] = merged["reg_bb_bw"].rolling(w_win).quantile(q)
    bw_low = (merged["reg_bb_bw"] < merged["reg_bw_q"]).fillna(False)

    slope_days = int(r.get("ma_slope_days", 20))
    slope_th = float(r.get("ma_slope_th", 0.02))
    ma = sma(merged["close"], int(c["sma_mid"]))
    merged["reg_ma116"] = ma
    # FIX: avoid divide-by-zero / extreme
    slope = (ma - ma.shift(slope_days)).abs() / (ma.abs() + 1e-9)
    ma_flat = (slope < slope_th).fillna(False)

    is_range = (bw_low & ma_flat).fillna(False)
    confirm_days = int(r.get("confirm_days", 1))
    if confirm_days > 1:
        is_range = (is_range.rolling(confirm_days).mean() >= (confirm_days - 1) / confirm_days).fillna(False)

    merged["is_range_regime"] = is_range
    merged["grid_mode"] = is_range

    # ---------------------------
    # B) Weekly BB squeeze/open/expanding
    # ---------------------------
    w_freq = filt.get("weekly_resample", "W-SUN")
    w_close = merged["close"].resample(w_freq).last()
    w_mid, _, _, w_bw = bollinger(w_close, bb_n, bb_k)

    # FIX: shift(1) before mapping to daily to avoid peeking current week
    merged["w_bb_mid"] = w_mid.shift(1).reindex(merged.index, method="ffill")
    merged["w_bb_bw"]  = w_bw.shift(1).reindex(merged.index, method="ffill")

    lookback_w = int(bbreg.get("lookback_weeks", 156))
    squeeze_q = float(bbreg.get("squeeze_q", 0.20))
    open_q = float(bbreg.get("open_q", 0.60))
    rise_weeks = int(bbreg.get("rise_weeks", 2))

    w_bw_q_low = w_bw.rolling(lookback_w).quantile(squeeze_q).shift(1)
    w_bw_q_hi = w_bw.rolling(lookback_w).quantile(open_q).shift(1)
    bw_low_d = w_bw_q_low.reindex(merged.index, method="ffill")
    bw_hi_d = w_bw_q_hi.reindex(merged.index, method="ffill")

    squeeze = (merged["w_bb_bw"] < bw_low_d).fillna(False)
    open_trend = (merged["w_bb_bw"] > bw_hi_d).fillna(False)

    w_bw_rising = (w_bw.diff() > 0).rolling(rise_weeks).sum() >= rise_weeks
    expanding = w_bw_rising.shift(1).reindex(merged.index, method="ffill").fillna(False)

    merged["bb_squeeze"] = squeeze
    merged["bb_open"] = open_trend
    merged["bb_expanding"] = expanding

    above_week_mid = (merged["close"] > merged["w_bb_mid"]).fillna(False)
    below_week_mid = (merged["close"] < merged["w_bb_mid"]).fillna(False)

    bear_trend = ((open_trend | expanding) & below_week_mid & (~squeeze)).fillna(False)
    merged["bear_trend"] = bear_trend

    # ---------------------------
    # C) Base pos: range->grid else->trend
    # ---------------------------
    pos_trend = merged["pos_trend_raw"].astype(float).fillna(0.0)
    pos_grid = merged["pos_grid_raw"].astype(float).fillna(0.0)

    pos = pos_trend.where(~is_range, pos_grid)

    # ---------------------------
    # D) Squeeze blend
    # ---------------------------
    alpha = float(bbreg.get("grid_alpha_in_squeeze", 0.50))
    trend_mult_sq = float(bbreg.get("trend_mult_in_squeeze", 0.60))
    pos_sq = (1 - alpha) * (pos_trend * trend_mult_sq) + alpha * pos_grid
    pos = pos.where(~squeeze, pos_sq)

    # ---------------------------
    # D2) Bull pullback: de-risk trend + add small grid (only in bull trend)
    # ---------------------------
    pb = cfg.get("pullback_grid", {})

    if bool(pb.get("enabled", True)):
        # define bull trend environment (weekly open/expanding and above weekly mid)
        bull_trend = (open_trend | expanding) & above_week_mid & (~squeeze)
        merged["bull_trend"] = bull_trend.fillna(False)

        # support zone from EMA/SMA (already in trend_df)
        if ("ema_fast" in merged.columns) and ("sma_mid" in merged.columns):
            zone_low = pd.concat([merged["ema_fast"], merged["sma_mid"]], axis=1).min(axis=1)
            zone_high = pd.concat([merged["ema_fast"], merged["sma_mid"]], axis=1).max(axis=1)
        else:
            # fallback if columns missing
            zone_low = merged["close"].rolling(56).mean()
            zone_high = merged["close"].rolling(116).mean()

        # pullback definition
        band = float(pb.get("support_band", 0.006))  # 0.6%
        in_support = (merged["close"] >= zone_low * (1 - band)) & (merged["close"] <= zone_high * (1 + band))
        below_daily_mid = (merged.get("bb_mid", merged["close"]).notna()) & (merged["close"] < merged["bb_mid"])

        pullback = bull_trend & (in_support | below_daily_mid)
        confirm = int(pb.get("confirm_days", 2))
        if confirm > 1:
            pullback = (pullback.fillna(False).rolling(confirm).sum() >= confirm).fillna(False)
        else:
            pullback = pullback.fillna(False)

        merged["bull_pullback"] = pullback

        # trend de-risk factor in pullback
        trend_mult_pb = float(pb.get("trend_mult", 0.75))   # keep most trend exposure
        # grid mix weight in pullback
        alpha_pb = float(pb.get("grid_alpha", 0.35))        # mix in some grid

        # grid cap for pullback (keep small so DD doesn't blow up)
        pb_grid_cap = float(pb.get("grid_cap", 0.30))
        pos_grid_cap = pos_grid.clip(-pb_grid_cap, pb_grid_cap)

        pos_pb = (1 - alpha_pb) * (pos_trend * trend_mult_pb) + alpha_pb * pos_grid_cap

        # apply only during pullback and NOT in range regime (this is "trend pullback grid", not pure range)
        pos = pos.where(~pullback, pos_pb)

    # neutral cap
    mid_mult = float(bbreg.get("trend_mult_in_neutral", 0.85))
    cap_neutral = float(bbreg.get("trend_cap_in_neutral", 0.80))
    neutral = (~squeeze) & (~open_trend) & (~expanding)
    pos[(pos > 0) & neutral] = (pos[(pos > 0) & neutral] * mid_mult).clip(upper=cap_neutral)
    pos[(pos < 0) & neutral] = (pos[(pos < 0) & neutral] * mid_mult).clip(lower=-cap_neutral)

    # ---------------------------
    # E) Fixed leverage boost (LONG only)
    # ---------------------------
    if "gap_adj" in merged.columns:
        gap = merged["gap_adj"].fillna(0.0)
    elif ("long_score_adj" in merged.columns) and ("short_score_adj" in merged.columns):
        gap = (merged["long_score_adj"] - merged["short_score_adj"]).fillna(0.0)
    elif ("long_score" in merged.columns) and ("short_score" in merged.columns):
        gap = (merged["long_score"] - merged["short_score"]).fillna(0.0)
    else:
        gap = pd.Series(0.0, index=merged.index)
    merged["gap_adj"] = gap

    rv = merged["close"].pct_change().rolling(20).std()
    rv_q = float(risk.get("rv_q", 0.75))
    rv_hi = (rv > rv.rolling(252).quantile(rv_q)).fillna(False)

    bull_env = (open_trend | expanding) & above_week_mid & (~squeeze)
    merged["bull_env"] = bull_env

    gap1 = float(lb.get("gap1", 10.0))
    gap2 = float(lb.get("gap2", 15.0))
    mult1 = float(lb.get("mult1", 1.05))
    mult2 = float(lb.get("mult2", 1.30))

    boost1 = (bull_env & (gap >= gap1) & (~rv_hi)).fillna(False)
    boost2 = (bull_env & (gap >= gap2) & (~rv_hi)).fillna(False)
    merged["boost1"] = boost1
    merged["boost2"] = boost2

    pos = pos.where(~(boost1 & (pos > 0)), pos * mult1)
    pos = pos.where(~(boost2 & (pos > 0)), pos * mult2)

    # ---------------------------
    # F) Directional grid by weekly mid (grid only)
    # ---------------------------
    mask = is_range
    pos.loc[mask & above_week_mid] = pos.loc[mask & above_week_mid].clip(lower=0.0)
    pos.loc[mask & below_week_mid] = pos.loc[mask & below_week_mid].clip(upper=0.0)

    # --- Bear range grid boost: only when bear_trend AND weekly squeeze (range inside bear) ---
    br = cfg.get("bear_grid", {})
    if bool(br.get("enabled", True)):
        bear = merged.get("bear_trend", pd.Series(False, index=merged.index)).fillna(False)

        not_expanding = ~merged.get("bb_expanding", pd.Series(False, index=merged.index)).fillna(False)

        in_week_box = (
            (merged["close"] <= merged.get("g_bb_up", merged["close"])) &
            (merged["close"] >= merged.get("g_bb_dn", merged["close"]))
        ).fillna(False)

        bw = merged.get("w_bb_bw", pd.Series(np.nan, index=merged.index))
        bw_med = bw.rolling(int(br.get("bw_ref_window", 156))).median()
        bw_lowish = (bw < bw_med).fillna(False)

        bear_box = bear & not_expanding & in_week_box & bw_lowish
        merged["bear_box"] = bear_box

        mult = float(br.get("mult", 1.25))
        cap  = float(br.get("cap", 0.60))

        on_grid = merged["grid_mode"].fillna(False)
        idx = bear_box & on_grid
        pos.loc[idx] = (pos.loc[idx] * mult).clip(-cap, cap)
    
    # ---------------------------
    # G) Bear multiplier (optional)
    # ---------------------------
    bear_mult = float(risk.get("bear_mult", 1.0))
    if bear_mult < 1.0:
        pos = pos.where(~bear_trend, pos * bear_mult)

    # --- Bear long cap (very light drawdown control) ---
    max_long_in_bear = float(risk.get("max_long_in_bear", 1.0))
    if max_long_in_bear < 1.0:
        pos[(pos > 0) & bear_trend] = pos[(pos > 0) & bear_trend].clip(upper=max_long_in_bear)
    # ---------------------------
    # H) High-vol soft cap
    # ---------------------------
    high_vol_mult = float(risk.get("high_vol_mult", 0.85))
    pos = pos.where(~rv_hi, pos * high_vol_mult)

    # ---------------------------
    # I) ATR soft stop
    # ---------------------------
    if bool(stop_cfg.get("enabled", False)):
        atr = _atr(merged, int(stop_cfg.get("atr_n", 14)))
        m1 = float(stop_cfg.get("m1", 2.0))
        m2 = float(stop_cfg.get("m2", 3.0))
        win = int(stop_cfg.get("trail_window", 20))

        close = merged["close"]
        peak = close.rolling(win).max()
        trough = close.rolling(win).min()

        long_stop1 = close < (peak - m1 * atr)
        long_stop2 = close < (peak - m2 * atr)
        short_stop1 = close > (trough + m1 * atr)
        short_stop2 = close > (trough + m2 * atr)

        big = pos.abs() > float(stop_cfg.get("only_if_abs_pos_gt", 0.8))
        mult_s1 = float(stop_cfg.get("mult1", 0.70))
        mult_s2 = float(stop_cfg.get("mult2", 0.40))

        hit1 = (pos > 0) & big & long_stop1
        hit2 = (pos > 0) & big & long_stop2
        hit1s = (pos < 0) & big & short_stop1
        hit2s = (pos < 0) & big & short_stop2

        pos = pos.where(~hit1, pos * mult_s1)
        pos = pos.where(~hit2, pos * mult_s2)
        pos = pos.where(~hit1s, pos * mult_s1)
        pos = pos.where(~hit2s, pos * mult_s2)

        merged["stop_hit1"] = (hit1 | hit1s).fillna(False)
        merged["stop_hit2"] = (hit2 | hit2s).fillna(False)

    # ---------------------------
    # J) Vol targeting (downscale only)
    # ---------------------------
    if bool(vt_cfg.get("enabled", False)):
        win = int(vt_cfg.get("win", 20))
        target = float(vt_cfg.get("target", 0.026))
        min_scale = float(vt_cfg.get("min_scale", 0.30))
        max_scale = float(vt_cfg.get("max_scale", 1.0))
        only_big = float(vt_cfg.get("only_if_abs_pos_gt", 0.8))

        rret = merged["close"].pct_change().fillna(0.0)
        vol = rret.rolling(win).std()
        scale = (target / vol.replace(0, np.nan)).clip(lower=min_scale, upper=max_scale).fillna(1.0)

        big2 = pos.abs() > only_big
        pos = pos.where(~big2, pos * scale)
        merged["vt_scale"] = scale

    # ---------------------------
    # Left-side small short + reclaim de-risk
    # ---------------------------
    sr = cfg.get("short_retest", {})
    band = float(sr.get("band", 0.012))
    size = float(sr.get("size", 0.20))
    require_expanding = bool(sr.get("require_expanding", True))

    near_mid = (
        (merged["close"] >= merged["w_bb_mid"] * (1 - band)) &
        (merged["close"] <= merged["w_bb_mid"] * (1 + band))
    ).fillna(False)

    bear_ok = merged.get("bear_trend", pd.Series(False, index=merged.index)).fillna(False)
    if require_expanding:
        bear_ok = bear_ok & merged.get("bb_expanding", pd.Series(False, index=merged.index)).fillna(False)

    left_short = bear_ok & near_mid
    pos[left_short] = np.minimum(pos[left_short], -size)
    merged["left_short"] = left_short

    short_reduce = float(cfg.get("risk", {}).get("short_reduce_on_reclaim", 0.5))
    reclaim_mid = (merged["close"] > merged["w_bb_mid"]).fillna(False)
    pos = pos.where(~((pos < 0) & reclaim_mid), pos * short_reduce)

    # ---------------------------
    # Final caps
    # ---------------------------
    Lmax = float(risk.get("leverage_cap", 1.5))
    pos = pos.clip(-Lmax, Lmax).fillna(0.0)

    grid_cap = float(g.get("grid_leverage_cap", Lmax))
    pos.loc[mask] = pos.loc[mask].clip(-grid_cap, grid_cap)

    merged["pos_daily_raw"] = pos
    return merged