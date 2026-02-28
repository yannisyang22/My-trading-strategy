import numpy as np
import pandas as pd

from .indicators import ema, sma, bollinger, macd
from .wyckoff import compute_wyckoff_bottom, compute_wyckoff_top
from .fib import compute_fib_targets


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df.resample(rule).agg(agg).dropna()


def _map_to_daily(s: pd.Series, daily_index: pd.DatetimeIndex, shift1: bool = True) -> pd.Series:
    """Map higher-timeframe series to daily index. shift1=True avoids peeking on incomplete bar."""
    if shift1:
        s = s.shift(1)
    return s.reindex(daily_index, method="ffill")


def _tf_indicators(
    df_tf: pd.DataFrame,
    ema_n: int,
    sma_n: int,
    bb_n: int,
    bb_k: float,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
) -> pd.DataFrame:
    close = df_tf["close"]
    out = pd.DataFrame(index=df_tf.index)
    out["ema"] = ema(close, ema_n)
    out["sma"] = sma(close, sma_n)
    bb_mid, bb_up, bb_dn, bb_bw = bollinger(close, bb_n, bb_k)
    out["bb_mid"] = bb_mid
    out["bb_up"] = bb_up
    out["bb_dn"] = bb_dn
    out["bb_bw"] = bb_bw
    macd_line, macd_sig, macd_hist = macd(close, macd_fast, macd_slow, macd_signal)
    out["macd_hist"] = macd_hist
    out["close"] = close
    return out


def build_trend_signals(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Multi-timeframe (synthetic bars) scoring trend module.
    - Timeframes: 1D,2D,3D,5D,W,ME
    - Score = trend alignment + momentum alignment, weighted by timeframe
    - Wyckoff + Fib auxiliary:
        * Wyckoff bottom/top add/subtract a small bonus
        * Fib tp1/tp2 reduce exposure

    FIXES in this version:
      1) ban_short now truly affects out["short_ok"] and position (previously order bug)
      2) wy_short_bonus threshold uses wyckoff_top.left_short_score (previously read from mtf_score)
    """
    c = cfg["indicators"]
    f = cfg.get("filters", {})
    t = cfg.get("trend", {})
    wbot = cfg.get("wyckoff", {})
    wtop = cfg.get("wyckoff_top", {})
    fibcfg = cfg.get("fib", {})
    mtf = cfg.get("mtf_score", {})

    out = df.copy()

    # ---------------------------
    # Base daily indicators (for logs / exits reference)
    # ---------------------------
    ema_n = int(c.get("ema_fast", 56))
    sma_n = int(c.get("sma_mid", 116))
    bb_n = int(c.get("bb_n", 20))
    bb_k = float(c.get("bb_k", 2.0))
    mf = int(c.get("macd_fast", 12))
    ms = int(c.get("macd_slow", 26))
    mg = int(c.get("macd_signal", 9))

    d_ind = _tf_indicators(out[["open", "high", "low", "close", "volume"]], ema_n, sma_n, bb_n, bb_k, mf, ms, mg)
    out["ema_fast"] = d_ind["ema"]
    out["sma_mid"] = d_ind["sma"]
    out["bb_mid"] = d_ind["bb_mid"]
    out["bb_up"] = d_ind["bb_up"]
    out["bb_dn"] = d_ind["bb_dn"]
    out["bb_bw"] = d_ind["bb_bw"]
    out["macd_hist"] = d_ind["macd_hist"]

    # ---------------------------
    # Multi-timeframe set
    # ---------------------------
    w_freq = str(f.get("weekly_resample", "W-SUN"))
    tfs = [
        ("1D", "1D", 1.0, False),  # no shift on daily
        ("2D", "2D", 1.0, True),
        ("3D", "3D", 1.0, True),
        ("5D", "5D", 1.0, True),
        ("W",  w_freq, 2.0, True),
        ("ME", "ME",   3.0, True),
    ]
    # allow overriding weights in config
    w_over = mtf.get("weights", {})

    def wgt(name, default):
        return float(w_over.get(name, default))

    # scoring thresholds
    long_th = float(mtf.get("long_th", 6.0))
    short_th = float(mtf.get("short_th", 6.0))
    gap_th = float(mtf.get("gap_th", 1.0))

    # trend logic knobs
    use_bb_mid = bool(mtf.get("use_bb_mid", True))
    use_macd = bool(mtf.get("use_macd", True))
    trend_mode = str(mtf.get("trend_mode", "both")).lower()  # "both" or "either"

    long_score = pd.Series(0.0, index=out.index)
    short_score = pd.Series(0.0, index=out.index)

    for name, rule, default_w, need_shift in tfs:
        df_tf = out[["open", "high", "low", "close", "volume"]].resample(rule).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()

        ind = _tf_indicators(df_tf, ema_n, sma_n, bb_n, bb_k, mf, ms, mg)

        close_tf = _map_to_daily(ind["close"], out.index, shift1=need_shift)
        ema_tf = _map_to_daily(ind["ema"], out.index, shift1=need_shift)
        sma_tf = _map_to_daily(ind["sma"], out.index, shift1=need_shift)
        bbm_tf = _map_to_daily(ind["bb_mid"], out.index, shift1=need_shift)
        macd_tf = _map_to_daily(ind["macd_hist"], out.index, shift1=need_shift)

        # trend alignment
        if trend_mode == "either":
            up_trend = (close_tf > ema_tf) | (close_tf > sma_tf)
            dn_trend = (close_tf < ema_tf) | (close_tf < sma_tf)
        else:
            up_trend = (close_tf > ema_tf) & (close_tf > sma_tf)
            dn_trend = (close_tf < ema_tf) & (close_tf < sma_tf)

        if use_bb_mid:
            up_trend = up_trend & (close_tf > bbm_tf)
            dn_trend = dn_trend & (close_tf < bbm_tf)

        # momentum alignment
        if use_macd:
            up_momo = (macd_tf > 0)
            dn_momo = (macd_tf < 0)
        else:
            up_momo = pd.Series(True, index=out.index)
            dn_momo = pd.Series(True, index=out.index)

        w = wgt(name, default_w)
        # each timeframe contributes up to 2*w (trend + momo)
        long_score += w * (up_trend.astype(int) + up_momo.astype(int))
        short_score += w * (dn_trend.astype(int) + dn_momo.astype(int))

        if bool(mtf.get("emit_debug_cols", False)):
            out[f"{name}_up"] = up_trend.fillna(False)
            out[f"{name}_dn"] = dn_trend.fillna(False)

    out["long_score"] = long_score
    out["short_score"] = short_score

    # ---------------------------
    # Wyckoff
    # ---------------------------
    wyb = compute_wyckoff_bottom(out, cfg)
    wyt = compute_wyckoff_top(out, cfg)

    for col in ["bottom_score", "sc", "ar", "st", "spring", "sos"]:
        if col in wyb.columns:
            out[col] = wyb[col]
    for col in ["top_score", "top_bc", "top_ar", "top_st", "top_utad"]:
        if col in wyt.columns:
            out[col] = wyt[col]

    wy_bonus = float(mtf.get("wy_bonus", 1.0))
    bottom_boost_th = float(wbot.get("early_long_score", 75))
    top_pen_th = float(wtop.get("ban_long_score", 60))
    left_short_th = float(wtop.get("left_short_score", 70))  # FIX: use wyckoff_top

    out["wy_long_bonus"] = (out.get("bottom_score", 0.0) >= bottom_boost_th).fillna(False)
    out["wy_short_bonus"] = (out.get("top_score", 0.0) >= left_short_th).fillna(False)  # FIX
    out["wy_ban_long"] = (out.get("top_score", 0.0) >= top_pen_th).fillna(False)

    out["long_score_adj"] = out["long_score"] + wy_bonus * out["wy_long_bonus"].astype(int)
    out["short_score_adj"] = out["short_score"] + wy_bonus * out["wy_short_bonus"].astype(int)

    # ---------------------------
    # Entry signals from adjusted score
    # ---------------------------
    gap = out["long_score_adj"] - out["short_score_adj"]
    long_ok = (out["long_score_adj"] >= long_th) & (gap >= gap_th) & (~out["wy_ban_long"])
    short_ok = (out["short_score_adj"] >= short_th) & (-gap >= gap_th)

    # --- Entry confirmation (require signal persists N days) ---
    entry_confirm_days = int(mtf.get("entry_confirm_days", 2))
    entry_confirm_days_bear = int(mtf.get("entry_confirm_days_bear", 4))

    confirm_n = entry_confirm_days
    if entry_confirm_days > 1:
        long_ok = (long_ok.fillna(False).rolling(entry_confirm_days).sum() >= entry_confirm_days)
        short_ok = (short_ok.fillna(False).rolling(entry_confirm_days).sum() >= entry_confirm_days)

    # ---- FIX: ban_short must apply BEFORE out["short_ok"] and position sizing ----
    ban_short_th = float(wbot.get("ban_short_score", 60))
    ban_short = (out.get("bottom_score", 0.0) >= ban_short_th).fillna(False)
    short_ok = short_ok & (~ban_short)

    out["long_ok"] = long_ok.fillna(False)
    out["short_ok"] = short_ok.fillna(False)
    out["gap_adj"] = gap.fillna(0.0)

    # ---------------------------
    # Position sizing
    # ---------------------------
    long_size = float(t.get("long_size", 1.0))
    short_size = float(t.get("short_size", 0.6))

    # early long (aux only)
    early_long_lev = float(wbot.get("early_long_leverage", 0.6))
    early_long = (out.get("bottom_score", 0.0) >= bottom_boost_th).fillna(False)

    pos = pd.Series(np.nan, index=out.index, dtype="float64")
    pos[out["long_ok"]] = long_size
    pos[out["short_ok"]] = -short_size
    pos[out["long_ok"] & early_long] = max(min(long_size, 1.0), early_long_lev)

    # Exits: hysteresis (exit_gap) + BB mid + MACD sign
    exit_gap = float(mtf.get("exit_gap", 1.5))
    exit_long = (
        (gap < -exit_gap) |
        ((out["close"] < out["bb_mid"]) & (out["macd_hist"] < 0))
    )
    exit_short = (
        (gap > exit_gap) |
        ((out["close"] > out["bb_mid"]) & (out["macd_hist"] > 0))
    )
    out["exit_long"] = exit_long.fillna(False)
    out["exit_short"] = exit_short.fillna(False)

    pos = pos.ffill().fillna(0.0)
    pos2 = pos.copy()
    pos2[(pos2 > 0) & out["exit_long"]] = 0.0
    pos2[(pos2 < 0) & out["exit_short"]] = 0.0
    pos2 = pos2.ffill().fillna(0.0)

    # ---------------------------
    # Early bottom probe long (minimal impact)
    # - Only when FLAT (pos2==0)
    # - Only when bottom_score >= early_long_score for N days
    # - Small size (0.2 by default)
    # ---------------------------
    probe_th = float(wbot.get("early_long_score", 75))
    probe_size = float(wbot.get("probe_long_size", 0.20))
    probe_days = int(wbot.get("probe_confirm_days", 2))

    bottom_score = out.get("bottom_score", pd.Series(0.0, index=out.index)).fillna(0.0)
    probe = (bottom_score >= probe_th)

    if probe_days > 1:
        probe = (probe.rolling(probe_days).sum() >= probe_days)

    probe = probe.fillna(False)
    out["probe_long"] = probe

    # apply only when flat, do NOT override normal long/short positions
    pos2 = pos2.copy()
    pos2[(pos2 == 0.0) & probe] = probe_size

    # ---------------------------
    # Fib auxiliary: tp1/tp2 reduce only
    # ---------------------------
    fib = compute_fib_targets(out, cfg)
    out = out.join(fib, how="left")

    tp1_reduce = float(fibcfg.get("tp1_reduce_to", 0.6))
    tp2_reduce = float(fibcfg.get("tp2_reduce_to", 0.3))

    p = pos2.copy()
    p[(p > 0) & out["tp1_long"].fillna(False)] = p[(p > 0) & out["tp1_long"].fillna(False)].clip(upper=tp1_reduce)
    p[(p > 0) & out["tp2_long"].fillna(False)] = p[(p > 0) & out["tp2_long"].fillna(False)].clip(upper=tp2_reduce)

    p[(p < 0) & out["tp1_short"].fillna(False)] = p[(p < 0) & out["tp1_short"].fillna(False)].clip(lower=-tp1_reduce)
    p[(p < 0) & out["tp2_short"].fillna(False)] = p[(p < 0) & out["tp2_short"].fillna(False)].clip(lower=-tp2_reduce)

    # minimum hold (optional, default 0)
    hold_days = int(t.get("min_hold_days", 0))
    if hold_days > 0:
        p = _apply_min_hold_allow_exit(p, hold_days)

    out["pos_trend_raw"] = p.fillna(0.0)
    return out


def _apply_min_hold_allow_exit(pos: pd.Series, hold_days: int) -> pd.Series:
    p = pos.copy().astype(float).fillna(0.0)
    current = 0.0
    entry_i = None
    idx = p.index
    for i in range(len(idx)):
        desired = float(p.iloc[i])
        if current == 0.0:
            if desired != 0.0:
                current = desired
                entry_i = i
            p.iloc[i] = current
            continue

        days_held = i - (entry_i if entry_i is not None else i)

        # exits always allowed
        if desired == 0.0:
            current = 0.0
            entry_i = None
            p.iloc[i] = current
            continue

        # during hold lock, ignore flips
        if days_held < hold_days:
            p.iloc[i] = current
            continue

        if desired != current:
            current = desired
            entry_i = i
        p.iloc[i] = current
    return p