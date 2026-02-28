import yaml
from src.data import load_daily_ohlcv
from src.strategy import build_signals  
from src.backtest import backtest
from src.metrics import summary_stats
from src.plot import plot_equity_and_benchmark, plot_drawdown


def main():
    cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))

    ticker = cfg["universe"]["ticker"]
    start = cfg["universe"]["start"]
    end = cfg["universe"]["end"]

    # 1) 数据
    df = load_daily_ohlcv(ticker, start, end)

    # 2) 指标与信号
    sig_df = build_signals(df, cfg)
    print("\n=== SIGNAL COUNTS ===")
    for col in ["long_ok","short_ok","bb_squeeze","bb_open","bb_expanding","grid_mode","is_range_regime","boost1","boost2"]:
        if col in sig_df.columns:
            print(col, int(sig_df[col].fillna(False).sum()))

    print("\n=== POS COUNTS (daily raw) ===")
    print(sig_df["pos_daily_raw"].value_counts(dropna=False).head(10))

    # --- 关键：次日执行，避免未来函数 ---
    pos_exec = sig_df["pos_daily_raw"].shift(1).fillna(0.0)

    # 3) 回测（带回撤熔断）
    bt = backtest(
        df=df,
        position=pos_exec,
        fee_bps=cfg["costs"]["fee_bps"],
        slippage_bps=cfg["costs"]["slippage_bps"],
        initial_equity=cfg["backtest"]["initial_equity"],
        max_drawdown_stop=cfg.get("risk", {}).get("max_drawdown_stop", 0.20),
        cooldown_days=cfg.get("risk", {}).get("cooldown_days", 21),
        pos_step_threshold=cfg.get("costs", {}).get("pos_step_threshold", 0.0),
    )

    # 4) 统计
    stats = summary_stats(bt)
    print("===== STATS =====")
    for k, v in stats.items():
        print(f"{k:>14}: {v}")

    # 参与度检查
    print("pct_long:", float((bt["pos"] > 0).mean()))
    print("pct_short:", float((bt["pos"] < 0).mean()))
    print("pct_flat:", float((bt["pos"] == 0).mean()))

    if "grid_mode" in sig_df.columns:
        print("grid_mode pct:", float(sig_df["grid_mode"].mean()))
    elif "is_range_regime" in sig_df.columns:
        print("grid_mode pct (is_range_regime):", float(sig_df["is_range_regime"].mean()))
    else:
        print("grid_mode pct: N/A")
    

    
    
    # squeeze_release 次数（诊断信号是否太少）
    if "squeeze_release" in sig_df.columns:
        sr_cnt = int(sig_df["squeeze_release"].fillna(False).sum())
        print("\nsqueeze_release count:", sr_cnt)

    # 5) Buy & Hold 基准（使用 backtest 里的 asset_ret）
    bh_equity = (1.0 + bt["asset_ret"]).cumprod()
    print("\nBuy&Hold final_equity:", float(bh_equity.iloc[-1]))

    # 6) 交易原因日志（对齐到“执行日”）
    reason_cols = [
        # --- core positions/regime ---
        "pos_trend_raw", "pos_grid_raw", "pos_daily_raw",
        "is_range_regime", "grid_mode",
        "bb_squeeze", "bb_open", "bb_expanding", "bear_trend", "bull_env",
        "boost1", "boost2",

        # --- trend indicators / scores ---
        "ema_fast", "sma_mid",
        "bb_mid", "bb_up", "bb_dn", "bb_bw",
        "macd_hist",
        "long_score", "short_score",
        "long_score_adj", "short_score_adj", "gap_adj",
        "long_ok", "short_ok",
        "exit_long", "exit_short",

        # --- wyckoff + fib overlays ---
        "bottom_score", "top_score",
        "wy_long_bonus", "wy_short_bonus", "wy_ban_long",
        "tp1_long", "tp2_long", "tp1_short", "tp2_short",

        # --- grid diagnostics ---
        "g_bb_mid", "g_bb_up", "g_bb_dn", "g_bb_bw", "grid_z",

        # --- risk overlays if enabled ---
        "stop_hit1", "stop_hit2",
        "vt_scale",
    ]
    reason_cols = [c for c in reason_cols if c in sig_df.columns]
    full = bt.join(sig_df[reason_cols].shift(1), how="left")
    
    trade_log = full[full["trade"]].copy()
    trade_log["pos_prev"] = trade_log["pos"].shift(1)
    trade_log["pos_change"] = trade_log["pos_prev"].astype(str) + " -> " + trade_log["pos"].astype(str)

    show_cols = [
    "pos_change", "pos", "cost", "equity", "drawdown", "cb_active",
    "close", "volume",
    "ema_fast", "sma_mid",
    "macd_hist",
    "bb_bw", "bw_q",
    "squeeze_ok", "is_expanding", "squeeze_release", "in_release_window",
    "mid_up", "mid_dn", "cross_up", "cross_dn",
    "trend_long_ok", "trend_short_ok",
    "long_ok", "short_ok",
]
    show_cols = [c for c in show_cols if c in trade_log.columns]

    print("\n===== TRADE REASONS (first 50 trades) =====")
    print(trade_log[show_cols].head(50))

    # 导出 CSV 方便复盘
    trade_log.to_csv("trade_reasons.csv")
    print("\nSaved: trade_reasons.csv")

    # 7) 画图
    plot_equity_and_benchmark(bt, bh_equity, title=f"{ticker} Strategy vs Buy&Hold")
    plot_drawdown(bt["equity"], title=f"{ticker} Strategy Drawdown")

    

if __name__ == "__main__":
    main()