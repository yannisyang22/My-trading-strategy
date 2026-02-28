import numpy as np
import pandas as pd


def compute_fib_targets(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Simple, non-peeking fib extension targets using *past-only* swing range.
    We approximate swing as rolling high/low over lookback and shift(1) to avoid future use.

    Outputs:
      - swing_high, swing_low (past-only)
      - tp1_long, tp2_long (bool): price reached extension targets
      - tp1_short, tp2_short (bool)
    """
    fcfg = cfg.get("fib", {})
    L = int(fcfg.get("lookback", 120))

    ext1 = float(fcfg.get("ext1", 1.272))
    ext2 = float(fcfg.get("ext2", 1.618))

    high = df["high"]
    low = df["low"]
    close = df["close"]

    swing_high = high.rolling(L).max().shift(1)
    swing_low = low.rolling(L).min().shift(1)

    rng = (swing_high - swing_low).replace(0, np.nan)

    # Long extension levels: above swing_high
    lv1_long = swing_high + (ext1 - 1.0) * rng
    lv2_long = swing_high + (ext2 - 1.0) * rng

    # Short extension levels: below swing_low
    lv1_short = swing_low - (ext1 - 1.0) * rng
    lv2_short = swing_low - (ext2 - 1.0) * rng

    out = pd.DataFrame(index=df.index)
    out["swing_high"] = swing_high
    out["swing_low"] = swing_low
    out["fib_lv1_long"] = lv1_long
    out["fib_lv2_long"] = lv2_long
    out["fib_lv1_short"] = lv1_short
    out["fib_lv2_short"] = lv2_short

    out["tp1_long"] = (close >= lv1_long).fillna(False)
    out["tp2_long"] = (close >= lv2_long).fillna(False)
    out["tp1_short"] = (close <= lv1_short).fillna(False)
    out["tp2_short"] = (close <= lv2_short).fillna(False)

    return out