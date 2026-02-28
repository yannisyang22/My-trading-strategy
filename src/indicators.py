import pandas as pd

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def bollinger(close: pd.Series, n: int, k: float):
    mid = sma(close, n)
    std = close.rolling(n).std()
    upper = mid + k * std
    lower = mid - k * std
    bandwidth = (upper - lower) / mid
    return mid, upper, lower, bandwidth

def macd(close: pd.Series, fast: int, slow: int, signal: int):
    m_fast = ema(close, fast)
    m_slow = ema(close, slow)
    line = m_fast - m_slow
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


    