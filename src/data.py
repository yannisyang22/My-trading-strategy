import pandas as pd
import yfinance as yf

def load_daily_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    # ✅ 关键：如果是 MultiIndex（比如 ('Open','BTC-USD')），先选出 ticker 那一层
    if isinstance(df.columns, pd.MultiIndex):
        # 通常第 2 层是 ticker
        df = df.xs(ticker, axis=1, level=1, drop_level=True)

    # 统一列名
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })

    # 只保留需要的列 + 去空
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df.index = pd.to_datetime(df.index)

    # ✅ 防御：确保每一列都是一维（Series），不是“同名多列”
    # 如果出现重复列名（极少见），就取第一列
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    return df