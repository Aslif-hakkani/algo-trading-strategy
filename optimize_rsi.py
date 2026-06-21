import yfinance as yf
import pandas as pd
import ta

df = yf.download(
    "RELIANCE.NS",
    start="2020-01-01",
    end="2025-01-01",
    auto_adjust=True
)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df["EMA50"] = ta.trend.EMAIndicator(
    close=df["Close"],
    window=50
).ema_indicator()

df["EMA200"] = ta.trend.EMAIndicator(
    close=df["Close"],
    window=200
).ema_indicator()

df["VOL_MA20"] = df["Volume"].rolling(20).mean()

for rsi_level in [50, 55, 60, 65, 70]:
    
    df["RSI"] = ta.momentum.RSIIndicator(
        close=df["Close"],
        window=14
    ).rsi()

    signal = (
        (df["EMA50"] > df["EMA200"]) &
        (df["RSI"] > rsi_level) &
        (df["Volume"] > df["VOL_MA20"])
    )

    temp = df.copy()

    temp["Future_Close"] = temp["Close"].shift(-5)

    temp["Return_%"] = (
        (temp["Future_Close"] - temp["Close"])
        / temp["Close"]
    ) * 100

    trades = temp[signal]

    avg_return = trades["Return_%"].mean()

    win_rate = (
        (trades["Return_%"] > 0).mean()
    ) * 100

    print(
        f"RSI>{rsi_level} | "
        f"Signals={len(trades)} | "
        f"Avg={avg_return:.2f}% | "
        f"Win={win_rate:.2f}%"
    )