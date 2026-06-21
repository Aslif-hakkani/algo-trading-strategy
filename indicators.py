import yfinance as yf
import pandas as pd
import ta

# Download Data
df = yf.download(
    "RELIANCE.NS",
    start="2020-01-01",
    end="2025-01-01",
    auto_adjust=True
)

# Flatten columns if needed
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# EMA50
df["EMA50"] = ta.trend.EMAIndicator(
    close=df["Close"],
    window=50
).ema_indicator()

# EMA200
df["EMA200"] = ta.trend.EMAIndicator(
    close=df["Close"],
    window=200
).ema_indicator()

# RSI
df["RSI"] = ta.momentum.RSIIndicator(
    close=df["Close"],
    window=14
).rsi()

print(
    df[
        ["Close", "EMA50", "EMA200", "RSI"]
    ].tail(10)
)