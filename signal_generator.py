import yfinance as yf
import pandas as pd
import ta

# Download data
df = yf.download(
    "RELIANCE.NS",
    start="2020-01-01",
    end="2025-01-01",
    auto_adjust=True
)

# Flatten columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Indicators
df["EMA50"] = ta.trend.EMAIndicator(
    close=df["Close"],
    window=50
).ema_indicator()

df["EMA200"] = ta.trend.EMAIndicator(
    close=df["Close"],
    window=200
).ema_indicator()

df["RSI"] = ta.momentum.RSIIndicator(
    close=df["Close"],
    window=14
).rsi()

# Buy Signal
df["BUY_SIGNAL"] = (
    (df["EMA50"] > df["EMA200"]) &
    (df["RSI"] > 55)
)

# Show latest rows
print(
    df[
        ["Close", "EMA50", "EMA200", "RSI", "BUY_SIGNAL"]
    ].tail(20)
)

print("\nLatest Signal:")

if df["BUY_SIGNAL"].iloc[-1]:
    print("BUY")
else:
    print("NO BUY")