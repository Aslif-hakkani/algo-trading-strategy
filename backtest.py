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

# Fix columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# =========================
# Indicators
# =========================

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

macd = ta.trend.MACD(close=df["Close"])

df["MACD"] = macd.macd()
df["MACD_SIGNAL"] = macd.macd_signal()

# =========================
# Strategy
# =========================

df["BUY_SIGNAL"] = (
    (df["EMA50"] > df["EMA200"]) &
    (df["RSI"] > 65) &
    (df["MACD"] > df["MACD_SIGNAL"])
)

# =========================
# Backtest Logic
# =========================

# Hold for 10 trading days
df["Future_Close"] = df["Close"].shift(-10)

# Gross Return
df["Return_%"] = (
    (df["Future_Close"] - df["Close"])
    / df["Close"]
) * 100

# Transaction Cost (0.20%)
df["Net_Return_%"] = df["Return_%"] - 0.20

# =========================
# Trade Selection
# =========================

trades = df[df["BUY_SIGNAL"]].copy()

print("\nTotal Signals:", len(trades))

print(
    "\nNet Average Return (%):",
    round(trades["Net_Return_%"].mean(), 2)
)

# Winning / Losing Trades
wins = trades[trades["Net_Return_%"] > 0]
losses = trades[trades["Net_Return_%"] <= 0]

# Win Rate
win_rate = (len(wins) / len(trades)) * 100

print(
    "Net Win Rate (%):",
    round(win_rate, 2)
)

# Average Win / Loss
avg_win = wins["Net_Return_%"].mean()
avg_loss = losses["Net_Return_%"].mean()

# Profit Factor
profit_factor = (
    wins["Net_Return_%"].sum()
    /
    abs(losses["Net_Return_%"].sum())
)

print(f"Net Avg Win: {avg_win:.2f}%")
print(f"Net Avg Loss: {avg_loss:.2f}%")
print(f"Net Profit Factor: {profit_factor:.2f}")