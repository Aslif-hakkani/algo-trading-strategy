import yfinance as yf
import pandas as pd
import ta

stocks = [
    "RELIANCE.NS",
    "INFY.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS"
]

results = []

for stock in stocks:

    print(f"\nTesting {stock}...")

    df = yf.download(
        stock,
        start="2020-01-01",
        end="2025-01-01",
        auto_adjust=True,
        progress=False
    )

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

    macd = ta.trend.MACD(close=df["Close"])

    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()

    # Strategy
    df["BUY_SIGNAL"] = (
        (df["EMA50"] > df["EMA200"]) &
        (df["RSI"] > 65) &
        (df["MACD"] > df["MACD_SIGNAL"])
    )

    # 10-day hold
    df["Future_Close"] = df["Close"].shift(-10)

    df["Return_%"] = (
        (df["Future_Close"] - df["Close"])
        / df["Close"]
    ) * 100

    trades = df[df["BUY_SIGNAL"]].copy()

    if len(trades) == 0:
        continue

    avg_return = trades["Return_%"].mean()

    win_rate = (
        (trades["Return_%"] > 0).mean()
    ) * 100

    results.append([
        stock,
        len(trades),
        round(avg_return, 2),
        round(win_rate, 2)
    ])

# Final Results
results_df = pd.DataFrame(
    results,
    columns=[
        "Stock",
        "Signals",
        "Avg Return %",
        "Win Rate %"
    ]
)

print("\n")
print("=" * 60)
print(results_df)
print("=" * 60)