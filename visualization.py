import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ================================
# 1. Load CSV (existing data)
# ================================

print("Loading reliance.csv...")

df = pd.read_csv(
    "reliance.csv",
    skiprows=[1, 2],       # Skip Ticker + empty rows
    index_col=0,
    parse_dates=True
)

df.columns = ["Close", "High", "Low", "Open", "Volume"]
df = df.apply(pd.to_numeric, errors="coerce")
df.dropna(inplace=True)
df.index.name = "Date"

print(f"Data loaded: {len(df)} rows | {df.index[0].date()} to {df.index[-1].date()}")

# ================================
# 2. Indicators
# ================================

df["EMA50"] = ta.trend.EMAIndicator(
    close=df["Close"], window=50
).ema_indicator()

df["EMA200"] = ta.trend.EMAIndicator(
    close=df["Close"], window=200
).ema_indicator()

df["RSI"] = ta.momentum.RSIIndicator(
    close=df["Close"], window=14
).rsi()

macd_obj = ta.trend.MACD(close=df["Close"])
df["MACD"]        = macd_obj.macd()
df["MACD_SIGNAL"] = macd_obj.macd_signal()

df.dropna(inplace=True)

# ================================
# 3. Buy Signals
# ================================

df["BUY_SIGNAL"] = (
    (df["EMA50"]  > df["EMA200"]) &
    (df["RSI"]    > 55) &
    (df["MACD"]   > df["MACD_SIGNAL"])
)

# ================================
# 4. Returns
# ================================

df["Future_Close"]  = df["Close"].shift(-10)
df["Return_%"]      = ((df["Future_Close"] - df["Close"]) / df["Close"]) * 100
df["Net_Return_%"]  = df["Return_%"] - 0.20

trades = df[df["BUY_SIGNAL"]].copy()

df["Daily_Return"]       = df["Close"].pct_change()
df["Strategy_Return"]    = df["Daily_Return"].where(df["BUY_SIGNAL"].shift(1), 0)
df["Cumulative_Market"]  = (1 + df["Daily_Return"]).cumprod()
df["Cumulative_Strategy"]= (1 + df["Strategy_Return"]).cumprod()

print(f"Total BUY Signals : {len(trades)}")
print(f"Win Rate          : {(trades['Net_Return_%'] > 0).mean()*100:.1f}%")
print(f"Avg Net Return    : {trades['Net_Return_%'].mean():.2f}%")

# ================================
# 5. CHART 1 — Main Analysis
# ================================

fig = plt.figure(figsize=(16, 14))
fig.suptitle(
    "RELIANCE.NS — Algo Trading Strategy Analysis",
    fontsize=16, fontweight="bold", y=0.98
)
gs = gridspec.GridSpec(4, 1, hspace=0.45)

# --- Price + Signals ---
ax1 = fig.add_subplot(gs[0:2])
ax1.plot(df.index, df["Close"],  color="#2196F3", lw=1.2, label="Close Price")
ax1.plot(df.index, df["EMA50"],  color="#FF9800", lw=1.5, ls="--", label="EMA 50")
ax1.plot(df.index, df["EMA200"], color="#F44336", lw=1.5, ls="--", label="EMA 200")
if len(trades) > 0:
    ax1.scatter(trades.index, trades["Close"],
                marker="^", color="#00C853", s=80, zorder=5, label="BUY Signal")
ax1.set_title("Price + EMA50 + EMA200 + BUY Signals", fontsize=12)
ax1.set_ylabel("Price (Rs)")
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, alpha=0.3)

# --- RSI ---
ax2 = fig.add_subplot(gs[2])
ax2.plot(df.index, df["RSI"], color="#9C27B0", lw=1.2, label="RSI (14)")
ax2.axhline(y=55, color="#2196F3", ls="--", lw=1, label="RSI=55 (Buy Level)")
ax2.axhline(y=70, color="#F44336", ls=":",  lw=1, label="RSI=70 (Overbought)")
ax2.axhline(y=30, color="#4CAF50", ls=":",  lw=1, label="RSI=30 (Oversold)")
ax2.fill_between(df.index, df["RSI"], 55,
                 where=(df["RSI"] > 55), alpha=0.12, color="#9C27B0")
ax2.set_ylim(0, 100)
ax2.set_title("RSI Indicator", fontsize=12)
ax2.set_ylabel("RSI")
ax2.legend(loc="upper left", fontsize=8)
ax2.grid(True, alpha=0.3)

# --- Cumulative Returns ---
ax3 = fig.add_subplot(gs[3])
ax3.plot(df.index, df["Cumulative_Market"],   color="#2196F3", lw=1.8, label="Buy & Hold")
ax3.plot(df.index, df["Cumulative_Strategy"], color="#00C853", lw=1.8, label="Our Strategy")
ax3.fill_between(
    df.index, df["Cumulative_Strategy"], df["Cumulative_Market"],
    where=(df["Cumulative_Strategy"] >= df["Cumulative_Market"]),
    alpha=0.15, color="#00C853"
)
ax3.set_title("Cumulative Returns — Strategy vs Buy & Hold", fontsize=12)
ax3.set_ylabel("Growth (1x = no change)")
ax3.legend(loc="upper left", fontsize=9)
ax3.grid(True, alpha=0.3)

plt.savefig("chart1_main_analysis.png", dpi=150, bbox_inches="tight")
print("\nSaved: chart1_main_analysis.png")

# ================================
# 6. CHART 2 — Performance
# ================================

fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle("Trade Performance Summary", fontsize=14, fontweight="bold")

# Pie chart
wins   = int((trades["Net_Return_%"] > 0).sum())
losses = int((trades["Net_Return_%"] <= 0).sum())

if wins + losses > 0:
    axes[0].pie(
        [wins, losses],
        labels=[f"Wins\n({wins})", f"Losses\n({losses})"],
        colors=["#00C853", "#F44336"],
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 11}
    )
else:
    axes[0].text(0.5, 0.5, "No trades found",
                 ha="center", va="center", fontsize=12)
axes[0].set_title("Win / Loss Ratio", fontsize=12)

# Histogram
clean = trades["Net_Return_%"].dropna()
if len(clean) > 0:
    axes[1].hist(clean, bins=20, color="#2196F3", edgecolor="white", alpha=0.8)
    axes[1].axvline(x=0, color="#F44336", ls="--", lw=2, label="Break Even")
    axes[1].axvline(x=clean.mean(), color="#00C853", ls="--", lw=2,
                    label=f"Avg: {clean.mean():.2f}%")
    axes[1].legend(fontsize=9)
else:
    axes[1].text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
axes[1].set_title("Return Distribution (10-day hold)", fontsize=12)
axes[1].set_xlabel("Return %")
axes[1].set_ylabel("Number of Trades")
axes[1].grid(True, alpha=0.3)

# Monthly bar
if len(trades) > 0:
    trades_copy = trades.copy()
    trades_copy.index = pd.to_datetime(trades_copy.index)
    monthly = trades_copy.resample("ME").size()
    axes[2].bar(monthly.index, monthly.values,
                color="#9C27B0", alpha=0.8, width=20)
    axes[2].tick_params(axis="x", rotation=45)
else:
    axes[2].text(0.5, 0.5, "No trades", ha="center", va="center", fontsize=12)
axes[2].set_title("BUY Signals per Month", fontsize=12)
axes[2].set_xlabel("Month")
axes[2].set_ylabel("Count")
axes[2].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("chart2_trade_performance.png", dpi=150, bbox_inches="tight")
print("Saved: chart2_trade_performance.png")

# ================================
# 7. Summary
# ================================

mkt_f  = df["Cumulative_Market"].iloc[-1]
strt_f = df["Cumulative_Strategy"].iloc[-1]

print("\n" + "=" * 45)
print("       STRATEGY SUMMARY REPORT")
print("=" * 45)
print(f"  Stock          : RELIANCE.NS")
print(f"  Period         : {df.index[0].date()} to {df.index[-1].date()}")
print(f"  Total Signals  : {len(trades)}")
if len(trades) > 0:
    print(f"  Win Rate       : {(trades['Net_Return_%']>0).mean()*100:.1f}%")
    print(f"  Avg Net Return : {trades['Net_Return_%'].mean():.2f}%")
    w = trades[trades['Net_Return_%']>0]['Net_Return_%']
    l = trades[trades['Net_Return_%']<=0]['Net_Return_%']
    if len(w): print(f"  Avg Win        : {w.mean():.2f}%")
    if len(l): print(f"  Avg Loss       : {l.mean():.2f}%")
print(f"  Market Return  : {(mkt_f-1)*100:.1f}%")
print(f"  Strategy Return: {(strt_f-1)*100:.1f}%")
print("=" * 45)

plt.show()
print("\nDone! Both charts saved successfully.")