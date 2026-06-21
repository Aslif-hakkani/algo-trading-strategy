import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# STEP 1 — Load Data
# ============================================================

print("=" * 50)
print("   HYBRID ALGO TRADING — ML MODEL (XGBoost)")
print("=" * 50)

print("\n[1] Loading data...")

df = pd.read_csv(
    "reliance.csv",
    skiprows=[1, 2],
    index_col=0,
    parse_dates=True
)
df.columns = ["Close", "High", "Low", "Open", "Volume"]
df = df.apply(pd.to_numeric, errors="coerce")
df.dropna(inplace=True)

print(f"    Rows: {len(df)} | {df.index[0].date()} to {df.index[-1].date()}")

# ============================================================
# STEP 2 — Feature Engineering
# ============================================================

print("\n[2] Building features...")

# --- Trend Indicators ---
df["EMA20"]  = ta.trend.EMAIndicator(df["Close"], 20).ema_indicator()
df["EMA50"]  = ta.trend.EMAIndicator(df["Close"], 50).ema_indicator()
df["EMA200"] = ta.trend.EMAIndicator(df["Close"], 200).ema_indicator()

# --- Momentum ---
df["RSI"]  = ta.momentum.RSIIndicator(df["Close"], 14).rsi()
df["RSI7"] = ta.momentum.RSIIndicator(df["Close"], 7).rsi()

stoch = ta.momentum.StochasticOscillator(
    df["High"], df["Low"], df["Close"]
)
df["STOCH_K"] = stoch.stoch()
df["STOCH_D"] = stoch.stoch_signal()

# --- MACD ---
macd = ta.trend.MACD(df["Close"])
df["MACD"]        = macd.macd()
df["MACD_SIGNAL"] = macd.macd_signal()
df["MACD_HIST"]   = macd.macd_diff()

# --- Volatility ---
bb = ta.volatility.BollingerBands(df["Close"])
df["BB_HIGH"]  = bb.bollinger_hband()
df["BB_LOW"]   = bb.bollinger_lband()
df["BB_WIDTH"] = (df["BB_HIGH"] - df["BB_LOW"]) / df["Close"]
df["BB_POS"]   = (df["Close"] - df["BB_LOW"]) / (df["BB_HIGH"] - df["BB_LOW"])

df["ATR"] = ta.volatility.AverageTrueRange(
    df["High"], df["Low"], df["Close"]
).average_true_range()

# --- Volume ---
df["OBV"] = ta.volume.OnBalanceVolumeIndicator(
    df["Close"], df["Volume"]
).on_balance_volume()
df["Vol_MA20"] = df["Volume"].rolling(20).mean()
df["Vol_Ratio"] = df["Volume"] / df["Vol_MA20"]

# --- Price Action ---
df["Return_1d"]  = df["Close"].pct_change(1)  * 100
df["Return_3d"]  = df["Close"].pct_change(3)  * 100
df["Return_5d"]  = df["Close"].pct_change(5)  * 100
df["Return_10d"] = df["Close"].pct_change(10) * 100

df["EMA_Gap"]    = (df["EMA50"] - df["EMA200"]) / df["EMA200"] * 100
df["Price_EMA50"]= (df["Close"] - df["EMA50"]) / df["EMA50"]   * 100

# --- Trend Filter (Traditional) ---
df["UPTREND"] = (df["EMA50"] > df["EMA200"]).astype(int)

print(f"    Features created: 25+")

# ============================================================
# STEP 3 — Target Variable
# ============================================================

print("\n[3] Creating target...")

# Target: 10 days later price > today + 2%?
# 1 = Strong BUY opportunity
# 0 = Skip
HOLD_DAYS   = 10
MIN_PROFIT  = 2.0   # 2% minimum profit threshold

df["Future_Close"] = df["Close"].shift(-HOLD_DAYS)
df["Future_Return"] = (
    (df["Future_Close"] - df["Close"]) / df["Close"]
) * 100

df["TARGET"] = (df["Future_Return"] > MIN_PROFIT).astype(int)

df.dropna(inplace=True)

pos = df["TARGET"].sum()
neg = len(df) - pos
print(f"    Target=1 (BUY opp): {pos} ({pos/len(df)*100:.1f}%)")
print(f"    Target=0 (Skip)   : {neg} ({neg/len(df)*100:.1f}%)")

# ============================================================
# STEP 4 — Train / Test Split
# ============================================================

print("\n[4] Splitting data (80/20 time-based)...")

FEATURES = [
    "EMA20", "EMA50", "EMA200",
    "RSI", "RSI7",
    "STOCH_K", "STOCH_D",
    "MACD", "MACD_SIGNAL", "MACD_HIST",
    "BB_WIDTH", "BB_POS",
    "ATR", "Vol_Ratio",
    "Return_1d", "Return_3d", "Return_5d", "Return_10d",
    "EMA_Gap", "Price_EMA50",
    "UPTREND"
]

X = df[FEATURES]
y = df["TARGET"]

split = int(len(X) * 0.80)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"    Train: {len(X_train)} rows ({X_train.index[0].date()} - {X_train.index[-1].date()})")
print(f"    Test : {len(X_test)} rows ({X_test.index[0].date()} - {X_test.index[-1].date()})")

# ============================================================
# STEP 5 — XGBoost Model
# ============================================================

print("\n[5] Training XGBoost model...")

model = xgb.XGBClassifier(
    n_estimators     = 200,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    scale_pos_weight = neg / pos,   # Handle imbalance
    random_state     = 42,
    eval_metric      = "logloss",
    verbosity        = 0
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

print("    Training complete!")

# ============================================================
# STEP 6 — Evaluate
# ============================================================

print("\n[6] Evaluating model...")

y_pred      = model.predict(X_test)
y_prob      = model.predict_proba(X_test)[:, 1]

accuracy    = accuracy_score(y_test, y_pred)

print(f"\n    Accuracy    : {accuracy*100:.1f}%")
print(f"\n    Report:\n")
print(classification_report(y_test, y_pred,
      target_names=["Skip(0)", "BUY(1)"]))

# ============================================================
# STEP 7 — Hybrid Signal (ML + Trend Filter)
# ============================================================

print("\n[7] Applying Hybrid Strategy...")

test_df = df.iloc[split:].copy()
test_df["ML_PROB"]    = y_prob
test_df["ML_PRED"]    = y_pred

# Hybrid Rule:
#   Condition 1: Traditional Uptrend (EMA50 > EMA200)
#   Condition 2: ML Probability > 60%
PROB_THRESHOLD = 0.60

test_df["HYBRID_SIGNAL"] = (
    (test_df["UPTREND"] == 1) &
    (test_df["ML_PROB"] > PROB_THRESHOLD)
).astype(int)

# Traditional Signal (for comparison)
test_df["TRAD_SIGNAL"] = (
    (test_df["UPTREND"] == 1) &
    (test_df["RSI"] > 55) &
    (test_df["MACD"] > test_df["MACD_SIGNAL"])
).astype(int)

hybrid_trades = test_df[test_df["HYBRID_SIGNAL"] == 1]
trad_trades   = test_df[test_df["TRAD_SIGNAL"]   == 1]

def win_rate(trades_df):
    if len(trades_df) == 0:
        return 0
    return (trades_df["Future_Return"] > 0).mean() * 100

def avg_return(trades_df):
    if len(trades_df) == 0:
        return 0
    return trades_df["Future_Return"].mean()

print(f"\n    {'':30s} {'Signals':>8} {'Win Rate':>9} {'Avg Ret':>8}")
print(f"    {'-'*57}")
print(f"    {'Traditional Strategy':30s} {len(trad_trades):>8} {win_rate(trad_trades):>8.1f}% {avg_return(trad_trades):>7.2f}%")
print(f"    {'Hybrid (Trad + ML)':30s} {len(hybrid_trades):>8} {win_rate(hybrid_trades):>8.1f}% {avg_return(hybrid_trades):>7.2f}%")

# ============================================================
# STEP 8 — Cumulative Returns
# ============================================================

test_df["Daily_Return"] = test_df["Close"].pct_change()

test_df["Hybrid_Ret"]   = test_df["Daily_Return"].where(
    test_df["HYBRID_SIGNAL"].shift(1) == 1, 0
)
test_df["Trad_Ret"]     = test_df["Daily_Return"].where(
    test_df["TRAD_SIGNAL"].shift(1) == 1, 0
)

test_df["Cum_Market"]   = (1 + test_df["Daily_Return"]).cumprod()
test_df["Cum_Hybrid"]   = (1 + test_df["Hybrid_Ret"]).cumprod()
test_df["Cum_Trad"]     = (1 + test_df["Trad_Ret"]).cumprod()

# ============================================================
# STEP 9 — Charts
# ============================================================

print("\n[8] Generating charts...")

fig = plt.figure(figsize=(16, 16))
fig.suptitle(
    "RELIANCE.NS — Hybrid ML Strategy Analysis",
    fontsize=16, fontweight="bold", y=0.99
)
gs = gridspec.GridSpec(4, 2, hspace=0.5, wspace=0.35)

# --- Chart 1: Price + Signals ---
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(test_df.index, test_df["Close"],
         color="#2196F3", lw=1.2, label="Price", alpha=0.8)
ax1.plot(test_df.index, test_df["EMA50"],
         color="#FF9800", lw=1.2, ls="--", label="EMA50")
ax1.plot(test_df.index, test_df["EMA200"],
         color="#F44336", lw=1.2, ls="--", label="EMA200")

# Traditional signals
t_buy = test_df[test_df["TRAD_SIGNAL"] == 1]
ax1.scatter(t_buy.index, t_buy["Close"],
            marker="^", color="#FF9800", s=60,
            alpha=0.6, label=f"Traditional ({len(t_buy)})", zorder=4)

# Hybrid signals
h_buy = test_df[test_df["HYBRID_SIGNAL"] == 1]
ax1.scatter(h_buy.index, h_buy["Close"],
            marker="^", color="#00C853", s=90,
            zorder=5, label=f"Hybrid ML ({len(h_buy)})")

ax1.set_title("Price + Traditional vs Hybrid BUY Signals", fontsize=12)
ax1.set_ylabel("Price (Rs)")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# --- Chart 2: ML Probability ---
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(test_df.index, test_df["ML_PROB"],
         color="#9C27B0", lw=1, label="ML Probability (BUY)")
ax2.axhline(y=PROB_THRESHOLD, color="#F44336",
            ls="--", lw=1.5,
            label=f"Threshold = {PROB_THRESHOLD:.0%}")
ax2.fill_between(test_df.index, test_df["ML_PROB"], PROB_THRESHOLD,
                 where=(test_df["ML_PROB"] > PROB_THRESHOLD),
                 alpha=0.2, color="#00C853", label="Buy Zone")
ax2.set_title("XGBoost ML Probability of Price Rising 2%+", fontsize=12)
ax2.set_ylabel("Probability")
ax2.set_ylim(0, 1)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# --- Chart 3: Cumulative Returns ---
ax3 = fig.add_subplot(gs[2, :])
ax3.plot(test_df.index, test_df["Cum_Market"],
         color="#2196F3", lw=2, label="Buy & Hold")
ax3.plot(test_df.index, test_df["Cum_Trad"],
         color="#FF9800", lw=1.5, ls="--",
         label="Traditional Strategy")
ax3.plot(test_df.index, test_df["Cum_Hybrid"],
         color="#00C853", lw=2,
         label="Hybrid ML Strategy")
ax3.set_title("Cumulative Returns Comparison", fontsize=12)
ax3.set_ylabel("Growth (1x = start)")
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# --- Chart 4: Feature Importance ---
ax4 = fig.add_subplot(gs[3, 0])
importance = pd.Series(
    model.feature_importances_,
    index=FEATURES
).sort_values(ascending=True).tail(12)

bars = ax4.barh(importance.index, importance.values,
                color="#2196F3", alpha=0.8)
ax4.set_title("Top Feature Importance", fontsize=12)
ax4.set_xlabel("Importance Score")
ax4.grid(True, alpha=0.3, axis="x")

# --- Chart 5: Confusion Matrix ---
ax5 = fig.add_subplot(gs[3, 1])
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Skip(0)", "BUY(1)"]
)
disp.plot(ax=ax5, colorbar=False, cmap="Blues")
ax5.set_title("Confusion Matrix", fontsize=12)

plt.savefig("chart3_ml_model.png", dpi=150, bbox_inches="tight")
print("    Saved: chart3_ml_model.png")

# ============================================================
# STEP 10 — Final Summary
# ============================================================

mkt = test_df["Cum_Market"].iloc[-1]
hyb = test_df["Cum_Hybrid"].iloc[-1]
trd = test_df["Cum_Trad"].iloc[-1]

print("\n" + "=" * 52)
print("          FINAL COMPARISON REPORT")
print("=" * 52)
print(f"  {'':28s} {'Trad':>8} {'Hybrid':>9}")
print(f"  {'-'*48}")
print(f"  {'Total Signals':28s} {len(trad_trades):>8} {len(hybrid_trades):>9}")
print(f"  {'Win Rate':28s} {win_rate(trad_trades):>7.1f}% {win_rate(hybrid_trades):>8.1f}%")
print(f"  {'Avg Return (10d)':28s} {avg_return(trad_trades):>7.2f}% {avg_return(hybrid_trades):>8.2f}%")
print(f"  {'Cumulative Return':28s} {(trd-1)*100:>7.1f}% {(hyb-1)*100:>8.1f}%")
print(f"  {'Market (Buy&Hold)':28s} {'':>8} {(mkt-1)*100:>8.1f}%")
print(f"  {'Model Accuracy':28s} {'':>8} {accuracy*100:>8.1f}%")
print("=" * 52)

top3 = pd.Series(
    model.feature_importances_,
    index=FEATURES
).sort_values(ascending=False).head(3)

print(f"\n  Top 3 Predictive Features:")
for i, (feat, score) in enumerate(top3.items(), 1):
    print(f"  {i}. {feat:20s} → {score:.4f}")

print("\n  Done! chart3_ml_model.png saved.")

plt.show()
