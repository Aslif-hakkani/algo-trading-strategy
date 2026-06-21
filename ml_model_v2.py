import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# STEP 1 — Load Data
# ============================================================

print("=" * 52)
print("   HYBRID ALGO TRADING — ML MODEL v2 (Fixed)")
print("=" * 52)

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
print(f"    Rows: {len(df)} | {df.index[0].date()} → {df.index[-1].date()}")

# ============================================================
# STEP 2 — Features
# ============================================================

print("\n[2] Building features...")

# Trend
df["EMA20"]  = ta.trend.EMAIndicator(df["Close"], 20).ema_indicator()
df["EMA50"]  = ta.trend.EMAIndicator(df["Close"], 50).ema_indicator()
df["EMA200"] = ta.trend.EMAIndicator(df["Close"], 200).ema_indicator()

# Momentum
df["RSI14"] = ta.momentum.RSIIndicator(df["Close"], 14).rsi()
df["RSI7"]  = ta.momentum.RSIIndicator(df["Close"], 7).rsi()

stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"])
df["STOCH_K"] = stoch.stoch()
df["STOCH_D"] = stoch.stoch_signal()

# MACD
macd = ta.trend.MACD(df["Close"])
df["MACD"]      = macd.macd()
df["MACD_SIG"]  = macd.macd_signal()
df["MACD_HIST"] = macd.macd_diff()

# Bollinger Bands
bb = ta.volatility.BollingerBands(df["Close"])
df["BB_HIGH"]  = bb.bollinger_hband()
df["BB_LOW"]   = bb.bollinger_lband()
df["BB_WIDTH"] = (df["BB_HIGH"] - df["BB_LOW"]) / df["Close"] * 100
df["BB_POS"]   = (df["Close"]  - df["BB_LOW"])  / (df["BB_HIGH"] - df["BB_LOW"])

# ATR
df["ATR"] = ta.volatility.AverageTrueRange(
    df["High"], df["Low"], df["Close"]
).average_true_range()

# Volume
df["Vol_MA20"]  = df["Volume"].rolling(20).mean()
df["Vol_Ratio"] = df["Volume"] / df["Vol_MA20"]

# Price action
df["Ret_1d"]  = df["Close"].pct_change(1)  * 100
df["Ret_3d"]  = df["Close"].pct_change(3)  * 100
df["Ret_5d"]  = df["Close"].pct_change(5)  * 100

# Derived
df["EMA_Gap"]     = (df["EMA50"]  - df["EMA200"]) / df["EMA200"] * 100
df["Price_EMA50"] = (df["Close"]  - df["EMA50"])  / df["EMA50"]  * 100
df["EMA20_50"]    = (df["EMA20"]  - df["EMA50"])  / df["EMA50"]  * 100

# ---- FIX 1: Bear Market Filter ----
# Strong uptrend: EMA50 > EMA200 AND gap > 1%
df["BULL_MARKET"] = (
    (df["EMA50"] > df["EMA200"]) &
    (df["EMA_Gap"] > 1.0)
).astype(int)

df.dropna(inplace=True)

FEATURES = [
    "EMA20", "EMA50", "EMA200",
    "RSI14", "RSI7",
    "STOCH_K", "STOCH_D",
    "MACD", "MACD_SIG", "MACD_HIST",
    "BB_WIDTH", "BB_POS",
    "ATR", "Vol_Ratio",
    "Ret_1d", "Ret_3d", "Ret_5d",
    "EMA_Gap", "Price_EMA50", "EMA20_50",
    "BULL_MARKET"
]

print(f"    {len(FEATURES)} features ready")

# ============================================================
# STEP 3 — FIX 2: Simpler Target (5-day direction)
# ============================================================

print("\n[3] Creating target (simpler)...")

HOLD_DAYS = 5   # 10 → 5 days (easier to predict)

df["Future_Close"]  = df["Close"].shift(-HOLD_DAYS)
df["Future_Return"] = (
    (df["Future_Close"] - df["Close"]) / df["Close"]
) * 100

# Target: price goes UP in 5 days (any positive return)
df["TARGET"] = (df["Future_Return"] > 0).astype(int)
df.dropna(inplace=True)

pos = df["TARGET"].sum()
neg = len(df) - pos
print(f"    Target=1 (UP): {pos} ({pos/len(df)*100:.1f}%)")
print(f"    Target=0 (DOWN): {neg} ({neg/len(df)*100:.1f}%)")

# ============================================================
# STEP 4 — FIX 3: Better Train/Test Split
# ============================================================

print("\n[4] Splitting data...")

# Use 70% train (more bull market data)
split = int(len(df) * 0.75)

X = df[FEATURES]
y = df["TARGET"]

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print(f"    Train: {len(X_train)} rows | {X_train.index[0].date()} → {X_train.index[-1].date()}")
print(f"    Test : {len(X_test)} rows  | {X_test.index[0].date()} → {X_test.index[-1].date()}")

# ============================================================
# STEP 5 — XGBoost (Better Params)
# ============================================================

print("\n[5] Training XGBoost v2...")

model = xgb.XGBClassifier(
    n_estimators     = 300,
    max_depth        = 3,        # 4 → 3 (less overfit)
    learning_rate    = 0.03,     # 0.05 → 0.03 (slower, better)
    subsample        = 0.7,
    colsample_bytree = 0.7,
    min_child_weight = 5,        # NEW: prevent overfit
    gamma            = 1,        # NEW: regularization
    reg_alpha        = 0.1,      # NEW: L1 regularization
    reg_lambda       = 1.5,      # NEW: L2 regularization
    scale_pos_weight = neg/pos,
    random_state     = 42,
    eval_metric      = "logloss",
    verbosity        = 0
)

model.fit(X_train, y_train, verbose=False)
print("    Done!")

# ============================================================
# STEP 6 — Evaluate
# ============================================================

print("\n[6] Evaluating...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
acc    = accuracy_score(y_test, y_pred)

print(f"\n    Accuracy: {acc*100:.1f}%")
print(f"\n{classification_report(y_test, y_pred, target_names=['DOWN(0)','UP(1)'])}")

# ============================================================
# STEP 7 — Hybrid Signal (FIX 4: Threshold 65%)
# ============================================================

print("\n[7] Applying Hybrid Strategy...")

test_df = df.iloc[split:].copy()
test_df["ML_PROB"] = y_prob

PROB_THRESHOLD = 0.65   # 60% → 65% (more selective)

# Hybrid: Bull Market + ML confident + MACD confirm
test_df["HYBRID_SIGNAL"] = (
    (test_df["BULL_MARKET"] == 1) &       # Strong uptrend only
    (test_df["ML_PROB"] > PROB_THRESHOLD) & # ML confident
    (test_df["MACD"] > test_df["MACD_SIG"])  # MACD confirm
).astype(int)

# Traditional (for compare)
test_df["TRAD_SIGNAL"] = (
    (test_df["EMA50"] > test_df["EMA200"]) &
    (test_df["RSI14"] > 55) &
    (test_df["MACD"] > test_df["MACD_SIG"])
).astype(int)

hybrid_trades = test_df[test_df["HYBRID_SIGNAL"] == 1]
trad_trades   = test_df[test_df["TRAD_SIGNAL"]   == 1]

def stats(trades):
    if len(trades) == 0:
        return 0, 0
    wr  = (trades["Future_Return"] > 0).mean() * 100
    avg = trades["Future_Return"].mean()
    return wr, avg

h_wr, h_avg = stats(hybrid_trades)
t_wr, t_avg = stats(trad_trades)

# ============================================================
# STEP 8 — Cumulative Returns
# ============================================================

test_df["Daily_Ret"] = test_df["Close"].pct_change()

test_df["Hybrid_Ret"] = test_df["Daily_Ret"].where(
    test_df["HYBRID_SIGNAL"].shift(1) == 1, 0
)
test_df["Trad_Ret"] = test_df["Daily_Ret"].where(
    test_df["TRAD_SIGNAL"].shift(1) == 1, 0
)

test_df["Cum_Market"] = (1 + test_df["Daily_Ret"]).cumprod()
test_df["Cum_Hybrid"] = (1 + test_df["Hybrid_Ret"]).cumprod()
test_df["Cum_Trad"]   = (1 + test_df["Trad_Ret"]).cumprod()

# ============================================================
# STEP 9 — Charts
# ============================================================

print("\n[8] Generating charts...")

fig = plt.figure(figsize=(16, 16))
fig.suptitle(
    "RELIANCE.NS — Hybrid ML Strategy v2",
    fontsize=16, fontweight="bold", y=0.99
)
gs = gridspec.GridSpec(4, 2, hspace=0.5, wspace=0.35)

# Chart 1: Price + Signals
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(test_df.index, test_df["Close"],
         color="#2196F3", lw=1.2, label="Price", alpha=0.8)
ax1.plot(test_df.index, test_df["EMA50"],
         color="#FF9800", lw=1.2, ls="--", label="EMA50")
ax1.plot(test_df.index, test_df["EMA200"],
         color="#F44336", lw=1.2, ls="--", label="EMA200")

# Bull market background
ax1.fill_between(
    test_df.index, test_df["Close"].min(), test_df["Close"].max(),
    where=(test_df["BULL_MARKET"] == 1),
    alpha=0.06, color="#4CAF50", label="Bull Market Zone"
)

t_pts = test_df[test_df["TRAD_SIGNAL"] == 1]
ax1.scatter(t_pts.index, t_pts["Close"],
            marker="^", color="#FF9800", s=60,
            alpha=0.6, label=f"Traditional ({len(t_pts)})", zorder=4)

h_pts = test_df[test_df["HYBRID_SIGNAL"] == 1]
ax1.scatter(h_pts.index, h_pts["Close"],
            marker="^", color="#00C853", s=100,
            zorder=5, label=f"Hybrid ML ({len(h_pts)})")

ax1.set_title("Price + Signals (Green zone = Bull Market)", fontsize=12)
ax1.set_ylabel("Price (Rs)")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Chart 2: ML Probability
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(test_df.index, test_df["ML_PROB"],
         color="#9C27B0", lw=1, label="ML Probability")
ax2.axhline(y=PROB_THRESHOLD, color="#F44336",
            ls="--", lw=1.5, label=f"Threshold = {PROB_THRESHOLD:.0%}")
ax2.fill_between(
    test_df.index, test_df["ML_PROB"], PROB_THRESHOLD,
    where=(test_df["ML_PROB"] > PROB_THRESHOLD),
    alpha=0.2, color="#00C853", label="High Confidence Zone"
)
ax2.set_title(f"ML Probability — Trade only when > {PROB_THRESHOLD:.0%}", fontsize=12)
ax2.set_ylabel("Probability")
ax2.set_ylim(0, 1)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Chart 3: Returns
ax3 = fig.add_subplot(gs[2, :])
ax3.plot(test_df.index, test_df["Cum_Market"],
         color="#2196F3", lw=2, label="Buy & Hold")
ax3.plot(test_df.index, test_df["Cum_Trad"],
         color="#FF9800", lw=1.5, ls="--",
         label=f"Traditional ({len(trad_trades)} trades, {t_wr:.0f}% WR)")
ax3.plot(test_df.index, test_df["Cum_Hybrid"],
         color="#00C853", lw=2,
         label=f"Hybrid ML ({len(hybrid_trades)} trades, {h_wr:.0f}% WR)")
ax3.axhline(y=1.0, color="gray", ls=":", lw=1)
ax3.set_title("Cumulative Returns", fontsize=12)
ax3.set_ylabel("Growth")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Chart 4: Feature Importance
ax4 = fig.add_subplot(gs[3, 0])
imp = pd.Series(model.feature_importances_, index=FEATURES)
imp = imp.sort_values(ascending=True).tail(12)
ax4.barh(imp.index, imp.values, color="#2196F3", alpha=0.8)
ax4.set_title("Top Feature Importance", fontsize=12)
ax4.set_xlabel("Score")
ax4.grid(True, alpha=0.3, axis="x")

# Chart 5: Confusion Matrix
ax5 = fig.add_subplot(gs[3, 1])
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["DOWN(0)", "UP(1)"])
disp.plot(ax=ax5, colorbar=False, cmap="Blues")
ax5.set_title("Confusion Matrix", fontsize=12)

plt.savefig("chart3_ml_model_v2.png", dpi=150, bbox_inches="tight")
print("    Saved: chart3_ml_model_v2.png")

# ============================================================
# STEP 10 — Summary
# ============================================================

mkt = test_df["Cum_Market"].iloc[-1]
hyb = test_df["Cum_Hybrid"].iloc[-1]
trd = test_df["Cum_Trad"].iloc[-1]

print("\n" + "=" * 54)
print("         FINAL COMPARISON REPORT (v2)")
print("=" * 54)
print(f"  {'':30s} {'Trad':>8} {'Hybrid':>9}")
print(f"  {'-'*50}")
print(f"  {'Total Signals':30s} {len(trad_trades):>8} {len(hybrid_trades):>9}")
print(f"  {'Win Rate (5-day)':30s} {t_wr:>7.1f}% {h_wr:>8.1f}%")
print(f"  {'Avg Return':30s} {t_avg:>7.2f}% {h_avg:>8.2f}%")
print(f"  {'Cumulative Return':30s} {(trd-1)*100:>7.1f}% {(hyb-1)*100:>8.1f}%")
print(f"  {'Market (Buy & Hold)':30s} {'':>8} {(mkt-1)*100:>8.1f}%")
print(f"  {'Model Accuracy':30s} {'':>8} {acc*100:>8.1f}%")
print("=" * 54)

top3 = pd.Series(
    model.feature_importances_, index=FEATURES
).sort_values(ascending=False).head(3)

print(f"\n  Top 3 Predictive Features:")
for i, (feat, score) in enumerate(top3.items(), 1):
    print(f"  {i}. {feat:22s} → {score:.4f}")

print(f"\n  Threshold used : {PROB_THRESHOLD:.0%}")
print(f"  Hold period    : {HOLD_DAYS} days")
print("\n  Done!")
plt.show()
