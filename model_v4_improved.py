import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

print("=" * 58)
print("   HYBRID ALGO MODEL v4 — IMPROVED FOR DEMO TRADING")
print("=" * 58)

# ============================================================
# CONFIG
# ============================================================

STOCKS = {
    "RELIANCE.NS": "Reliance",
    "TCS.NS"     : "TCS",
    "INFY.NS"    : "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "WIPRO.NS"   : "Wipro",
}

STOP_LOSS   = 0.02   # 2%  stop loss
TAKE_PROFIT = 0.04   # 4%  take profit  (1:2 risk-reward)
HOLD_DAYS   = 10     # max hold period
THRESHOLD   = 0.60   # ML confidence threshold

# ============================================================
# STEP 1 — Download All 5 Stocks
# ============================================================

print("\n[STEP 1] Downloading 5 stocks...")

import yfinance as yf

all_dfs = []

for ticker, name in STOCKS.items():
    try:
        df = yf.download(
            ticker,
            start="2018-01-01",
            end="2024-12-31",
            auto_adjust=True,
            progress=False
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)

        if len(df) > 200:
            df["Ticker"] = ticker
            df["Name"]   = name
            all_dfs.append(df)
            print(f"    {name:20s} → {len(df)} rows ✅")
        else:
            print(f"    {name:20s} → Too few rows ❌")

    except Exception as e:
        print(f"    {name:20s} → Failed: {e} ❌")

print(f"\n    Total stocks loaded: {len(all_dfs)}")

# ============================================================
# STEP 2 — Feature Engineering (per stock)
# ============================================================

print("\n[STEP 2] Adding features to each stock...")

def add_features(df):
    d = df.copy()

    # Trend
    d["EMA20"]  = ta.trend.EMAIndicator(d["Close"], 20).ema_indicator()
    d["EMA50"]  = ta.trend.EMAIndicator(d["Close"], 50).ema_indicator()
    d["EMA200"] = ta.trend.EMAIndicator(d["Close"], 200).ema_indicator()

    # Momentum
    d["RSI14"]  = ta.momentum.RSIIndicator(d["Close"], 14).rsi()
    d["RSI7"]   = ta.momentum.RSIIndicator(d["Close"], 7).rsi()
    d["RSI21"]  = ta.momentum.RSIIndicator(d["Close"], 21).rsi()

    st = ta.momentum.StochasticOscillator(d["High"], d["Low"], d["Close"])
    d["STOCH_K"] = st.stoch()
    d["STOCH_D"] = st.stoch_signal()

    # MACD
    m = ta.trend.MACD(d["Close"])
    d["MACD"]      = m.macd()
    d["MACD_SIG"]  = m.macd_signal()
    d["MACD_HIST"] = m.macd_diff()
    # Normalise MACD by price
    d["MACD_N"]    = d["MACD"] / d["Close"] * 100

    # Bollinger
    bb = ta.volatility.BollingerBands(d["Close"])
    d["BB_HIGH"]  = bb.bollinger_hband()
    d["BB_LOW"]   = bb.bollinger_lband()
    d["BB_WIDTH"] = (d["BB_HIGH"] - d["BB_LOW"]) / d["Close"] * 100
    d["BB_POS"]   = (d["Close"]  - d["BB_LOW"]) / \
                    (d["BB_HIGH"] - d["BB_LOW"] + 1e-9)

    # ATR
    atr = ta.volatility.AverageTrueRange(
        d["High"], d["Low"], d["Close"])
    d["ATR"]     = atr.average_true_range()
    d["ATR_PCT"] = d["ATR"] / d["Close"] * 100

    # Volume
    d["Vol_MA20"]  = d["Volume"].rolling(20).mean()
    d["Vol_Ratio"] = d["Volume"] / (d["Vol_MA20"] + 1)
    d["Vol_Trend"] = d["Vol_MA20"].pct_change(5) * 100

    # Price returns
    d["Ret_1d"]  = d["Close"].pct_change(1)  * 100
    d["Ret_3d"]  = d["Close"].pct_change(3)  * 100
    d["Ret_5d"]  = d["Close"].pct_change(5)  * 100
    d["Ret_10d"] = d["Close"].pct_change(10) * 100

    # Normalised EMA gaps (no price leakage)
    d["EMA_Gap"]   = (d["EMA50"]  - d["EMA200"]) / d["EMA200"] * 100
    d["P_EMA50"]   = (d["Close"]  - d["EMA50"])  / d["EMA50"]  * 100
    d["EMA20_50"]  = (d["EMA20"]  - d["EMA50"])  / d["EMA50"]  * 100

    # Candlestick body
    d["Body_Size"] = abs(d["Close"] - d["Open"]) / d["Open"] * 100
    d["Upper_Wick"] = (d["High"] - d[["Close","Open"]].max(axis=1)) / d["Open"] * 100
    d["Lower_Wick"] = (d[["Close","Open"]].min(axis=1) - d["Low"]) / d["Open"] * 100

    # Market regime
    d["BULL"]   = (d["EMA50"]  > d["EMA200"]).astype(int)
    d["STRONG"] = ((d["EMA50"] > d["EMA200"]) &
                   (d["EMA_Gap"] > 1.0)).astype(int)

    return d


feat_dfs = []
for df in all_dfs:
    try:
        fdf = add_features(df)
        feat_dfs.append(fdf)
    except Exception as e:
        print(f"    Feature error: {e}")

print(f"    Features added to {len(feat_dfs)} stocks")

# ============================================================
# STEP 3 — IMPROVED TARGET (Stop Loss + Take Profit)
# ============================================================

print(f"\n[STEP 3] Creating Risk-Reward target...")
print(f"    Stop Loss   : -{STOP_LOSS*100:.0f}%")
print(f"    Take Profit :  +{TAKE_PROFIT*100:.0f}%")
print(f"    Max Hold    :  {HOLD_DAYS} days")
print(f"    Risk-Reward :  1:{int(TAKE_PROFIT/STOP_LOSS)}")

def create_sl_tp_target(df, sl=STOP_LOSS, tp=TAKE_PROFIT, hold=HOLD_DAYS):
    """
    For each day, simulate forward:
    - If price hits +TP first  → TARGET = 1 (Win)
    - If price hits -SL first  → TARGET = 0 (Loss)
    - If neither in hold days  → TARGET based on final direction
    """
    targets = []
    closes  = df["Close"].values
    highs   = df["High"].values
    lows    = df["Low"].values
    n       = len(closes)

    for i in range(n):
        entry = closes[i]
        tp_price = entry * (1 + tp)
        sl_price = entry * (1 - sl)
        result   = -1  # unknown

        for j in range(1, min(hold + 1, n - i)):
            h = highs[i + j]
            l = lows[i + j]

            if h >= tp_price:
                result = 1   # Take profit hit → WIN
                break
            if l <= sl_price:
                result = 0   # Stop loss hit  → LOSS
                break

        if result == -1:
            # Neither hit — use final close
            final = closes[min(i + hold, n - 1)]
            result = 1 if final > entry else 0

        targets.append(result)

    return pd.Series(targets, index=df.index)


all_labeled = []

for df in feat_dfs:
    df = df.copy()
    df["TARGET"] = create_sl_tp_target(df)
    all_labeled.append(df)
    name = df["Name"].iloc[0]
    wr   = df["TARGET"].mean() * 100
    print(f"    {name:20s} → Win opp: {wr:.1f}%")

# ============================================================
# STEP 4 — Combine All Stocks
# ============================================================

print("\n[STEP 4] Combining all stocks...")

FEATURES = [
    "RSI14", "RSI7", "RSI21",
    "STOCH_K", "STOCH_D",
    "MACD_N", "MACD_HIST",
    "BB_WIDTH", "BB_POS",
    "ATR_PCT", "Vol_Ratio", "Vol_Trend",
    "Ret_1d", "Ret_3d", "Ret_5d", "Ret_10d",
    "EMA_Gap", "P_EMA50", "EMA20_50",
    "Body_Size", "Upper_Wick", "Lower_Wick",
    "BULL", "STRONG"
]

combined = pd.concat(all_labeled, axis=0)
combined = combined.dropna(subset=FEATURES + ["TARGET"])
combined = combined.sort_index()

print(f"    Total rows  : {len(combined)}")
print(f"    Total features: {len(FEATURES)}")
print(f"    Overall Win% : {combined['TARGET'].mean()*100:.1f}%")
print(f"    Date range  : {combined.index.min().date()} → {combined.index.max().date()}")

# ============================================================
# STEP 5 — Train / Test Split (time-based)
# ============================================================

print("\n[STEP 5] Splitting data (train: 2018-2022, test: 2023-2024)...")

train = combined[combined.index <= "2022-12-31"]
test  = combined[combined.index >  "2022-12-31"]

X_train = train[FEATURES]
y_train = train["TARGET"]
X_test  = test[FEATURES]
y_test  = test["TARGET"]

pos = int(y_train.sum())
neg = int(len(y_train) - pos)

print(f"    Train: {len(X_train)} rows | Win: {y_train.mean()*100:.1f}%")
print(f"    Test : {len(X_test)} rows  | Win: {y_test.mean()*100:.1f}%")

# ============================================================
# STEP 6 — XGBoost v4 (Better Regularisation)
# ============================================================

print("\n[STEP 6] Training XGBoost v4...")

base_model = xgb.XGBClassifier(
    n_estimators     = 500,
    max_depth        = 3,
    learning_rate    = 0.02,    # Slower learning
    subsample        = 0.7,
    colsample_bytree = 0.6,     # Use 60% features per tree
    min_child_weight = 10,      # More conservative splits
    gamma            = 2,       # Pruning threshold
    reg_alpha        = 0.5,     # L1
    reg_lambda       = 2.0,     # L2
    scale_pos_weight = neg / max(pos, 1),
    random_state     = 42,
    eval_metric      = "logloss",
    verbosity        = 0
)

# Calibrate probabilities (more reliable % estimates)
model = CalibratedClassifierCV(base_model, cv=3, method="isotonic")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
acc    = accuracy_score(y_test, y_pred)

# Train accuracy (overfit check)
tr_acc = accuracy_score(y_train, model.predict(X_train))

print(f"\n    Train Accuracy : {tr_acc*100:.1f}%")
print(f"    Test  Accuracy : {acc*100:.1f}%")
print(f"    Overfit Gap    : {(tr_acc-acc)*100:.1f}% (target < 10%)")
print(f"\n{classification_report(y_test, y_pred, target_names=['LOSS(0)','WIN(1)'])}")

# ============================================================
# STEP 7 — Walk-Forward Validation
# ============================================================

print("\n[STEP 7] Walk-Forward Validation...")

# Use RELIANCE only for WF (consistent timeline)
rel_df = [df for df in all_labeled if "RELIANCE" in df["Ticker"].iloc[0]][0]
rel_df = rel_df.dropna(subset=FEATURES + ["TARGET"])

X_rel = rel_df[FEATURES]
y_rel = rel_df["TARGET"]

tscv = TimeSeriesSplit(n_splits=5, test_size=60)
wf_accs = []

for fold, (tr_i, te_i) in enumerate(tscv.split(X_rel), 1):
    X_tr_, X_te_ = X_rel.iloc[tr_i], X_rel.iloc[te_i]
    y_tr_, y_te_ = y_rel.iloc[tr_i], y_rel.iloc[te_i]

    m_ = xgb.XGBClassifier(
        n_estimators=300, max_depth=3,
        learning_rate=0.02, subsample=0.7,
        colsample_bytree=0.6, min_child_weight=10,
        gamma=2, reg_alpha=0.5, reg_lambda=2.0,
        scale_pos_weight=(len(y_tr_)-y_tr_.sum())/max(y_tr_.sum(),1),
        random_state=42, eval_metric="logloss", verbosity=0
    )
    m_.fit(X_tr_, y_tr_, verbose=False)
    fold_acc = accuracy_score(y_te_, m_.predict(X_te_))
    wf_accs.append(fold_acc)

    s = rel_df.index[te_i[0]].strftime("%Y-%m")
    e = rel_df.index[te_i[-1]].strftime("%Y-%m")
    bar  = "█" * int(fold_acc * 30)
    sign = "✅" if fold_acc >= 0.50 else "❌"
    print(f"    Fold {fold} | {s}→{e} | {fold_acc*100:.1f}% {bar} {sign}")

avg_wf = np.mean(wf_accs)
print(f"\n    Walk-Forward Average: {avg_wf*100:.1f}%")

# ============================================================
# STEP 8 — Backtest with Stop Loss on RELIANCE 2023-2024
# ============================================================

print("\n[STEP 8] Backtesting with Stop Loss on RELIANCE 2023-2024...")

bt_df = rel_df[rel_df.index > "2022-12-31"].copy()
bt_X  = bt_df[FEATURES]
bt_df["ML_PROB"] = model.predict_proba(bt_X)[:, 1]

bt_df["HYBRID"] = (
    (bt_df["ML_PROB"]   > THRESHOLD) &
    (bt_df["MACD"]      > bt_df["MACD_SIG"]) &
    (bt_df["RSI14"]     > 45) &
    (bt_df["STRONG"]    == 1)
).astype(int)

bt_df["TRAD"] = (
    (bt_df["BULL"]   == 1) &
    (bt_df["RSI14"]  > 55) &
    (bt_df["MACD"]   > bt_df["MACD_SIG"])
).astype(int)

# Simulate trades with stop loss
def simulate_trades(df, signal_col, sl=STOP_LOSS, tp=TAKE_PROFIT, hold=HOLD_DAYS):
    results = []
    signals = df[df[signal_col] == 1]
    closes  = df["Close"].values
    highs   = df["High"].values
    lows    = df["Low"].values
    idx     = df.index

    for date in signals.index:
        pos_i = df.index.get_loc(date)
        entry = closes[pos_i]
        tp_p  = entry * (1 + tp)
        sl_p  = entry * (1 - sl)
        exit_ret  = 0
        exit_type = "timeout"

        for j in range(1, min(hold + 1, len(closes) - pos_i)):
            h = highs[pos_i + j]
            l = lows[pos_i + j]
            if h >= tp_p:
                exit_ret  = tp
                exit_type = "TP"
                break
            if l <= sl_p:
                exit_ret  = -sl
                exit_type = "SL"
                break

        if exit_type == "timeout":
            final_i  = min(pos_i + hold, len(closes) - 1)
            exit_ret = (closes[final_i] - entry) / entry

        results.append({
            "entry_date": date,
            "entry_price": entry,
            "return"    : exit_ret * 100,
            "exit_type" : exit_type,
            "win"       : exit_ret > 0
        })

    return pd.DataFrame(results)


hybrid_trades = simulate_trades(bt_df, "HYBRID")
trad_trades   = simulate_trades(bt_df, "TRAD")

def summary(trades, name):
    if len(trades) == 0:
        return f"    {name}: No trades"
    wr  = trades["win"].mean() * 100
    avg = trades["return"].mean()
    tp_ = (trades["exit_type"] == "TP").sum()
    sl_ = (trades["exit_type"] == "SL").sum()
    to_ = (trades["exit_type"] == "timeout").sum()
    return (f"    {name:20s} | Trades:{len(trades):3d} | "
            f"WR:{wr:5.1f}% | Avg:{avg:+5.2f}% | "
            f"TP:{tp_} SL:{sl_} TO:{to_}")

print(summary(trad_trades,   "Traditional"))
print(summary(hybrid_trades, "Hybrid ML v4"))

# Cumulative equity curves
def equity_curve(trades, label, color):
    if trades.empty:
        return pd.Series([1.0], name=label)
    curve = [1.0]
    for r in trades["return"]:
        curve.append(curve[-1] * (1 + r / 100))
    return pd.Series(curve, name=label)

hyb_curve  = equity_curve(hybrid_trades, "Hybrid ML", "#3fb950")
trd_curve  = equity_curve(trad_trades,   "Traditional", "#d29922")

# ============================================================
# STEP 9 — Charts
# ============================================================

print("\n[STEP 9] Generating charts...")

fig = plt.figure(figsize=(16, 18))
fig.patch.set_facecolor("#0d1117")
fig.suptitle(
    "ALGO TRADING MODEL v4 — Improved Results",
    fontsize=16, fontweight="bold", color="#e6edf3", y=0.99
)

gs = gridspec.GridSpec(4, 2, hspace=0.55, wspace=0.35,
                       top=0.95, bottom=0.04)

DARK  = "#0d1117"
CARD  = "#161b22"
GRID  = "#21262d"
TEXT  = "#e6edf3"
MUTED = "#8b949e"

def style_ax(ax):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.grid(True, color=GRID, alpha=0.6, linewidth=0.5)

# ── Chart 1: Walk-Forward Accuracy ──────────────
ax0 = fig.add_subplot(gs[0, :])
style_ax(ax0)

folds_x  = [f"Fold {i+1}" for i in range(len(wf_accs))]
bar_cols = ["#3fb950" if a >= 0.50 else "#f85149" for a in wf_accs]
bars     = ax0.bar(folds_x, [a*100 for a in wf_accs],
                   color=bar_cols, alpha=0.85, width=0.5, zorder=3)

ax0.axhline(50, color="#f85149", ls="--", lw=1.5, label="Random (50%)", zorder=4)
ax0.axhline(avg_wf*100, color="#58a6ff", ls="-", lw=2,
            label=f"Avg: {avg_wf*100:.1f}%", zorder=4)

for bar, val in zip(bars, wf_accs):
    ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val*100:.1f}%", ha="center", va="bottom",
             color=TEXT, fontsize=11, fontweight="bold")

ax0.set_title("Walk-Forward Cross Validation — Accuracy per Fold", fontsize=12)
ax0.set_ylabel("Accuracy %", color=MUTED)
ax0.set_ylim(30, 80)
ax0.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)

# ── Chart 2: Train vs Test accuracy ─────────────
ax1 = fig.add_subplot(gs[1, 0])
style_ax(ax1)

cats = ["v3 Train", "v3 Test", "v4 Train", "v4 Test"]
vals = [89.8, 46.5, tr_acc*100, acc*100]
col_ = ["#f85149", "#f85149", "#3fb950", "#3fb950" if acc >= 0.50 else "#d29922"]

b2 = ax1.bar(cats, vals, color=col_, alpha=0.85, width=0.5, zorder=3)
ax1.axhline(50, color="#8b949e", ls="--", lw=1.2, label="Random 50%")
for bar, val in zip(b2, vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val:.1f}%", ha="center", va="bottom",
             color=TEXT, fontsize=10, fontweight="bold")
ax1.set_title("v3 vs v4 — Overfit Comparison", fontsize=12)
ax1.set_ylabel("Accuracy %", color=MUTED)
ax1.set_ylim(30, 100)
ax1.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)

# ── Chart 3: Feature Importance ─────────────────
ax2 = fig.add_subplot(gs[1, 1])
style_ax(ax2)

# Get importance from base model
base_imp = base_model.named_estimators_[0].feature_importances_ \
           if hasattr(base_model, "named_estimators_") else \
           base_model.estimator.feature_importances_ \
           if hasattr(base_model, "estimator") else \
           np.ones(len(FEATURES))

imp_s = pd.Series(base_imp, index=FEATURES).sort_values(ascending=True).tail(12)
colors_imp = plt.cm.Blues(np.linspace(0.4, 1.0, len(imp_s)))
ax2.barh(imp_s.index, imp_s.values, color=colors_imp, alpha=0.9, zorder=3)
ax2.set_title("Top Feature Importance", fontsize=12)
ax2.set_xlabel("Score", color=MUTED)

# ── Chart 4: ML Probability 2023-2024 ───────────
ax3 = fig.add_subplot(gs[2, :])
style_ax(ax3)

ax3.plot(bt_df.index, bt_df["ML_PROB"],
         color="#bc8cff", lw=1, label="ML Prob", alpha=0.8)
ax3.axhline(THRESHOLD, color="#f85149", ls="--", lw=1.5,
            label=f"Threshold {THRESHOLD*100:.0f}%")
ax3.fill_between(bt_df.index, bt_df["ML_PROB"], THRESHOLD,
                 where=(bt_df["ML_PROB"] > THRESHOLD),
                 alpha=0.25, color="#3fb950", label="Buy Zone")

if len(hybrid_trades) > 0:
    for _, row in hybrid_trades.iterrows():
        color = "#3fb950" if row["win"] else "#f85149"
        ax3.axvline(row["entry_date"], color=color, alpha=0.3, lw=0.8)

ax3.set_title(f"ML Probability — 2023-2024 Test Period (threshold {THRESHOLD*100:.0f}%)",
              fontsize=12)
ax3.set_ylabel("Probability", color=MUTED)
ax3.set_ylim(0, 1)
ax3.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)

# ── Chart 5: Equity Curves ───────────────────────
ax4 = fig.add_subplot(gs[3, :])
style_ax(ax4)

ax4.plot(range(len(hyb_curve)), hyb_curve.values,
         color="#3fb950", lw=2, label=f"Hybrid ML v4 ({len(hybrid_trades)} trades)")
ax4.plot(range(len(trd_curve)), trd_curve.values,
         color="#d29922", lw=1.5, ls="--",
         label=f"Traditional ({len(trad_trades)} trades)")
ax4.axhline(1.0, color=MUTED, ls=":", lw=1)
ax4.fill_between(range(len(hyb_curve)), hyb_curve.values, 1.0,
                 where=(np.array(hyb_curve.values) >= 1.0),
                 alpha=0.12, color="#3fb950")
ax4.fill_between(range(len(hyb_curve)), hyb_curve.values, 1.0,
                 where=(np.array(hyb_curve.values) < 1.0),
                 alpha=0.12, color="#f85149")

ax4.set_title("Equity Curve — With Stop Loss & Take Profit Simulation",
              fontsize=12)
ax4.set_ylabel("Portfolio Growth", color=MUTED)
ax4.set_xlabel("Trade Number", color=MUTED)
ax4.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)

plt.savefig("chart5_model_v4.png", dpi=150,
            bbox_inches="tight", facecolor=DARK)
print("    Saved: chart5_model_v4.png")

# ============================================================
# STEP 10 — Final Report
# ============================================================

def trade_stats(trades):
    if trades.empty:
        return dict(n=0, wr=0, avg=0, final=1.0, tp=0, sl=0)
    wr    = trades["win"].mean() * 100
    avg   = trades["return"].mean()
    final = np.prod(1 + trades["return"] / 100)
    tp_   = (trades["exit_type"] == "TP").sum()
    sl_   = (trades["exit_type"] == "SL").sum()
    return dict(n=len(trades), wr=wr, avg=avg,
                final=final, tp=tp_, sl=sl_)

h = trade_stats(hybrid_trades)
t = trade_stats(trad_trades)

print("\n" + "=" * 60)
print("            FINAL REPORT — MODEL v4")
print("=" * 60)
print(f"\n  {'METRIC':30s} {'v3':>10} {'v4':>10}")
print(f"  {'-'*52}")
print(f"  {'Train Accuracy':30s} {'89.8%':>10} {tr_acc*100:>9.1f}%")
print(f"  {'Test  Accuracy':30s} {'46.5%':>10} {acc*100:>9.1f}%")
print(f"  {'Overfit Gap':30s} {'43.3%':>10} {(tr_acc-acc)*100:>9.1f}%")
print(f"  {'Walk-Forward Avg':30s} {'46.0%':>10} {avg_wf*100:>9.1f}%")
print()
print(f"  {'':30s} {'Trad':>10} {'Hybrid v4':>10}")
print(f"  {'-'*52}")
print(f"  {'Total Trades':30s} {t['n']:>10} {h['n']:>10}")
print(f"  {'Win Rate':30s} {t['wr']:>9.1f}% {h['wr']:>9.1f}%")
print(f"  {'Avg Return/Trade':30s} {t['avg']:>+9.2f}% {h['avg']:>+9.2f}%")
print(f"  {'TP hits':30s} {t['tp']:>10} {h['tp']:>10}")
print(f"  {'SL hits':30s} {t['sl']:>10} {h['sl']:>10}")
print(f"  {'Final Portfolio':30s} {t['final']:>9.2f}x {h['final']:>9.2f}x")
print("=" * 60)

# Demo trade ready?
demo_ready = (
    acc >= 0.52 and
    h["wr"] >= 50 and
    h["n"] >= 5 and
    (tr_acc - acc) < 0.25
)

print(f"""
  Demo Trade Ready Checklist:
  ─────────────────────────────────────────────
  {"✅" if acc >= 0.52     else "⚠️"} Test Accuracy > 52%       → {acc*100:.1f}%
  {"✅" if h['wr'] >= 50   else "⚠️"} Win Rate > 50%            → {h['wr']:.1f}%
  {"✅" if h['n'] >= 5     else "⚠️"} Enough trades (5+)        → {h['n']}
  {"✅" if (tr_acc-acc)<.25 else "⚠️"} Overfit < 25%            → {(tr_acc-acc)*100:.1f}%
  {"✅" if STOP_LOSS > 0   else "⚠️"} Stop Loss implemented     → {STOP_LOSS*100:.0f}%
  ─────────────────────────────────────────────
  {"🟢 MODEL IS DEMO-READY!" if demo_ready else "🟡 More improvements needed"}
""")

plt.show()
print("Done!")
