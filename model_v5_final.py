import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("   HYBRID ALGO MODEL v5 — ML as Smart Filter")
print("   (Train on Traditional Signals → Filter Best)")
print("=" * 60)

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

STOP_LOSS   = 0.02    # 2%
TAKE_PROFIT = 0.04    # 4%
HOLD_DAYS   = 10
ML_THRESHOLD = 0.52   # Lower threshold (more signals)

# ============================================================
# STEP 1 — Download
# ============================================================

print("\n[1] Downloading 5 stocks...")
import yfinance as yf

raw = {}
for ticker, name in STOCKS.items():
    try:
        df = yf.download(ticker, start="2018-01-01", end="2024-12-31",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)
        if len(df) > 200:
            raw[ticker] = df
            print(f"    {name:20s} → {len(df)} rows ✅")
    except Exception as e:
        print(f"    {name:20s} → Error: {e} ❌")

# ============================================================
# STEP 2 — Features
# ============================================================

print("\n[2] Adding features...")

def add_features(df):
    d = df.copy()
    d["EMA20"]   = ta.trend.EMAIndicator(d["Close"], 20).ema_indicator()
    d["EMA50"]   = ta.trend.EMAIndicator(d["Close"], 50).ema_indicator()
    d["EMA200"]  = ta.trend.EMAIndicator(d["Close"], 200).ema_indicator()
    d["RSI14"]   = ta.momentum.RSIIndicator(d["Close"], 14).rsi()
    d["RSI7"]    = ta.momentum.RSIIndicator(d["Close"], 7).rsi()

    st = ta.momentum.StochasticOscillator(d["High"], d["Low"], d["Close"])
    d["STOCH_K"] = st.stoch()
    d["STOCH_D"] = st.stoch_signal()

    m = ta.trend.MACD(d["Close"])
    d["MACD"]      = m.macd()
    d["MACD_SIG"]  = m.macd_signal()
    d["MACD_HIST"] = m.macd_diff()
    d["MACD_N"]    = d["MACD"] / d["Close"] * 100

    bb = ta.volatility.BollingerBands(d["Close"])
    d["BB_HIGH"]  = bb.bollinger_hband()
    d["BB_LOW"]   = bb.bollinger_lband()
    d["BB_WIDTH"] = (d["BB_HIGH"] - d["BB_LOW"]) / d["Close"] * 100
    d["BB_POS"]   = (d["Close"]  - d["BB_LOW"]) / \
                    (d["BB_HIGH"] - d["BB_LOW"] + 1e-9)

    atr = ta.volatility.AverageTrueRange(d["High"], d["Low"], d["Close"])
    d["ATR_PCT"] = atr.average_true_range() / d["Close"] * 100

    d["Vol_Ratio"] = d["Volume"] / (d["Volume"].rolling(20).mean() + 1)
    d["Ret_1d"]    = d["Close"].pct_change(1)  * 100
    d["Ret_3d"]    = d["Close"].pct_change(3)  * 100
    d["Ret_5d"]    = d["Close"].pct_change(5)  * 100
    d["EMA_Gap"]   = (d["EMA50"]  - d["EMA200"]) / d["EMA200"] * 100
    d["P_EMA50"]   = (d["Close"]  - d["EMA50"])  / d["EMA50"]  * 100
    d["EMA20_50"]  = (d["EMA20"]  - d["EMA50"])  / d["EMA50"]  * 100
    d["Body"]      = abs(d["Close"] - d["Open"]) / d["Open"]   * 100
    d["BULL"]      = (d["EMA50"]  > d["EMA200"]).astype(int)

    # Traditional Signal (simpler — no STRONG filter)
    d["TRAD"] = (
        (d["BULL"]    == 1) &
        (d["RSI14"]   > 50) &
        (d["MACD"]    > d["MACD_SIG"]) &
        (d["BB_POS"]  < 0.85)
    ).astype(int)
    return d


FEATURES = [
    "RSI14", "RSI7", "STOCH_K", "STOCH_D",
    "MACD_N", "MACD_HIST", "BB_WIDTH", "BB_POS",
    "ATR_PCT", "Vol_Ratio",
    "Ret_1d", "Ret_3d", "Ret_5d",
    "EMA_Gap", "P_EMA50", "EMA20_50", "Body"
]

feat_data = {}
for ticker, df in raw.items():
    feat_data[ticker] = add_features(df)
    name = STOCKS[ticker]
    n_sig = feat_data[ticker]["TRAD"].sum()
    print(f"    {name:20s} → {n_sig} traditional signals")

# ============================================================
# STEP 3 — SL/TP Target
# ============================================================

print(f"\n[3] Creating SL/TP target (SL:{STOP_LOSS*100:.0f}% TP:{TAKE_PROFIT*100:.0f}%)...")

def sl_tp_target(df, sl=STOP_LOSS, tp=TAKE_PROFIT, hold=HOLD_DAYS):
    closes = df["Close"].values
    highs  = df["High"].values
    lows   = df["Low"].values
    n      = len(closes)
    out    = []
    for i in range(n):
        entry  = closes[i]
        tp_p   = entry * (1 + tp)
        sl_p   = entry * (1 - sl)
        result = -1
        for j in range(1, min(hold + 1, n - i)):
            if highs[i+j] >= tp_p:
                result = 1; break
            if lows[i+j]  <= sl_p:
                result = 0; break
        if result == -1:
            fi = min(i + hold, n - 1)
            result = 1 if closes[fi] > entry else 0
        out.append(result)
    return pd.Series(out, index=df.index)

for ticker in feat_data:
    feat_data[ticker]["TARGET"] = sl_tp_target(feat_data[ticker])
    wr = feat_data[ticker]["TARGET"].mean() * 100
    print(f"    {STOCKS[ticker]:20s} → Overall win: {wr:.1f}%")

# ============================================================
# STEP 4 — KEY CHANGE: Train only on TRADITIONAL SIGNAL days
# ============================================================

print("\n[4] Combining — SIGNAL DAYS ONLY (smart filter approach)...")

signal_dfs = []
all_dfs_list = []

for ticker, df in feat_data.items():
    df = df.copy()
    df["Ticker"] = ticker
    all_dfs_list.append(df)

    # ML trains ONLY on days where traditional signal fired
    sig_df = df[df["TRAD"] == 1].copy()
    signal_dfs.append(sig_df)

all_signals = pd.concat(signal_dfs).dropna(subset=FEATURES + ["TARGET"])
all_signals = all_signals.sort_index()

print(f"    Signal days   : {len(all_signals)}")
print(f"    Signal Win%   : {all_signals['TARGET'].mean()*100:.1f}%")
print(f"\n    ML will learn: 'Which traditional signals will WIN?'")

# ============================================================
# STEP 5 — Train / Test (time-based)
# ============================================================

print("\n[5] Splitting...")

train_sig = all_signals[all_signals.index <= "2022-12-31"]
test_sig  = all_signals[all_signals.index >  "2022-12-31"]

X_tr = train_sig[FEATURES]
y_tr = train_sig["TARGET"]
X_te = test_sig[FEATURES]
y_te = test_sig["TARGET"]

pos = int(y_tr.sum())
neg = int(len(y_tr) - pos)

print(f"    Train signals : {len(X_tr)} | Win: {y_tr.mean()*100:.1f}%")
print(f"    Test  signals : {len(X_te)} | Win: {y_te.mean()*100:.1f}%")

# ============================================================
# STEP 6 — XGBoost (tuned for signal filtering)
# ============================================================

print("\n[6] Training XGBoost signal filter...")

model = xgb.XGBClassifier(
    n_estimators     = 400,
    max_depth        = 4,
    learning_rate    = 0.03,
    subsample        = 0.8,
    colsample_bytree = 0.7,
    min_child_weight = 5,
    gamma            = 1.0,
    reg_alpha        = 0.3,
    reg_lambda       = 1.5,
    scale_pos_weight = neg / max(pos, 1),
    random_state     = 42,
    eval_metric      = "logloss",
    verbosity        = 0
)
model.fit(X_tr, y_tr, verbose=False)

y_pred = model.predict(X_te)
y_prob = model.predict_proba(X_te)[:, 1]
acc    = accuracy_score(y_te, y_pred)
tr_acc = accuracy_score(y_tr, model.predict(X_tr))

print(f"\n    Train Accuracy : {tr_acc*100:.1f}%")
print(f"    Test  Accuracy : {acc*100:.1f}%")
print(f"    Overfit Gap    : {(tr_acc-acc)*100:.1f}%")
print(f"\n{classification_report(y_te, y_pred, target_names=['LOSS(0)','WIN(1)'])}")

# ============================================================
# STEP 7 — Walk-Forward on RELIANCE signals
# ============================================================

print("\n[7] Walk-Forward Validation (Reliance signals)...")

rel_sig = feat_data["RELIANCE.NS"]
rel_sig = rel_sig[rel_sig["TRAD"] == 1].dropna(subset=FEATURES + ["TARGET"])

if len(rel_sig) > 50:
    X_rel = rel_sig[FEATURES]
    y_rel = rel_sig["TARGET"]
    tscv  = TimeSeriesSplit(n_splits=5, test_size=max(10, len(rel_sig)//8))
    wf_accs = []

    for fold, (tr_i, te_i) in enumerate(tscv.split(X_rel), 1):
        X_tr_, X_te_ = X_rel.iloc[tr_i], X_rel.iloc[te_i]
        y_tr_, y_te_ = y_rel.iloc[tr_i], y_rel.iloc[te_i]
        if len(X_tr_) < 10 or len(X_te_) < 3:
            continue
        p_ = int(y_tr_.sum()); n_ = len(y_tr_) - p_
        m_ = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            gamma=1.0, reg_alpha=0.3, reg_lambda=1.5,
            scale_pos_weight=n_/max(p_,1),
            random_state=42, eval_metric="logloss", verbosity=0
        )
        m_.fit(X_tr_, y_tr_, verbose=False)
        fa = accuracy_score(y_te_, m_.predict(X_te_))
        wf_accs.append(fa)
        s = rel_sig.index[te_i[0]].strftime("%Y-%m")
        e = rel_sig.index[te_i[-1]].strftime("%Y-%m")
        sign = "✅" if fa >= 0.50 else "❌"
        bar  = "█" * int(fa * 30)
        print(f"    Fold {fold} | {s}→{e} | {fa*100:.1f}% {bar} {sign}")

    avg_wf = np.mean(wf_accs) if wf_accs else 0.5
    print(f"\n    Walk-Forward Average: {avg_wf*100:.1f}%")
else:
    wf_accs = [0.5]; avg_wf = 0.5
    print("    Not enough signal data for WF")

# ============================================================
# STEP 8 — Backtest per Stock
# ============================================================

print("\n[8] Backtesting all 5 stocks (2023-2024)...")

def simulate(df, signal_col, ml_prob_series=None,
             ml_thr=ML_THRESHOLD, sl=STOP_LOSS, tp=TAKE_PROFIT, hold=HOLD_DAYS):
    results = []
    if ml_prob_series is not None:
        signal_idx = df.index[
            (df[signal_col] == 1) &
            (ml_prob_series > ml_thr)
        ]
    else:
        signal_idx = df.index[df[signal_col] == 1]

    closes = df["Close"].values
    highs  = df["High"].values
    lows   = df["Low"].values

    for date in signal_idx:
        try:
            i = df.index.get_loc(date)
        except Exception:
            continue
        entry  = closes[i]
        tp_p   = entry * (1 + tp)
        sl_p   = entry * (1 - sl)
        ret    = 0
        etype  = "timeout"

        for j in range(1, min(hold+1, len(closes)-i)):
            if highs[i+j] >= tp_p:
                ret   = tp;  etype = "TP"; break
            if lows[i+j]  <= sl_p:
                ret   = -sl; etype = "SL"; break

        if etype == "timeout":
            fi  = min(i + hold, len(closes) - 1)
            ret = (closes[fi] - entry) / entry

        results.append({
            "date"      : date,
            "ticker"    : df["Ticker"].iloc[0] if "Ticker" in df.columns else "?",
            "entry"     : entry,
            "return_pct": ret * 100,
            "exit_type" : etype,
            "win"       : ret > 0
        })
    return pd.DataFrame(results)


print(f"\n    {'Stock':12s} {'Trad':>6} {'TradWR':>7} {'TradAvg':>8} "
      f"{'Hybrid':>7} {'HybWR':>7} {'HybAvg':>8}")
print(f"    {'-'*62}")

all_hybrid = []
all_trad   = []

for ticker, df in feat_data.items():
    bt  = df[df.index > "2022-12-31"].copy()
    if len(bt) < 20:
        continue
    bt["Ticker"] = ticker

    # Get ML probs for this stock's signal days
    bt_sig = bt[bt["TRAD"] == 1].dropna(subset=FEATURES)
    if len(bt_sig) > 0:
        probs = model.predict_proba(bt_sig[FEATURES])[:, 1]
        ml_s  = pd.Series(probs, index=bt_sig.index)
        ml_full = ml_s.reindex(bt.index)
    else:
        ml_full = pd.Series(0.0, index=bt.index)

    t_res = simulate(bt, "TRAD")
    h_res = simulate(bt, "TRAD", ml_prob_series=ml_full, ml_thr=ML_THRESHOLD)

    all_trad.append(t_res)
    all_hybrid.append(h_res)

    name  = STOCKS[ticker]
    t_wr  = t_res["win"].mean()*100 if len(t_res) else 0
    t_avg = t_res["return_pct"].mean() if len(t_res) else 0
    h_wr  = h_res["win"].mean()*100  if len(h_res) else 0
    h_avg = h_res["return_pct"].mean() if len(h_res) else 0

    print(f"    {name:12s} {len(t_res):>6} {t_wr:>6.1f}% {t_avg:>+7.2f}%"
          f" {len(h_res):>7} {h_wr:>6.1f}% {h_avg:>+7.2f}%")

all_trad_df   = pd.concat(all_trad,   ignore_index=True) if all_trad   else pd.DataFrame()
all_hybrid_df = pd.concat(all_hybrid, ignore_index=True) if all_hybrid else pd.DataFrame()

# ============================================================
# STEP 9 — Charts
# ============================================================

print("\n[9] Generating charts...")

DARK  = "#0d1117"
CARD  = "#161b22"
GRID  = "#21262d"
TEXT  = "#e6edf3"
MUTED = "#8b949e"

def style_ax(ax):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED, labelsize=9)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.grid(True, color=GRID, alpha=0.6, lw=0.5)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)

fig = plt.figure(figsize=(18, 18))
fig.patch.set_facecolor(DARK)
fig.suptitle("ALGO TRADING MODEL v5 — ML as Smart Signal Filter",
             fontsize=16, fontweight="bold", color=TEXT, y=0.99)
gs = gridspec.GridSpec(4, 2, hspace=0.55, wspace=0.35,
                       top=0.95, bottom=0.04)

# Chart 1: Walk-Forward
ax0 = fig.add_subplot(gs[0, :])
style_ax(ax0)
wf_x = [f"Fold {i+1}" for i in range(len(wf_accs))]
bc   = ["#3fb950" if a >= 0.50 else "#f85149" for a in wf_accs]
bars = ax0.bar(wf_x, [a*100 for a in wf_accs], color=bc, alpha=0.85,
               width=0.5, zorder=3)
ax0.axhline(50, color="#f85149", ls="--", lw=1.5, label="Random 50%")
ax0.axhline(avg_wf*100, color="#58a6ff", ls="-", lw=2,
            label=f"Avg: {avg_wf*100:.1f}%")
for bar, val in zip(bars, wf_accs):
    ax0.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f"{val*100:.1f}%", ha="center", va="bottom",
             color=TEXT, fontsize=11, fontweight="bold")
ax0.set_title("Walk-Forward Accuracy — Signal Filter", fontsize=12)
ax0.set_ylabel("Accuracy %", color=MUTED)
ax0.set_ylim(30, 80)
ax0.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)

# Chart 2: Win Rate per Stock
ax1 = fig.add_subplot(gs[1, 0])
style_ax(ax1)

stock_names = list(STOCKS.values())
t_wrs_list  = []
h_wrs_list  = []
for ticker in STOCKS:
    t_ = all_trad_df[all_trad_df["ticker"] == ticker]   if not all_trad_df.empty   else pd.DataFrame()
    h_ = all_hybrid_df[all_hybrid_df["ticker"] == ticker] if not all_hybrid_df.empty else pd.DataFrame()
    t_wrs_list.append(t_["win"].mean()*100 if len(t_) else 0)
    h_wrs_list.append(h_["win"].mean()*100 if len(h_) else 0)

x_    = np.arange(len(stock_names))
width = 0.35
ax1.bar(x_-width/2, t_wrs_list, width, label="Traditional",
        color="#d29922", alpha=0.85, zorder=3)
ax1.bar(x_+width/2, h_wrs_list, width, label="Hybrid ML v5",
        color="#3fb950", alpha=0.85, zorder=3)
ax1.axhline(50, color="#8b949e", ls="--", lw=1.2)
ax1.set_xticks(x_)
ax1.set_xticklabels(stock_names, rotation=20, ha="right")
ax1.set_title("Win Rate per Stock (2023-2024)", fontsize=12)
ax1.set_ylabel("Win Rate %", color=MUTED)
ax1.set_ylim(30, 80)
ax1.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)

# Chart 3: Avg Return per Stock
ax2 = fig.add_subplot(gs[1, 1])
style_ax(ax2)

t_avgs_list = []
h_avgs_list = []
for ticker in STOCKS:
    t_ = all_trad_df[all_trad_df["ticker"] == ticker]    if not all_trad_df.empty   else pd.DataFrame()
    h_ = all_hybrid_df[all_hybrid_df["ticker"] == ticker] if not all_hybrid_df.empty else pd.DataFrame()
    t_avgs_list.append(t_["return_pct"].mean() if len(t_) else 0)
    h_avgs_list.append(h_["return_pct"].mean() if len(h_) else 0)

ax2.bar(x_-width/2, t_avgs_list, width, label="Traditional",
        color="#d29922", alpha=0.85, zorder=3)
ax2.bar(x_+width/2, h_avgs_list, width, label="Hybrid ML v5",
        color="#3fb950", alpha=0.85, zorder=3)
ax2.axhline(0, color="#8b949e", ls="--", lw=1.2)
ax2.set_xticks(x_)
ax2.set_xticklabels(stock_names, rotation=20, ha="right")
ax2.set_title("Avg Return per Trade per Stock (2023-2024)", fontsize=12)
ax2.set_ylabel("Avg Return %", color=MUTED)
ax2.legend(fontsize=9, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)

# Chart 4: Equity Curves (RELIANCE)
rel_idx   = list(STOCKS.keys()).index("RELIANCE.NS")
rel_t_res = all_trad_df[all_trad_df["ticker"] == "RELIANCE.NS"]   if not all_trad_df.empty   else pd.DataFrame()
rel_h_res = all_hybrid_df[all_hybrid_df["ticker"] == "RELIANCE.NS"] if not all_hybrid_df.empty else pd.DataFrame()

ax3 = fig.add_subplot(gs[2, :])
style_ax(ax3)

def equity(trades):
    if trades.empty: return [1.0]
    c = [1.0]
    for r in trades["return_pct"]:
        c.append(c[-1] * (1 + r/100))
    return c

rel_t_eq = equity(rel_t_res)
rel_h_eq = equity(rel_h_res)

ax3.plot(rel_t_eq, color="#d29922", lw=1.8, ls="--",
         label=f"Traditional ({len(rel_t_res)} trades, WR:{rel_t_res['win'].mean()*100:.1f}%)" if len(rel_t_res) else "Traditional (0)")
ax3.plot(rel_h_eq, color="#3fb950", lw=2,
         label=f"Hybrid ML v5 ({len(rel_h_res)} trades, WR:{rel_h_res['win'].mean()*100:.1f}%)" if len(rel_h_res) else "Hybrid (0)")
ax3.axhline(1.0, color=MUTED, ls=":", lw=1)
ax3.fill_between(range(len(rel_h_eq)), rel_h_eq, 1.0,
                 where=np.array(rel_h_eq) >= 1.0,
                 alpha=0.12, color="#3fb950")
ax3.fill_between(range(len(rel_h_eq)), rel_h_eq, 1.0,
                 where=np.array(rel_h_eq) < 1.0,
                 alpha=0.12, color="#f85149")
ax3.set_title("RELIANCE — Equity Curve (2023-2024)", fontsize=12)
ax3.set_ylabel("Portfolio Growth", color=MUTED)
ax3.set_xlabel("Trade #", color=MUTED)
ax3.legend(fontsize=10, facecolor=CARD, edgecolor=GRID, labelcolor=TEXT)

# Chart 5: Feature Importance
ax4 = fig.add_subplot(gs[3, 0])
style_ax(ax4)

imp  = pd.Series(model.feature_importances_, index=FEATURES)
imp  = imp.sort_values(ascending=True).tail(12)
cols = plt.cm.Blues(np.linspace(0.4, 1.0, len(imp)))
ax4.barh(imp.index, imp.values, color=cols, alpha=0.9, zorder=3)
ax4.set_title("Top Feature Importance", fontsize=12)
ax4.set_xlabel("Score", color=MUTED)

# Chart 6: Exit Type breakdown
ax5 = fig.add_subplot(gs[3, 1])
style_ax(ax5)

if not all_hybrid_df.empty and len(all_hybrid_df) > 0:
    exit_counts = all_hybrid_df["exit_type"].value_counts()
    colors_pie  = {"TP": "#3fb950", "SL": "#f85149", "timeout": "#8b949e"}
    pie_colors  = [colors_pie.get(k, "#58a6ff") for k in exit_counts.index]
    ax5.pie(exit_counts.values,
            labels=[f"{k}\n({v})" for k,v in exit_counts.items()],
            colors=pie_colors, autopct="%1.1f%%", startangle=90,
            textprops={"color": TEXT, "fontsize": 10})
    ax5.set_title("Hybrid Trade Exits (All 5 Stocks)", fontsize=12)
else:
    ax5.text(0.5, 0.5, "No hybrid trades", ha="center",
             va="center", color=MUTED, fontsize=12)
    ax5.set_title("Hybrid Trade Exits", fontsize=12)

plt.savefig("chart6_model_v5.png", dpi=150,
            bbox_inches="tight", facecolor=DARK)
print("    Saved: chart6_model_v5.png")

# ============================================================
# STEP 10 — Final Report
# ============================================================

def summarize(df, label):
    if df.empty or len(df) == 0:
        return f"  {label:25s} : No trades"
    wr  = df["win"].mean() * 100
    avg = df["return_pct"].mean()
    tp_ = (df["exit_type"] == "TP").sum()
    sl_ = (df["exit_type"] == "SL").sum()
    to_ = (df["exit_type"] == "timeout").sum()
    final = np.prod(1 + df["return_pct"]/100)
    return (f"  {label:25s} : {len(df):3d} trades | "
            f"WR:{wr:5.1f}% | Avg:{avg:+5.2f}% | "
            f"TP:{tp_} SL:{sl_} TO:{to_} | "
            f"Final:{final:.2f}x")

print("\n" + "=" * 65)
print("            FINAL REPORT — MODEL v5")
print("=" * 65)
print(f"\n  Test  Accuracy   : {acc*100:.1f}%")
print(f"  Overfit Gap      : {(tr_acc-acc)*100:.1f}%")
print(f"  Walk-Fwd Avg     : {avg_wf*100:.1f}%")
print(f"  ML Threshold     : {ML_THRESHOLD*100:.0f}%")
print()
print(summarize(all_trad_df,   "Traditional (All 5)"))
print(summarize(all_hybrid_df, "Hybrid ML v5 (All 5)"))
print()

# Per stock
for ticker, name in STOCKS.items():
    t_ = all_trad_df[all_trad_df["ticker"]==ticker]   if not all_trad_df.empty   else pd.DataFrame()
    h_ = all_hybrid_df[all_hybrid_df["ticker"]==ticker] if not all_hybrid_df.empty else pd.DataFrame()
    t_wr = f"{t_['win'].mean()*100:.1f}%" if len(t_) else "N/A"
    h_wr = f"{h_['win'].mean()*100:.1f}%" if len(h_) else "N/A"
    print(f"  {name:12s} | Trad:{len(t_):3d} trades WR:{t_wr:>6} | "
          f"Hybrid:{len(h_):3d} trades WR:{h_wr:>6}")

# Demo Ready Check
t_wr_all = all_trad_df["win"].mean()*100   if not all_trad_df.empty   else 0
h_wr_all = all_hybrid_df["win"].mean()*100 if not all_hybrid_df.empty else 0
h_n      = len(all_hybrid_df)

checks = {
    f"Test Accuracy > 52%"   : acc >= 0.52,
    f"Hybrid WR > 50%"       : h_wr_all >= 50,
    f"Hybrid trades ≥ 10"    : h_n >= 10,
    f"Overfit Gap < 20%"     : (tr_acc - acc) < 0.20,
    f"Stop Loss implemented" : True,
    f"Multi-stock tested"    : True,
}

print(f"\n  Demo Trade Ready Checklist:")
print(f"  {'─'*50}")
all_pass = True
for label, result in checks.items():
    icon = "✅" if result else "⚠️"
    if not result: all_pass = False
    print(f"  {icon} {label}")
print(f"  {'─'*50}")
print(f"  {'🟢 MODEL IS DEMO-READY! 🎉' if all_pass else '🟡 Almost there — check ⚠️ items'}")
print()
plt.show()
print("Done!")
