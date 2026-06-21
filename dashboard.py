import streamlit as st
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AlgoView — NSE Hybrid Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
  }

  .main { background-color: #0d1117; }
  .block-container { padding: 1.5rem 2rem; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
  }
  [data-testid="stSidebar"] * { color: #e6edf3 !important; }

  /* Metric cards */
  .metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
    transition: border-color 0.2s;
  }
  .metric-card:hover { border-color: #58a6ff; }
  .metric-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #8b949e;
    text-transform: uppercase;
    margin-bottom: 6px;
  }
  .metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 26px;
    font-weight: 700;
    color: #e6edf3;
  }
  .metric-value.green { color: #3fb950; }
  .metric-value.red   { color: #f85149; }
  .metric-value.blue  { color: #58a6ff; }
  .metric-value.gold  { color: #d29922; }

  /* Signal badge */
  .signal-buy {
    display: inline-block;
    background: #1a4a2e;
    border: 1px solid #3fb950;
    color: #3fb950;
    font-family: 'JetBrains Mono', monospace;
    font-size: 15px;
    font-weight: 700;
    padding: 8px 22px;
    border-radius: 6px;
    letter-spacing: 0.05em;
  }
  .signal-skip {
    display: inline-block;
    background: #2d1a1e;
    border: 1px solid #f85149;
    color: #f85149;
    font-family: 'JetBrains Mono', monospace;
    font-size: 15px;
    font-weight: 700;
    padding: 8px 22px;
    border-radius: 6px;
    letter-spacing: 0.05em;
  }
  .signal-watch {
    display: inline-block;
    background: #2d2200;
    border: 1px solid #d29922;
    color: #d29922;
    font-family: 'JetBrains Mono', monospace;
    font-size: 15px;
    font-weight: 700;
    padding: 8px 22px;
    border-radius: 6px;
    letter-spacing: 0.05em;
  }

  /* Section header */
  .section-header {
    font-size: 13px;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    border-bottom: 1px solid #30363d;
    padding-bottom: 8px;
    margin-bottom: 16px;
  }

  /* Hide Streamlit elements */
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #30363d;
  }
  .stTabs [data-baseweb="tab"] {
    color: #8b949e !important;
    font-weight: 500;
    border-radius: 6px;
    padding: 6px 16px;
  }
  .stTabs [aria-selected="true"] {
    background: #1f6feb !important;
    color: #ffffff !important;
  }

  /* Logo */
  .logo-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 22px;
    font-weight: 700;
    color: #58a6ff;
    letter-spacing: -0.02em;
  }
  .logo-sub {
    font-size: 11px;
    color: #8b949e;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

STOCKS = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS"     : "Tata Consultancy Services",
    "INFY.NS"    : "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "WIPRO.NS"   : "Wipro",
}

COLORS = {
    "price"   : "#58a6ff",
    "ema50"   : "#d29922",
    "ema200"  : "#f85149",
    "buy"     : "#3fb950",
    "volume"  : "#388bfd44",
    "macd"    : "#bc8cff",
    "signal"  : "#f0883e",
    "hist_pos": "#3fb950",
    "hist_neg": "#f85149",
}

# ─────────────────────────────────────────────
# DATA & MODEL FUNCTIONS
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end,
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


def add_indicators(df):
    d = df.copy()
    d["EMA20"]  = ta.trend.EMAIndicator(d["Close"], 20).ema_indicator()
    d["EMA50"]  = ta.trend.EMAIndicator(d["Close"], 50).ema_indicator()
    d["EMA200"] = ta.trend.EMAIndicator(d["Close"], 200).ema_indicator()
    d["RSI"]    = ta.momentum.RSIIndicator(d["Close"], 14).rsi()
    d["RSI7"]   = ta.momentum.RSIIndicator(d["Close"], 7).rsi()

    st_ = ta.momentum.StochasticOscillator(d["High"], d["Low"], d["Close"])
    d["STOCH_K"] = st_.stoch()
    d["STOCH_D"] = st_.stoch_signal()

    m = ta.trend.MACD(d["Close"])
    d["MACD"]      = m.macd()
    d["MACD_SIG"]  = m.macd_signal()
    d["MACD_HIST"] = m.macd_diff()

    bb = ta.volatility.BollingerBands(d["Close"])
    d["BB_HIGH"]  = bb.bollinger_hband()
    d["BB_LOW"]   = bb.bollinger_lband()
    d["BB_MID"]   = bb.bollinger_mavg()
    d["BB_WIDTH"] = (d["BB_HIGH"] - d["BB_LOW"]) / d["Close"] * 100
    d["BB_POS"]   = (d["Close"]   - d["BB_LOW"]) / (d["BB_HIGH"] - d["BB_LOW"])

    d["ATR"]     = ta.volatility.AverageTrueRange(
        d["High"], d["Low"], d["Close"]).average_true_range()
    d["ATR_PCT"] = d["ATR"] / d["Close"] * 100

    d["Vol_Ratio"] = d["Volume"] / d["Volume"].rolling(20).mean()
    d["Ret_1d"]    = d["Close"].pct_change(1) * 100
    d["Ret_3d"]    = d["Close"].pct_change(3) * 100
    d["Ret_5d"]    = d["Close"].pct_change(5) * 100
    d["EMA_Gap"]   = (d["EMA50"]  - d["EMA200"]) / d["EMA200"] * 100
    d["P_EMA50"]   = (d["Close"]  - d["EMA50"])  / d["EMA50"]  * 100
    d["EMA20_50"]  = (d["EMA20"]  - d["EMA50"])  / d["EMA50"]  * 100
    d["BULL"]      = (d["EMA50"]  > d["EMA200"]).astype(int)
    d["STRONG"]    = ((d["EMA50"] > d["EMA200"]) &
                      (d["EMA_Gap"] > 0.5)).astype(int)
    return d


FEATURES = [
    "RSI", "RSI7", "STOCH_K", "STOCH_D",
    "MACD", "MACD_SIG", "MACD_HIST",
    "BB_WIDTH", "BB_POS", "ATR_PCT", "Vol_Ratio",
    "Ret_1d", "Ret_3d", "Ret_5d",
    "EMA_Gap", "P_EMA50", "EMA20_50",
    "BULL", "STRONG"
]


@st.cache_data(ttl=3600)
def train_model(ticker, start, end):
    df = load_data(ticker, start, end)
    if df.empty or len(df) < 250:
        return None, 0.0, pd.DataFrame()
    df = add_indicators(df)
    df["Future_Ret"] = (df["Close"].shift(-5) - df["Close"]) / df["Close"] * 100
    df["TARGET"]     = (df["Future_Ret"] > 0).astype(int)
    df.dropna(inplace=True)

    split  = int(len(df) * 0.80)
    X_tr   = df[FEATURES].iloc[:split]
    y_tr   = df["TARGET"].iloc[:split]
    X_te   = df[FEATURES].iloc[split:]
    y_te   = df["TARGET"].iloc[split:]

    pos = int(y_tr.sum()); neg = len(y_tr) - pos

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, gamma=0.5,
        reg_alpha=0.1, reg_lambda=1.5,
        scale_pos_weight=neg / max(pos, 1),
        random_state=42, eval_metric="logloss", verbosity=0
    )
    model.fit(X_tr, y_tr, verbose=False)
    acc = accuracy_score(y_te, model.predict(X_te))
    return model, acc, df


def get_signals(df, model, threshold=0.55):
    d = df.copy()
    feat_df      = d[FEATURES].dropna()
    probs        = model.predict_proba(feat_df)[:, 1]
    prob_series  = pd.Series(probs, index=feat_df.index)
    d["ML_PROB"] = prob_series

    d["HYBRID"] = (
        (d["ML_PROB"]  > threshold) &
        (d["MACD"]     > d["MACD_SIG"]) &
        (d["RSI"]      > 45)
    ).astype(int)

    d["TRAD"] = (
        (d["BULL"]  == 1) &
        (d["RSI"]   > 55) &
        (d["MACD"]  > d["MACD_SIG"])
    ).astype(int)

    return d


def run_backtest(df):
    d = df.copy()
    d["Daily_Ret"]    = d["Close"].pct_change()
    d["Hybrid_Ret"]   = d["Daily_Ret"].where(d["HYBRID"].shift(1) == 1, 0)
    d["Trad_Ret"]     = d["Daily_Ret"].where(d["TRAD"].shift(1) == 1, 0)
    d["Cum_Market"]   = (1 + d["Daily_Ret"]).cumprod()
    d["Cum_Hybrid"]   = (1 + d["Hybrid_Ret"]).cumprod()
    d["Cum_Trad"]     = (1 + d["Trad_Ret"]).cumprod()

    d["Future_Ret5"]  = (d["Close"].shift(-5) - d["Close"]) / d["Close"] * 100

    hybrid_trades     = d[d["HYBRID"] == 1]["Future_Ret5"].dropna()
    trad_trades       = d[d["TRAD"]   == 1]["Future_Ret5"].dropna()

    def wr(s): return (s > 0).mean() * 100 if len(s) else 0
    def ar(s): return s.mean() if len(s) else 0

    return d, {
        "hybrid_n"  : len(hybrid_trades),
        "hybrid_wr" : wr(hybrid_trades),
        "hybrid_avg": ar(hybrid_trades),
        "hybrid_cum": (d["Cum_Hybrid"].iloc[-1] - 1) * 100,
        "trad_n"    : len(trad_trades),
        "trad_wr"   : wr(trad_trades),
        "trad_avg"  : ar(trad_trades),
        "trad_cum"  : (d["Cum_Trad"].iloc[-1] - 1) * 100,
        "mkt_cum"   : (d["Cum_Market"].iloc[-1] - 1) * 100,
    }

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="logo-text">📈 AlgoView</div>', unsafe_allow_html=True)
    st.markdown('<div class="logo-sub">NSE Hybrid Trader</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Stock**")
    ticker = st.selectbox(
        "Select Stock",
        list(STOCKS.keys()),
        format_func=lambda x: f"{x.replace('.NS','')} — {STOCKS[x]}",
        label_visibility="collapsed"
    )

    st.markdown("**Date Range**")
    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("From", datetime(2020, 1, 1),
                                   label_visibility="collapsed")
    with col_e:
        end_date = st.date_input("To", datetime.today(),
                                 label_visibility="collapsed")

    st.markdown("**ML Threshold**")
    threshold = st.slider(
        "ML Probability Threshold",
        min_value=0.50, max_value=0.80,
        value=0.55, step=0.01,
        format="%.2f",
        label_visibility="collapsed",
        help="Higher = fewer but more confident signals"
    )

    st.markdown("**Chart Type**")
    chart_type = st.radio(
        "Chart", ["Candlestick", "Line"],
        horizontal=True, label_visibility="collapsed"
    )

    st.markdown("---")
    run_btn = st.button("▶ Run Analysis", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown('<div class="logo-sub">Strategy Rules</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:12px; color:#8b949e; line-height:1.8;">
    🤖 <b style="color:#58a6ff">Hybrid ML</b><br>
    &nbsp;&nbsp;ML Prob > threshold<br>
    &nbsp;&nbsp;+ MACD crossover<br>
    &nbsp;&nbsp;+ RSI > 45<br><br>
    📐 <b style="color:#d29922">Traditional</b><br>
    &nbsp;&nbsp;EMA50 > EMA200<br>
    &nbsp;&nbsp;+ RSI > 55<br>
    &nbsp;&nbsp;+ MACD crossover
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

company = STOCKS[ticker]
sym     = ticker.replace(".NS", "")

st.markdown(f"""
<div style="display:flex; align-items:baseline; gap:12px; margin-bottom:4px;">
  <span style="font-size:28px; font-weight:700; color:#e6edf3;">{company}</span>
  <span style="font-family:'JetBrains Mono',monospace; font-size:14px;
               color:#8b949e; background:#161b22; border:1px solid #30363d;
               border-radius:4px; padding:2px 8px;">{sym} · NSE</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD & TRAIN
# ─────────────────────────────────────────────

with st.spinner(f"Loading {sym} data & training ML model..."):
    raw_df = load_data(ticker, str(start_date), str(end_date))

if raw_df.empty:
    st.error(f"⚠️ Could not load data for {ticker}. Check internet connection.")
    st.stop()

model, acc, full_df = train_model(ticker, str(start_date), str(end_date))

if model is None:
    st.error("Not enough data to train model (need 250+ rows).")
    st.stop()

df_ind   = add_indicators(raw_df)
df_sig   = get_signals(df_ind, model, threshold)
df_bt, m = run_backtest(df_sig)

# Current price info
latest    = df_ind.iloc[-1]
prev      = df_ind.iloc[-2]
price_chg = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100

# Today's ML signal
today_feat = df_ind[FEATURES].iloc[[-1]].dropna()
if not today_feat.empty:
    today_prob = model.predict_proba(today_feat)[0][1]
    macd_ok    = latest["MACD"] > latest["MACD_SIG"]
    rsi_ok     = latest["RSI"]  > 45
    if today_prob > threshold and macd_ok and rsi_ok:
        signal_html = '<span class="signal-buy">▲ BUY SIGNAL</span>'
        signal_text = "BUY"
    elif today_prob > 0.48:
        signal_html = '<span class="signal-watch">◆ WATCH</span>'
        signal_text = "WATCH"
    else:
        signal_html = '<span class="signal-skip">▼ SKIP</span>'
        signal_text = "SKIP"
else:
    today_prob  = 0.5
    signal_html = '<span class="signal-watch">◆ WATCH</span>'
    signal_text = "WATCH"

# ─────────────────────────────────────────────
# TOP METRICS ROW
# ─────────────────────────────────────────────

c1, c2, c3, c4, c5, c6 = st.columns(6)

def metric_card(label, value, cls=""):
    return f"""<div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value {cls}">{value}</div>
    </div>"""

price_cls = "green" if price_chg >= 0 else "red"
chg_cls   = "green" if price_chg >= 0 else "red"
hyb_cls   = "green" if m["hybrid_cum"] >= 0 else "red"
mkt_cls   = "green" if m["mkt_cum"]    >= 0 else "red"

with c1:
    st.markdown(metric_card("Current Price",
        f"₹{latest['Close']:.1f}"), unsafe_allow_html=True)
with c2:
    st.markdown(metric_card("Day Change",
        f"{price_chg:+.2f}%", chg_cls), unsafe_allow_html=True)
with c3:
    st.markdown(metric_card("RSI (14)",
        f"{latest['RSI']:.1f}",
        "red" if latest['RSI'] > 70 else "green" if latest['RSI'] < 30 else "blue"),
        unsafe_allow_html=True)
with c4:
    st.markdown(metric_card("ML Confidence",
        f"{today_prob*100:.1f}%", "blue"), unsafe_allow_html=True)
with c5:
    st.markdown(metric_card("Hybrid Return",
        f"{m['hybrid_cum']:+.1f}%", hyb_cls), unsafe_allow_html=True)
with c6:
    st.markdown(metric_card("Model Accuracy",
        f"{acc*100:.1f}%", "gold"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Today's signal banner
sig_col, info_col = st.columns([1, 3])
with sig_col:
    st.markdown(f"""
    <div style="background:#161b22; border:1px solid #30363d; border-radius:10px;
                padding:16px; text-align:center;">
      <div class="metric-label" style="margin-bottom:10px;">Today's Signal</div>
      {signal_html}
      <div style="font-size:11px; color:#8b949e; margin-top:10px;">
        ML Prob: {today_prob*100:.1f}% · Threshold: {threshold*100:.0f}%
      </div>
    </div>
    """, unsafe_allow_html=True)

with info_col:
    bull = latest["EMA50"] > latest["EMA200"]
    macd_cross = latest["MACD"] > latest["MACD_SIG"]
    rsi_val    = latest["RSI"]

    def check(v): return "✅" if v else "❌"

    st.markdown(f"""
    <div style="background:#161b22; border:1px solid #30363d; border-radius:10px;
                padding:16px;">
      <div class="metric-label" style="margin-bottom:10px;">Signal Conditions</div>
      <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; font-size:13px;">
        <div>{check(today_prob > threshold)} ML Prob > {threshold*100:.0f}%
             <span style="color:#8b949e; font-size:11px;">({today_prob*100:.1f}%)</span></div>
        <div>{check(macd_cross)} MACD Crossover
             <span style="color:#8b949e; font-size:11px;">
             ({latest['MACD']:.2f} vs {latest['MACD_SIG']:.2f})</span></div>
        <div>{check(rsi_val > 45)} RSI > 45
             <span style="color:#8b949e; font-size:11px;">({rsi_val:.1f})</span></div>
        <div>{check(bull)} Uptrend (EMA50>200)
             <span style="color:#8b949e; font-size:11px;">
             (gap {latest['EMA_Gap']:.1f}%)</span></div>
        <div>{check(latest['BB_POS'] < 0.8)} Not Overbought (BB)
             <span style="color:#8b949e; font-size:11px;">
             (pos {latest['BB_POS']:.2f})</span></div>
        <div>{check(latest['Vol_Ratio'] > 0.8)} Volume OK
             <span style="color:#8b949e; font-size:11px;">
             ({latest['Vol_Ratio']:.2f}x avg)</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Price Chart",
    "📊 Indicators",
    "🤖 ML Analysis",
    "💰 Backtest"
])

# ── TAB 1: PRICE CHART ──────────────────────

with tab1:
    buy_pts = df_bt[df_bt["HYBRID"] == 1]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03
    )

    # Price
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=df_bt.index,
            open=df_bt["Open"], high=df_bt["High"],
            low=df_bt["Low"],   close=df_bt["Close"],
            name="Price",
            increasing_line_color="#3fb950",
            decreasing_line_color="#f85149",
            increasing_fillcolor="#1a4a2e",
            decreasing_fillcolor="#3d1a1f",
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=df_bt.index, y=df_bt["Close"],
            name="Price", line=dict(color=COLORS["price"], width=1.5)
        ), row=1, col=1)

    # EMAs
    fig.add_trace(go.Scatter(
        x=df_bt.index, y=df_bt["EMA50"],
        name="EMA 50", line=dict(color=COLORS["ema50"], width=1.2, dash="dash")
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_bt.index, y=df_bt["EMA200"],
        name="EMA 200", line=dict(color=COLORS["ema200"], width=1.2, dash="dash")
    ), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df_bt.index, y=df_bt["BB_HIGH"],
        name="BB Upper",
        line=dict(color="#8b949e", width=0.5, dash="dot"),
        showlegend=False
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df_bt.index, y=df_bt["BB_LOW"],
        name="BB Lower",
        line=dict(color="#8b949e", width=0.5, dash="dot"),
        fill="tonexty",
        fillcolor="rgba(88,166,255,0.04)",
        showlegend=False
    ), row=1, col=1)

    # BUY Signals
    fig.add_trace(go.Scatter(
        x=buy_pts.index, y=buy_pts["Close"],
        mode="markers",
        name=f"Hybrid BUY ({len(buy_pts)})",
        marker=dict(
            symbol="triangle-up",
            size=10,
            color=COLORS["buy"],
            line=dict(color="#0d1117", width=1)
        )
    ), row=1, col=1)

    # Traditional signals
    trad_pts = df_bt[df_bt["TRAD"] == 1]
    fig.add_trace(go.Scatter(
        x=trad_pts.index, y=trad_pts["Close"],
        mode="markers",
        name=f"Traditional ({len(trad_pts)})",
        marker=dict(
            symbol="triangle-up",
            size=7,
            color=COLORS["ema50"],
            opacity=0.6
        )
    ), row=1, col=1)

    # Volume
    vol_colors = ["#3fb950" if c >= o else "#f85149"
                  for c, o in zip(df_bt["Close"], df_bt["Open"])]
    fig.add_trace(go.Bar(
        x=df_bt.index, y=df_bt["Volume"],
        name="Volume",
        marker_color=vol_colors,
        opacity=0.6
    ), row=2, col=1)

    fig.update_layout(
        height=560,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(family="Inter", color="#8b949e", size=12),
        legend=dict(
            bgcolor="#161b22", bordercolor="#30363d",
            borderwidth=1, font=dict(size=11)
        ),
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    fig.update_xaxes(gridcolor="#21262d", zeroline=False)
    fig.update_yaxes(gridcolor="#21262d", zeroline=False)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

# ── TAB 2: INDICATORS ───────────────────────

with tab2:
    fig2 = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.4, 0.3, 0.3],
        vertical_spacing=0.04,
        subplot_titles=["RSI (14)", "MACD", "Stochastic Oscillator"]
    )

    # RSI
    fig2.add_trace(go.Scatter(
        x=df_bt.index, y=df_bt["RSI"],
        name="RSI", line=dict(color="#bc8cff", width=1.5)
    ), row=1, col=1)
    for level, color, name in [(70, "#f85149", "Overbought"),
                                 (30, "#3fb950", "Oversold"),
                                 (50, "#8b949e", "Midline")]:
        fig2.add_hline(y=level, line=dict(color=color, width=1, dash="dash"),
                       row=1, col=1)
    fig2.add_hrect(y0=70, y1=100, fillcolor="rgba(248,81,73,0.06)",
                   line_width=0, row=1, col=1)
    fig2.add_hrect(y0=0,  y1=30,  fillcolor="rgba(63,185,80,0.06)",
                   line_width=0, row=1, col=1)

    # MACD
    fig2.add_trace(go.Scatter(
        x=df_bt.index, y=df_bt["MACD"],
        name="MACD", line=dict(color=COLORS["macd"], width=1.5)
    ), row=2, col=1)
    fig2.add_trace(go.Scatter(
        x=df_bt.index, y=df_bt["MACD_SIG"],
        name="Signal", line=dict(color=COLORS["signal"], width=1.2)
    ), row=2, col=1)
    hist_colors = [COLORS["hist_pos"] if v >= 0 else COLORS["hist_neg"]
                   for v in df_bt["MACD_HIST"]]
    fig2.add_trace(go.Bar(
        x=df_bt.index, y=df_bt["MACD_HIST"],
        name="Histogram", marker_color=hist_colors, opacity=0.7
    ), row=2, col=1)

    # Stochastic
    fig2.add_trace(go.Scatter(
        x=df_bt.index, y=df_bt["STOCH_K"],
        name="%K", line=dict(color="#58a6ff", width=1.3)
    ), row=3, col=1)
    fig2.add_trace(go.Scatter(
        x=df_bt.index, y=df_bt["STOCH_D"],
        name="%D", line=dict(color="#f0883e", width=1.2, dash="dash")
    ), row=3, col=1)
    fig2.add_hline(y=80, line=dict(color="#f85149", width=1, dash="dash"), row=3, col=1)
    fig2.add_hline(y=20, line=dict(color="#3fb950", width=1, dash="dash"), row=3, col=1)

    fig2.update_layout(
        height=560,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(family="Inter", color="#8b949e"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    fig2.update_xaxes(gridcolor="#21262d", zeroline=False)
    fig2.update_yaxes(gridcolor="#21262d", zeroline=False)
    st.plotly_chart(fig2, use_container_width=True)

# ── TAB 3: ML ANALYSIS ──────────────────────

with tab3:
    ml_col1, ml_col2 = st.columns([2, 1])

    with ml_col1:
        fig3 = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.5, 0.5],
            vertical_spacing=0.06,
            subplot_titles=["ML Probability of Price Going UP",
                            "Hybrid vs Traditional Signals"]
        )

        # ML Probability
        fig3.add_trace(go.Scatter(
            x=df_bt.index, y=df_bt["ML_PROB"],
            name="ML Probability",
            line=dict(color="#bc8cff", width=1.3),
            fill="tozeroy",
            fillcolor="rgba(188,140,255,0.07)"
        ), row=1, col=1)

        fig3.add_hline(
            y=threshold,
            line=dict(color="#f85149", width=1.5, dash="dash"),
            annotation_text=f"Threshold {threshold*100:.0f}%",
            annotation_font=dict(color="#f85149", size=11),
            row=1, col=1
        )

        # Signals comparison
        fig3.add_trace(go.Scatter(
            x=df_bt.index, y=df_bt["Close"],
            name="Price", line=dict(color="#58a6ff", width=1), opacity=0.6
        ), row=2, col=1)

        fig3.add_trace(go.Scatter(
            x=trad_pts.index, y=trad_pts["Close"],
            mode="markers", name=f"Traditional ({len(trad_pts)})",
            marker=dict(symbol="triangle-up", size=7,
                        color="#d29922", opacity=0.7)
        ), row=2, col=1)

        fig3.add_trace(go.Scatter(
            x=buy_pts.index, y=buy_pts["Close"],
            mode="markers", name=f"Hybrid ML ({len(buy_pts)})",
            marker=dict(symbol="triangle-up", size=10, color="#3fb950")
        ), row=2, col=1)

        fig3.update_layout(
            height=500,
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(family="Inter", color="#8b949e"),
            legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        fig3.update_xaxes(gridcolor="#21262d", zeroline=False)
        fig3.update_yaxes(gridcolor="#21262d", zeroline=False)
        st.plotly_chart(fig3, use_container_width=True)

    with ml_col2:
        # Feature importance
        imp = pd.Series(model.feature_importances_, index=FEATURES)
        imp = imp.sort_values(ascending=False).head(10).sort_values()

        fig_imp = go.Figure(go.Bar(
            x=imp.values, y=imp.index,
            orientation="h",
            marker=dict(
                color=imp.values,
                colorscale=[[0, "#1f6feb"], [1, "#58a6ff"]],
                showscale=False
            )
        ))
        fig_imp.update_layout(
            title=dict(text="Feature Importance", font=dict(size=13, color="#e6edf3")),
            height=500,
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(family="Inter", color="#8b949e", size=11),
            margin=dict(l=0, r=10, t=40, b=0),
            xaxis=dict(gridcolor="#21262d", zeroline=False),
            yaxis=dict(gridcolor="#21262d", zeroline=False),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# ── TAB 4: BACKTEST ─────────────────────────

with tab4:
    # Cumulative returns chart
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=df_bt.index, y=df_bt["Cum_Market"],
        name="Buy & Hold",
        line=dict(color="#58a6ff", width=2)
    ))
    fig4.add_trace(go.Scatter(
        x=df_bt.index, y=df_bt["Cum_Trad"],
        name=f"Traditional ({m['trad_n']} trades)",
        line=dict(color="#d29922", width=1.5, dash="dash")
    ))
    fig4.add_trace(go.Scatter(
        x=df_bt.index, y=df_bt["Cum_Hybrid"],
        name=f"Hybrid ML ({m['hybrid_n']} trades)",
        line=dict(color="#3fb950", width=2)
    ))
    fig4.add_hline(y=1.0, line=dict(color="#8b949e", width=1, dash="dot"))

    fig4.update_layout(
        title=dict(text="Cumulative Returns Comparison",
                   font=dict(size=14, color="#e6edf3")),
        height=360,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(family="Inter", color="#8b949e"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
        yaxis_title="Growth (1x = start)",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig4.update_xaxes(gridcolor="#21262d", zeroline=False)
    fig4.update_yaxes(gridcolor="#21262d", zeroline=False)
    st.plotly_chart(fig4, use_container_width=True)

    # Stats table
    st.markdown('<div class="section-header">Strategy Comparison</div>',
                unsafe_allow_html=True)

    def color_val(v, fmt="{:.1f}%", positive_good=True):
        val = fmt.format(v)
        if positive_good:
            c = "#3fb950" if v > 0 else "#f85149"
        else:
            c = "#f85149" if v > 0 else "#3fb950"
        return f'<span style="color:{c}; font-family:JetBrains Mono,monospace; font-weight:600;">{val}</span>'

    table_html = f"""
    <table style="width:100%; border-collapse:collapse; font-size:13px;">
      <thead>
        <tr style="border-bottom:1px solid #30363d; color:#8b949e; font-size:11px; text-transform:uppercase; letter-spacing:0.06em;">
          <th style="padding:10px 0; text-align:left;">Metric</th>
          <th style="padding:10px; text-align:right;">Traditional</th>
          <th style="padding:10px; text-align:right;">Hybrid ML</th>
          <th style="padding:10px; text-align:right;">Market</th>
        </tr>
      </thead>
      <tbody>
        <tr style="border-bottom:1px solid #21262d;">
          <td style="padding:10px 0; color:#8b949e;">Total Signals</td>
          <td style="padding:10px; text-align:right; font-family:JetBrains Mono,monospace;">{m['trad_n']}</td>
          <td style="padding:10px; text-align:right; font-family:JetBrains Mono,monospace;">{m['hybrid_n']}</td>
          <td style="padding:10px; text-align:right; color:#8b949e;">—</td>
        </tr>
        <tr style="border-bottom:1px solid #21262d;">
          <td style="padding:10px 0; color:#8b949e;">Win Rate (5d)</td>
          <td style="padding:10px; text-align:right;">{color_val(m['trad_wr'])}</td>
          <td style="padding:10px; text-align:right;">{color_val(m['hybrid_wr'])}</td>
          <td style="padding:10px; text-align:right; color:#8b949e;">—</td>
        </tr>
        <tr style="border-bottom:1px solid #21262d;">
          <td style="padding:10px 0; color:#8b949e;">Avg Trade Return</td>
          <td style="padding:10px; text-align:right;">{color_val(m['trad_avg'], '{:.2f}%')}</td>
          <td style="padding:10px; text-align:right;">{color_val(m['hybrid_avg'], '{:.2f}%')}</td>
          <td style="padding:10px; text-align:right; color:#8b949e;">—</td>
        </tr>
        <tr>
          <td style="padding:10px 0; color:#8b949e;">Total Return</td>
          <td style="padding:10px; text-align:right;">{color_val(m['trad_cum'])}</td>
          <td style="padding:10px; text-align:right;">{color_val(m['hybrid_cum'])}</td>
          <td style="padding:10px; text-align:right;">{color_val(m['mkt_cum'])}</td>
        </tr>
      </tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">⚠️ Risk Disclaimer</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:12px; color:#8b949e; line-height:1.8;
                background:#161b22; border:1px solid #30363d;
                border-radius:8px; padding:14px;">
    This dashboard is for <b style="color:#e6edf3">educational & research purposes only</b>.
    Past performance does not guarantee future results.
    Always do your own research before investing.
    </div>
    """, unsafe_allow_html=True)
