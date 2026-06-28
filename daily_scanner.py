"""
╔══════════════════════════════════════════════════════════════╗
║           ALGO TRADING — Daily Signal Scanner v3             ║
║           5 Core Conditions + ADX & Volume Extra Info        ║
╚══════════════════════════════════════════════════════════════╝
"""

import yfinance as yf
import pandas as pd
import ta
from datetime import datetime, timedelta
import warnings
import csv
import os
import time
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────

STOCKS = {
    "BAJFINANCE.NS" : "Bajaj Finance",
    "TITAN.NS"      : "Titan",
    "SUNPHARMA.NS"  : "Sun Pharma",
    "MARUTI.NS"     : "Maruti",
    "ADANIENT.NS"   : "Adani Ent",
}

STOP_LOSS_PCT   = 2.0
TAKE_PROFIT_PCT = 4.0
JOURNAL_FILE    = "trade_journal.csv"
MIN_ROWS        = 210

G    = "\033[92m"
R    = "\033[91m"
Y    = "\033[93m"
B    = "\033[94m"
C    = "\033[96m"
W    = "\033[97m"
D    = "\033[90m"
BOLD = "\033[1m"
RST  = "\033[0m"

# ──────────────────────────────────────────────────────────────
# DATA DOWNLOAD
# ──────────────────────────────────────────────────────────────

def download_data(ticker):
    end   = datetime.today()
    start = end - timedelta(days=500)

    for attempt in range(3):
        try:
            df = yf.download(
                ticker,
                start       = start.strftime("%Y-%m-%d"),
                end         = end.strftime("%Y-%m-%d"),
                auto_adjust = True,
                progress    = False,
                threads     = False
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open","High","Low","Close","Volume"]].copy()
            df.dropna(inplace=True)
            if len(df) >= MIN_ROWS:
                return df
        except Exception:
            time.sleep(2)

    try:
        df = yf.Ticker(ticker).history(period="2y")
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[["Open","High","Low","Close","Volume"]].copy()
        df.dropna(inplace=True)
        if len(df) >= MIN_ROWS:
            return df
    except Exception:
        pass

    return pd.DataFrame()

# ──────────────────────────────────────────────────────────────
# INDICATORS
# ──────────────────────────────────────────────────────────────

def add_indicators(df):
    d     = df.copy()
    close = d["Close"].squeeze()
    high  = d["High"].squeeze()
    low   = d["Low"].squeeze()
    vol   = d["Volume"].squeeze()

    # ── 5 MAIN CONDITIONS ──────────────────────
    # 1. EMA
    d["EMA20"]  = ta.trend.EMAIndicator(close, 20).ema_indicator()
    d["EMA50"]  = ta.trend.EMAIndicator(close, 50).ema_indicator()
    d["EMA200"] = ta.trend.EMAIndicator(close, 200).ema_indicator()

    # 2. RSI
    d["RSI"] = ta.momentum.RSIIndicator(close, 14).rsi()

    # 3. MACD
    macd          = ta.trend.MACD(close)
    d["MACD"]     = macd.macd()
    d["MACD_SIG"] = macd.macd_signal()
    d["MACD_HIST"]= macd.macd_diff()

    # 4. Bollinger Bands
    bb          = ta.volatility.BollingerBands(close)
    d["BB_POS"] = (close - bb.bollinger_lband()) / \
                  (bb.bollinger_hband() - bb.bollinger_lband() + 1e-9)

    # 5. EMA20 > EMA50 (Short uptrend)
    # Already calculated above

    # ── EXTRA INFO (Bonus) ─────────────────────
    # ADX — Trend Strength
    adx        = ta.trend.ADXIndicator(high, low, close, 14)
    d["ADX"]   = adx.adx()

    # Volume vs 20-day average
    d["VOL_MA20"]  = vol.rolling(20).mean()
    d["VOL_RATIO"] = vol / (d["VOL_MA20"] + 1)

    # EMA Gap
    d["EMA_GAP"] = (d["EMA50"] - d["EMA200"]) / d["EMA200"] * 100

    return d.dropna()

# ──────────────────────────────────────────────────────────────
# SIGNAL CHECK — 5 MAIN CONDITIONS
# ──────────────────────────────────────────────────────────────

def check_signal(df):
    if len(df) < 5:
        return False, {}, None

    last = df.iloc[-1]

    # ── 5 CORE RULES (All must pass!) ──────────
    core_rules = {
        "EMA50 > EMA200   [Uptrend]"         : bool(last["EMA50"]  > last["EMA200"]),
        "RSI 50-70        [Momentum]"         : bool(50 < last["RSI"] < 70),
        "MACD > Signal    [Bullish]"          : bool(last["MACD"]   > last["MACD_SIG"]),
        "BB Position < 0.85 [Not Overbought]" : bool(last["BB_POS"] < 0.85),
        "EMA20 > EMA50    [Short Uptrend]"    : bool(last["EMA20"]  > last["EMA50"]),
    }

    all_pass = all(core_rules.values())
    return all_pass, core_rules, last

# ──────────────────────────────────────────────────────────────
# EXTRA INFO DISPLAY
# ──────────────────────────────────────────────────────────────

def extra_info_str(last):
    adx       = float(last["ADX"])
    vol_ratio = float(last["VOL_RATIO"])
    ema_gap   = float(last["EMA_GAP"])
    macd_hist = float(last["MACD_HIST"])

    adx_str = f"{adx:.1f} {'✅ Strong' if adx > 25 else '⚠️  Weak'}"
    vol_str = f"{vol_ratio:.2f}x {'✅ High'   if vol_ratio > 1 else '⚠️  Low'}"
    gap_str = f"{ema_gap:+.1f}%"
    mcd_str = f"{macd_hist:+.2f} {'✅ Rising' if macd_hist > 0 else '⚠️  Falling'}"

    return adx_str, vol_str, gap_str, mcd_str

# ──────────────────────────────────────────────────────────────
# JOURNAL
# ──────────────────────────────────────────────────────────────

def init_journal():
    if not os.path.exists(JOURNAL_FILE):
        with open(JOURNAL_FILE, "w", newline="") as f:
            csv.writer(f).writerow([
                "Date","Stock","Ticker",
                "Entry_Price","Stop_Loss","Take_Profit",
                "ADX","Vol_Ratio",
                "Exit_Date","Exit_Price","Exit_Type",
                "Return_%","Result","Notes"
            ])

def log_trade(ticker, name, price, sl, tp, adx, vol):
    with open(JOURNAL_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.today().strftime("%Y-%m-%d"),
            name, ticker,
            f"{price:.2f}", f"{sl:.2f}", f"{tp:.2f}",
            f"{adx:.1f}", f"{vol:.2f}",
            "","","","","OPEN",""
        ])

def show_open_trades():
    if not os.path.exists(JOURNAL_FILE):
        return
    df = pd.read_csv(JOURNAL_FILE)
    open_tr = df[df["Result"] == "OPEN"]
    if len(open_tr) == 0:
        return
    print(f"\n  {BOLD}{Y}  OPEN TRADES ({len(open_tr)}){RST}")
    print(f"  {'─'*65}")
    print(f"  {D}  {'Date':<12} {'Stock':<14} {'Entry':>8} "
          f"{'SL':>8} {'TP':>8} {'Days':>5}{RST}")
    print(f"  {'─'*65}")
    today = datetime.today().date()
    for _, row in open_tr.iterrows():
        try:
            days = (today - pd.to_datetime(row["Date"]).date()).days
            print(f"  {W}  {str(row['Date']):<12} "
                  f"{str(row['Stock']):<14} "
                  f"₹{float(row['Entry_Price']):>7.1f} "
                  f"{R}₹{float(row['Stop_Loss']):>7.1f}{RST} "
                  f"{G}₹{float(row['Take_Profit']):>7.1f}{RST} "
                  f"{'⚠️' if days > 8 else ' '}{days:>3}d")
        except Exception:
            pass
    print()

def show_performance():
    if not os.path.exists(JOURNAL_FILE):
        return
    df   = pd.read_csv(JOURNAL_FILE)
    done = df[df["Result"].isin(["WIN","LOSS"])]
    if len(done) == 0:
        print(f"  {D}  No completed trades yet.{RST}\n")
        return
    wins  = (done["Result"] == "WIN").sum()
    loss  = (done["Result"] == "LOSS").sum()
    wr    = wins / len(done) * 100
    rets  = pd.to_numeric(done["Return_%"], errors="coerce")
    avg   = rets.mean()
    total = rets.sum()
    wc    = G if wr   >= 50 else R
    ac    = G if avg  >  0  else R
    tc    = G if total > 0  else R
    print(f"\n  {BOLD}{C}  PERFORMANCE SUMMARY{RST}")
    print(f"  {'─'*45}")
    print(f"  {W}  Total   : {BOLD}{len(done)}{RST}  "
          f"({G}W:{wins}{RST} / {R}L:{loss}{RST})")
    print(f"  {W}  Win Rate: {wc}{BOLD}{wr:.1f}%{RST}")
    print(f"  {W}  Avg Ret : {ac}{BOLD}{avg:+.2f}%{RST}")
    print(f"  {W}  Total   : {tc}{BOLD}{total:+.2f}%{RST}\n")

# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    now   = datetime.now()
    today = now.strftime("%Y-%m-%d")

    print(f"\n{BOLD}{B}"
          f"╔══════════════════════════════════════════════════╗\n"
          f"║        ALGO TRADING — Daily Signal Scanner v3    ║\n"
          f"║        {today}  {now.strftime('%I:%M %p')}  |  NSE India        ║\n"
          f"╚══════════════════════════════════════════════════╝"
          f"{RST}\n")

    init_journal()
    show_open_trades()

    print(f"  {BOLD}{W}  Scanning {len(STOCKS)} stocks...{RST}\n")

    buy_signals = []
    watch_list  = []
    no_signals  = []
    errors      = []

    for ticker, name in STOCKS.items():
        print(f"  {D}  Scanning {name}...{RST}", end=" ", flush=True)

        df = download_data(ticker)
        if df.empty:
            errors.append(name)
            print(f"{R}FAILED{RST}")
            continue

        try:
            df = add_indicators(df)
            ok, rules, last = check_signal(df)

            price   = float(last["Close"])
            rsi     = float(last["RSI"])
            score   = sum(rules.values())
            adx     = float(last["ADX"])
            vol_r   = float(last["VOL_RATIO"])

            if ok:
                sl = price * (1 - STOP_LOSS_PCT / 100)
                tp = price * (1 + TAKE_PROFIT_PCT / 100)
                buy_signals.append(dict(
                    ticker=ticker, name=name,
                    price=price, sl=sl, tp=tp,
                    rsi=rsi, rules=rules,
                    adx=adx, vol_r=vol_r,
                    last=last
                ))
                print(f"{G}BUY SIGNAL! ({score}/5){RST}")
            elif score >= 3:
                waiting = [
                    k.split("[")[0].strip()
                    for k, v in rules.items() if not v
                ]
                watch_list.append(dict(
                    name=name, price=price,
                    rsi=rsi, score=score,
                    waiting=waiting,
                    adx=adx, vol_r=vol_r
                ))
                print(f"{Y}Watch ({score}/5){RST}")
            else:
                no_signals.append(name)
                print(f"{D}No signal ({score}/5){RST}")

        except Exception as e:
            errors.append(name)
            print(f"{R}Error{RST}")

    # ── BUY SIGNALS ──────────────────────────────

    print()
    if buy_signals:
        print(f"  {BOLD}{G}"
              f"╔══════════════════════════════════════════════╗\n"
              f"  ║   ▲  BUY SIGNAL(S) FOUND  —  "
              f"{len(buy_signals)} stock(s)          ║\n"
              f"  ╚══════════════════════════════════════════════╝"
              f"{RST}\n")

        for s in buy_signals:
            sym = s["ticker"].replace(".NS","")
            adx_s, vol_s, gap_s, mcd_s = extra_info_str(s["last"])

            print(f"  {BOLD}{G}  ▲  {s['name']} ({sym}){RST}")
            print(f"  {'─'*55}")

            # Price info
            print(f"  {W}  Current Price  : {BOLD}₹{s['price']:.2f}{RST}")
            print(f"  {G}  Take Profit    : {BOLD}₹{s['tp']:.2f}{RST}"
                  f"  {D}(+{TAKE_PROFIT_PCT}%){RST}")
            print(f"  {R}  Stop Loss      : {BOLD}₹{s['sl']:.2f}{RST}"
                  f"  {D}(-{STOP_LOSS_PCT}%){RST}")

            # 5 Main Conditions
            print(f"\n  {BOLD}{W}  ★ 5 CORE CONDITIONS (All Pass!):{RST}")
            for rule, result in s["rules"].items():
                icon = f"{G}  ✔{RST}" if result else f"{R}  ✘{RST}"
                print(f"     {icon}  {W}{rule}{RST}")

            # Extra Info
            print(f"\n  {BOLD}{C}  ➕ EXTRA INFO (Bonus):{RST}")
            print(f"     {D}  ADX (Trend Strength) : {W}{adx_s}{RST}")
            print(f"     {D}  Volume vs Avg20      : {W}{vol_s}{RST}")
            print(f"     {D}  EMA Gap (50-200)     : {W}{gap_s}{RST}")
            print(f"     {D}  MACD Histogram       : {W}{mcd_s}{RST}")

            # Action
            print(f"\n  {BOLD}{Y}  ACTION:{RST}")
            print(f"  {W}  1. TradingView → {sym} Chart verify")
            print(f"     2. Note: Entry ₹{s['price']:.0f} | "
                  f"SL ₹{s['sl']:.0f} | TP ₹{s['tp']:.0f}")
            print(f"     3. Set SL Alert : ₹{s['sl']:.0f}")
            print(f"     4. Set TP Alert : ₹{s['tp']:.0f}{RST}")
            print(f"  {'─'*55}\n")

        save = input(
            f"  {Y}  Save to journal? (y/n) : {RST}"
        ).strip().lower()
        if save == "y":
            for s in buy_signals:
                log_trade(s["ticker"], s["name"],
                          s["price"], s["sl"], s["tp"],
                          s["adx"], s["vol_r"])
            print(f"\n  {G}  ✔  Saved to {JOURNAL_FILE}{RST}\n")

    else:
        print(f"  {Y}  ◆  No BUY signals today.{RST}\n")

    # ── WATCH LIST ───────────────────────────────

    if watch_list:
        print(f"  {BOLD}{Y}  WATCH LIST — {len(watch_list)} stocks{RST}")
        print(f"  {'─'*55}")
        for w in watch_list:
            adx_label = f"ADX:{w['adx']:.0f}" \
                        f"{'✅' if w['adx'] > 25 else '⚠️'}"
            vol_label = f"Vol:{w['vol_r']:.1f}x" \
                        f"{'✅' if w['vol_r'] > 1 else '⚠️'}"
            print(f"\n  {Y}  ◈  {w['name']}{RST}"
                  f"  ₹{w['price']:.1f}"
                  f"  |  {w['score']}/5"
                  f"  |  RSI:{w['rsi']:.1f}"
                  f"  |  {adx_label}"
                  f"  |  {vol_label}")
            print(f"     {R}  Waiting : "
                  f"{', '.join(w['waiting'])}{RST}")
        print()

    # ── NO SIGNAL ────────────────────────────────

    if no_signals:
        print(f"  {D}  ○  No Signal : "
              f"{' | '.join(no_signals)}{RST}\n")

    if errors:
        print(f"  {R}  ⚠  Failed    : "
              f"{' | '.join(errors)}{RST}\n")

    show_performance()

    print(f"""
  {BOLD}{W}  STRATEGY RULES{RST}
  {'─'*50}
  {W}  Core Signal  : ALL 5 conditions must pass{RST}
  {W}  Extra Info   : ADX + Volume (bonus context){RST}
  {G}  ✔  ADX > 25  = Strong trend (better signal){RST}
  {G}  ✔  Vol > 1x  = High interest (better signal){RST}
  {'─'*50}
  {G}  ✔  SL hit    →  Close immediately{RST}
  {G}  ✔  TP hit    →  Close & celebrate{RST}
  {G}  ✔  10 days   →  Close regardless{RST}
  {R}  ✘  No signal →  No trade!{RST}
  {'─'*50}
  {D}  Next scan : Tomorrow 9:20 AM
  Command   : python daily_scanner.py{RST}
""")

if __name__ == "__main__":
    main()
