"""
╔══════════════════════════════════════════════════════════════╗
║     ALGO TRADING — Daily Signal Scanner v5                   ║
║     NSE India (10) + US Stocks (10) = 20 stocks             ║
║     With Model Success Rate Tracker                          ║
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
    # 🇮🇳 NSE India (10 stocks) — Moneybhai Paper Trade
    "TCS.NS"        : "TCS",
    "INFY.NS"       : "Infosys",
    "HDFCBANK.NS"   : "HDFC Bank",
    "ICICIBANK.NS"  : "ICICI Bank",
    "TITAN.NS"      : "Titan",
    "RELIANCE.NS"   : "Reliance",
    "SUNPHARMA.NS"  : "Sun Pharma",
    "LT.NS"         : "L&T",
    "ASIANPAINT.NS" : "Asian Paints",
    "DIVISLAB.NS"   : "Divi's Lab",

    # 🇺🇸 US Stocks (10 stocks) — TradingView Paper Trade
    "AAPL"  : "Apple",
    "MSFT"  : "Microsoft",
    "GOOGL" : "Google",
    "AMZN"  : "Amazon",
    "NVDA"  : "Nvidia",
    "META"  : "Meta",
    "AMD"   : "AMD",
    "JPM"   : "JPMorgan",
    "V"     : "Visa",
    "LLY"   : "Eli Lilly",
}

NSE_STOCKS = [k for k in STOCKS if k.endswith(".NS")]
US_STOCKS  = [k for k in STOCKS if not k.endswith(".NS")]

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

    d["EMA20"]  = ta.trend.EMAIndicator(close, 20).ema_indicator()
    d["EMA50"]  = ta.trend.EMAIndicator(close, 50).ema_indicator()
    d["EMA200"] = ta.trend.EMAIndicator(close, 200).ema_indicator()
    d["RSI"]    = ta.momentum.RSIIndicator(close, 14).rsi()

    macd           = ta.trend.MACD(close)
    d["MACD"]      = macd.macd()
    d["MACD_SIG"]  = macd.macd_signal()
    d["MACD_HIST"] = macd.macd_diff()

    bb          = ta.volatility.BollingerBands(close)
    d["BB_POS"] = (close - bb.bollinger_lband()) / \
                  (bb.bollinger_hband() - bb.bollinger_lband() + 1e-9)

    adx        = ta.trend.ADXIndicator(high, low, close, 14)
    d["ADX"]   = adx.adx()

    d["VOL_MA20"]  = vol.rolling(20).mean()
    d["VOL_RATIO"] = vol / (d["VOL_MA20"] + 1)
    d["EMA_GAP"]   = (d["EMA50"] - d["EMA200"]) / d["EMA200"] * 100

    return d.dropna()

# ──────────────────────────────────────────────────────────────
# SIGNAL CHECK — 5 CORE CONDITIONS
# ──────────────────────────────────────────────────────────────

def check_signal(df):
    if len(df) < 5:
        return False, {}, None

    last = df.iloc[-1]

    core_rules = {
        "EMA50 > EMA200   [Uptrend]"          : bool(last["EMA50"]  > last["EMA200"]),
        "RSI 50-70        [Momentum]"          : bool(50 < last["RSI"] < 70),
        "MACD > Signal    [Bullish]"           : bool(last["MACD"]   > last["MACD_SIG"]),
        "BB Position < 0.85 [Not Overbought]"  : bool(last["BB_POS"] < 0.85),
        "EMA20 > EMA50    [Short Uptrend]"     : bool(last["EMA20"]  > last["EMA50"]),
    }

    return all(core_rules.values()), core_rules, last

# ──────────────────────────────────────────────────────────────
# JOURNAL
# ──────────────────────────────────────────────────────────────

def init_journal():
    if not os.path.exists(JOURNAL_FILE):
        with open(JOURNAL_FILE, "w", newline="") as f:
            csv.writer(f).writerow([
                "Date", "Stock", "Ticker", "Market",
                "Entry_Price", "Stop_Loss", "Take_Profit",
                "ADX", "Vol_Ratio",
                "Exit_Date", "Exit_Price", "Exit_Type",
                "Return_%", "Result", "Notes"
            ])

def log_trade(ticker, name, price, sl, tp, adx, vol, market):
    with open(JOURNAL_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.today().strftime("%Y-%m-%d"),
            name, ticker, market,
            f"{price:.2f}", f"{sl:.2f}", f"{tp:.2f}",
            f"{adx:.1f}", f"{vol:.2f}",
            "", "", "", "", "OPEN", ""
        ])

def is_already_open(ticker):
    if not os.path.exists(JOURNAL_FILE):
        return False
    try:
        df = pd.read_csv(JOURNAL_FILE)
        return len(df[(df["Result"] == "OPEN") &
                      (df["Ticker"] == ticker)]) > 0
    except Exception:
        return False

def show_open_trades():
    if not os.path.exists(JOURNAL_FILE):
        return
    try:
        df      = pd.read_csv(JOURNAL_FILE)
        open_tr = df[df["Result"] == "OPEN"]
        if len(open_tr) == 0:
            return

        print(f"\n  {BOLD}{Y}  OPEN TRADES ({len(open_tr)}){RST}")
        print(f"  {'─'*72}")
        print(f"  {D}  {'Date':<12} {'Stock':<14} {'Mkt':>4} "
              f"{'Entry':>9} {'SL':>9} {'TP':>9} {'Days':>5}{RST}")
        print(f"  {'─'*72}")

        today = datetime.today().date()
        for _, row in open_tr.iterrows():
            try:
                days = (today - pd.to_datetime(row["Date"]).date()).days
                mkt  = str(row.get("Market", "NSE"))[:3]
                curr = "₹" if mkt == "NSE" else "$"
                warn = "⚠️ " if days > 8 else "   "
                print(f"  {W}  {str(row['Date']):<12} "
                      f"{str(row['Stock']):<14} "
                      f"{mkt:>4} "
                      f"{curr}{float(row['Entry_Price']):>8.1f} "
                      f"{R}{curr}{float(row['Stop_Loss']):>8.1f}{RST} "
                      f"{G}{curr}{float(row['Take_Profit']):>8.1f}{RST} "
                      f"{warn}{days:>2}d")
            except Exception:
                pass
        print()
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────
# SUCCESS RATE TRACKER
# ──────────────────────────────────────────────────────────────

def show_success_rate():
    if not os.path.exists(JOURNAL_FILE):
        return
    try:
        df     = pd.read_csv(JOURNAL_FILE)
        done   = df[df["Result"].isin(["WIN", "LOSS"])]
        open_t = df[df["Result"] == "OPEN"]

        print(f"\n  {BOLD}{C}  MODEL SUCCESS RATE TRACKER{RST}")
        print(f"  {'═'*52}")

        if len(done) == 0:
            print(f"  {D}  No completed trades yet.")
            print(f"  {D}  Need 20+ trades for reliable data!{RST}\n")
            return

        wins   = (done["Result"] == "WIN").sum()
        losses = (done["Result"] == "LOSS").sum()
        wr     = wins / len(done) * 100
        rets   = pd.to_numeric(done["Return_%"], errors="coerce")
        avg    = rets.mean()
        total  = rets.sum()
        tp_cnt = (done["Exit_Type"] == "TP").sum()
        sl_cnt = (done["Exit_Type"] == "SL").sum()

        wc = G if wr  >= 55 else Y if wr >= 50 else R
        ac = G if avg >   0 else R
        tc = G if total > 0 else R

        print(f"  {W}  Completed : {BOLD}{len(done)}{RST}  "
              f"({D}Open: {len(open_t)}{RST})")
        print(f"  {G}  Wins      : {BOLD}{wins}{RST}  "
              f"{R}Losses: {BOLD}{losses}{RST}")
        print(f"  {W}  Win Rate  : {wc}{BOLD}{wr:.1f}%{RST}")
        print(f"  {W}  Avg Return: {ac}{BOLD}{avg:+.2f}%{RST}")
        print(f"  {W}  Total     : {tc}{BOLD}{total:+.2f}%{RST}")
        print(f"  {'─'*52}")
        print(f"  {W}  TP: {G}{tp_cnt}{RST}  SL: {R}{sl_cnt}{RST}")

        # Per market
        if "Market" in done.columns:
            print(f"  {'─'*52}")
            for mkt in done["Market"].unique():
                m    = done[done["Market"] == mkt]
                m_wr = (m["Result"] == "WIN").mean() * 100
                m_av = pd.to_numeric(
                    m["Return_%"], errors="coerce").mean()
                mc   = G if m_wr >= 55 else R
                print(f"  {D}  {mkt:<4}: "
                      f"{mc}{m_wr:.1f}% WR{RST}  "
                      f"Avg:{m_av:+.2f}%  "
                      f"({len(m)} trades)")

        print(f"  {'─'*52}")

        # Verdict
        if len(done) >= 20:
            if wr >= 60:
                print(f"  {G}{BOLD}  ✅ EXCELLENT! Real money ready!{RST}")
            elif wr >= 55:
                print(f"  {G}  ✅ GOOD! Consider real money.{RST}")
            elif wr >= 50:
                print(f"  {Y}  ⚠️  AVERAGE. Continue paper trade.{RST}")
            else:
                print(f"  {R}  ❌ POOR. Strategy review needed.{RST}")
        else:
            rem = 20 - len(done)
            print(f"  {Y}  ⏳ Need {rem} more trades for verdict!{RST}")

        print(f"  {'═'*52}\n")

    except Exception as e:
        print(f"  {R}  Journal error: {e}{RST}\n")

# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    now   = datetime.now()
    today = now.strftime("%Y-%m-%d")

    print(f"\n{BOLD}{B}"
          f"╔══════════════════════════════════════════════════╗\n"
          f"║   ALGO TRADING — Daily Signal Scanner v5         ║\n"
          f"║   {today}  {now.strftime('%I:%M %p')}                     ║\n"
          f"║   NSE India (10) + US Stocks (10)                ║\n"
          f"╚══════════════════════════════════════════════════╝"
          f"{RST}\n")

    init_journal()
    show_open_trades()

    print(f"  {BOLD}{W}  Scanning {len(STOCKS)} stocks "
          f"({B}NSE: {len(NSE_STOCKS)}{RST} | "
          f"{C}US: {len(US_STOCKS)}{RST})...\n")

    nse_buy = []
    us_buy  = []
    watch   = []
    no_sig  = []
    errors  = []

    for ticker, name in STOCKS.items():
        is_nse = ticker.endswith(".NS")
        flag   = "🇮🇳" if is_nse else "🇺🇸"
        mkt_l  = "NSE" if is_nse else "US"
        curr   = "₹"   if is_nse else "$"

        print(f"  {D}  [{flag} {mkt_l}] {name:15s}...{RST}",
              end=" ", flush=True)

        df = download_data(ticker)
        if df.empty:
            errors.append(name)
            print(f"{R}FAILED{RST}")
            continue

        try:
            df = add_indicators(df)
            ok, rules, last = check_signal(df)

            price = float(last["Close"])
            rsi   = float(last["RSI"])
            adx   = float(last["ADX"])
            vol_r = float(last["VOL_RATIO"])
            score = sum(rules.values())

            if ok:
                sl = price * (1 - STOP_LOSS_PCT / 100)
                tp = price * (1 + TAKE_PROFIT_PCT / 100)
                sig = dict(
                    ticker=ticker, name=name,
                    price=price, sl=sl, tp=tp,
                    rsi=rsi, adx=adx, vol_r=vol_r,
                    rules=rules, last=last,
                    market=mkt_l, curr=curr
                )
                if is_nse:
                    nse_buy.append(sig)
                else:
                    us_buy.append(sig)
                print(f"{G}{BOLD}BUY SIGNAL! ({score}/5){RST}")

            elif score >= 3:
                waiting = [
                    k.split("[")[0].strip()
                    for k, v in rules.items() if not v
                ]
                watch.append(dict(
                    name=name, price=price, curr=curr,
                    rsi=rsi, score=score,
                    adx=adx, vol_r=vol_r,
                    waiting=waiting, market=mkt_l
                ))
                print(f"{Y}Watch ({score}/5){RST}")
            else:
                no_sig.append(name)
                print(f"{D}No signal ({score}/5){RST}")

        except Exception:
            errors.append(name)
            print(f"{R}Error{RST}")

    all_buy = nse_buy + us_buy
    print()

    # ── BUY SIGNALS ──────────────────────────────────────

    if all_buy:
        print(f"\n  {BOLD}{G}"
              f"╔══════════════════════════════════════════════╗\n"
              f"  ║   ▲  BUY SIGNAL(S) — {len(all_buy)} stock(s) found"
              f"              ║\n"
              f"  ╚══════════════════════════════════════════════╝"
              f"{RST}\n")

        for s in all_buy:
            sym     = s["ticker"].replace(".NS", "")
            curr    = s["curr"]
            already = is_already_open(s["ticker"])
            adx_i   = "✅" if s["adx"]   > 25 else "⚠️"
            vol_i   = "✅" if s["vol_r"] >  1 else "⚠️"

            mkt_lbl = (f"{B}[NSE 🇮🇳] Moneybhai{RST}"
                       if s["market"] == "NSE"
                       else f"{C}[US 🇺🇸] TradingView{RST}")

            print(f"  {BOLD}{G}  ▲  {s['name']} ({sym}){RST}"
                  f"  {mkt_lbl}")
            print(f"  {'─'*55}")
            print(f"  {W}  Price  : {BOLD}{curr}{s['price']:.2f}{RST}")
            print(f"  {G}  Target : {BOLD}{curr}{s['tp']:.2f}{RST}"
                  f"  {D}(+{TAKE_PROFIT_PCT:.0f}%){RST}")
            print(f"  {R}  Stop L : {BOLD}{curr}{s['sl']:.2f}{RST}"
                  f"  {D}(-{STOP_LOSS_PCT:.0f}%){RST}")
            print(f"\n  {D}  ADX: {s['adx']:.1f}{adx_i}  "
                  f"Vol: {s['vol_r']:.2f}x{vol_i}  "
                  f"RSI: {s['rsi']:.1f}{RST}")

            print(f"\n  {BOLD}{W}  ★ 5 Core Conditions:{RST}")
            for rule, result in s["rules"].items():
                icon = f"{G}✔{RST}" if result else f"{R}✘{RST}"
                print(f"     {icon}  {W}{rule}{RST}")

            print(f"\n  {BOLD}{Y}  ACTION:{RST}")
            if s["market"] == "NSE":
                print(f"  {W}  Platform: Moneybhai")
                print(f"     Search : {sym}")
                print(f"     BUY    : 1-5 shares")
            else:
                print(f"  {W}  Platform: TradingView Paper Trade")
                print(f"     Search : {sym}")
                print(f"     BUY    : 5-10 shares")

            print(f"     SL Alert: {curr}{s['sl']:.0f}")
            print(f"     TP Alert: {curr}{s['tp']:.0f}{RST}")

            if already:
                print(f"\n  {Y}  ⚠️  Already tracking {s['name']}!"
                      f" Skip save.{RST}")
            print(f"  {'─'*55}\n")

        to_save = [s for s in all_buy
                   if not is_already_open(s["ticker"])]
        if to_save:
            save = input(
                f"  {Y}  Save {len(to_save)} new trade(s)? (y/n): {RST}"
            ).strip().lower()
            if save == "y":
                for s in to_save:
                    log_trade(s["ticker"], s["name"],
                              s["price"], s["sl"], s["tp"],
                              s["adx"], s["vol_r"], s["market"])
                print(f"  {G}  ✔  Saved!{RST}\n")
        else:
            print(f"  {D}  All signals already in journal.{RST}\n")

    else:
        print(f"  {Y}  ◆  No BUY signals today.{RST}\n")

    # ── WATCH LIST ───────────────────────────────────────

    if watch:
        nse_w = [w for w in watch if w["market"] == "NSE"]
        us_w  = [w for w in watch if w["market"] == "US"]

        print(f"  {BOLD}{Y}  WATCH LIST — {len(watch)} stocks{RST}")
        print(f"  {'─'*58}")

        if nse_w:
            print(f"\n  {B}  NSE India 🇮🇳{RST}")
            for w in nse_w:
                ai = "✅" if w["adx"]   > 25 else "⚠️"
                vi = "✅" if w["vol_r"] >  1 else "⚠️"
                print(f"  {Y}  ◈ {w['name']:14s}{RST}"
                      f"  {w['curr']}{w['price']:.1f}"
                      f"  {w['score']}/5"
                      f"  RSI:{w['rsi']:.1f}"
                      f"  ADX:{w['adx']:.0f}{ai}"
                      f"  Vol:{w['vol_r']:.1f}x{vi}")
                print(f"     {R}  → {', '.join(w['waiting'])}{RST}")

        if us_w:
            print(f"\n  {C}  US Stocks 🇺🇸{RST}")
            for w in us_w:
                ai = "✅" if w["adx"]   > 25 else "⚠️"
                vi = "✅" if w["vol_r"] >  1 else "⚠️"
                print(f"  {Y}  ◈ {w['name']:14s}{RST}"
                      f"  {w['curr']}{w['price']:.2f}"
                      f"  {w['score']}/5"
                      f"  RSI:{w['rsi']:.1f}"
                      f"  ADX:{w['adx']:.0f}{ai}"
                      f"  Vol:{w['vol_r']:.1f}x{vi}")
                print(f"     {R}  → {', '.join(w['waiting'])}{RST}")
        print()

    # ── NO SIGNAL ────────────────────────────────────────

    if no_sig:
        print(f"  {D}  ○  No Signal : {' | '.join(no_sig)}{RST}\n")

    if errors:
        print(f"  {R}  ⚠  Failed   : {' | '.join(errors)}{RST}\n")

    # ── SUCCESS RATE ─────────────────────────────────────

    show_success_rate()

    # ── FOOTER ───────────────────────────────────────────

    print(f"""
  {BOLD}{W}  STRATEGY{RST}
  {'─'*50}
  {B}  🇮🇳 NSE  → Moneybhai paper trade{RST}
  {C}  🇺🇸 US   → TradingView paper trade{RST}
  {'─'*50}
  {W}  ★  All 5 conditions pass = BUY signal
  ➕  ADX > 25 + Vol > 1x = Stronger signal{RST}
  {'─'*50}
  {G}  ✔  SL hit   → Close immediately
  ✔  TP hit   → Close & celebrate
  ✔  10 days  → Close regardless{RST}
  {R}  ✘  No signal → No trade!{RST}
  {'─'*50}
  {D}  Stocks   : NSE(10) + US(10) = 20
  Next scan: Tomorrow 9:20 AM
  Command  : python daily_scanner.py{RST}
""")


if __name__ == "__main__":
    main()
