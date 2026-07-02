"""
╔══════════════════════════════════════════════════════════════╗
║     ALGO TRADING — Daily Signal Scanner v4                   ║
║     NSE India + US Stocks Combined                           ║
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
    # 🇮🇳 NSE India — Moneybhai
    "BAJFINANCE.NS" : "Bajaj Finance",
    "TITAN.NS"      : "Titan",
    "SUNPHARMA.NS"  : "Sun Pharma",
    "ADANIENT.NS"   : "Adani Ent",
    "MARUTI.NS"     : "Maruti",

    # 🇺🇸 US Stocks — TradingView
    "AAPL"  : "Apple",
    "MSFT"  : "Microsoft",
    "GOOGL" : "Google",
    "TSLA"  : "Tesla",
    "NVDA"  : "Nvidia",

    # 💱 Forex — eToro / XM
    "EURUSD=X" : "EUR/USD",
    "GBPUSD=X" : "GBP/USD",
    "USDJPY=X" : "USD/JPY",
    "USDLKR=X" : "USD/LKR",
    "AUDUSD=X" : "AUD/USD",
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

    macd          = ta.trend.MACD(close)
    d["MACD"]     = macd.macd()
    d["MACD_SIG"] = macd.macd_signal()
    d["MACD_HIST"]= macd.macd_diff()

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
# SIGNAL CHECK
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
                "Date","Stock","Ticker","Market",
                "Entry_Price","Stop_Loss","Take_Profit",
                "ADX","Vol_Ratio",
                "Exit_Date","Exit_Price","Exit_Type",
                "Return_%","Result","Notes"
            ])

def log_trade(ticker, name, price, sl, tp, adx, vol, market):
    with open(JOURNAL_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.today().strftime("%Y-%m-%d"),
            name, ticker, market,
            f"{price:.2f}", f"{sl:.2f}", f"{tp:.2f}",
            f"{adx:.1f}", f"{vol:.2f}",
            "","","","","OPEN",""
        ])

def is_already_open(ticker):
    """Check if same stock already has open trade"""
    if not os.path.exists(JOURNAL_FILE):
        return False
    try:
        df = pd.read_csv(JOURNAL_FILE)
        open_same = df[
            (df["Result"] == "OPEN") &
            (df["Ticker"] == ticker)
        ]
        return len(open_same) > 0
    except Exception:
        return False

def show_open_trades():
    if not os.path.exists(JOURNAL_FILE):
        return
    try:
        df = pd.read_csv(JOURNAL_FILE)
        open_tr = df[df["Result"] == "OPEN"]
        if len(open_tr) == 0:
            return

        print(f"\n  {BOLD}{Y}  OPEN TRADES ({len(open_tr)}){RST}")
        print(f"  {'─'*70}")
        print(f"  {D}  {'Date':<12} {'Stock':<14} {'Mkt':>5} "
              f"{'Entry':>9} {'SL':>9} {'TP':>9} {'Days':>5}{RST}")
        print(f"  {'─'*70}")

        today = datetime.today().date()
        for _, row in open_tr.iterrows():
            try:
                days  = (today - pd.to_datetime(row["Date"]).date()).days
                mkt   = str(row.get("Market","NSE"))[:4]
                warn  = "⚠️" if days > 8 else " "
                print(f"  {W}  {str(row['Date']):<12} "
                      f"{str(row['Stock']):<14} "
                      f"{mkt:>4} "
                      f"₹{float(row['Entry_Price']):>8.1f} "
                      f"{R}₹{float(row['Stop_Loss']):>8.1f}{RST} "
                      f"{G}₹{float(row['Take_Profit']):>8.1f}{RST} "
                      f"{warn}{days:>3}d")
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
        df   = pd.read_csv(JOURNAL_FILE)
        done = df[df["Result"].isin(["WIN","LOSS"])]
        open_t = df[df["Result"] == "OPEN"]

        print(f"\n  {BOLD}{C}  MODEL SUCCESS RATE TRACKER{RST}")
        print(f"  {'═'*50}")

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
        to_cnt = (done["Exit_Type"] == "manual").sum()

        # Reliability message
        if len(done) < 10:
            rel = f"{R}  Need more trades for reliable data{RST}"
        elif len(done) < 20:
            rel = f"{Y}  Getting there! Need 20+ trades{RST}"
        else:
            rel = f"{G}  Reliable data! Model proven!{RST}"

        wc = G if wr   >= 55 else Y if wr >= 50 else R
        ac = G if avg  >  0  else R
        tc = G if total > 0  else R

        print(f"  {W}  Completed Trades : {BOLD}{len(done)}{RST}  "
              f"({D}Open: {len(open_t)}{RST})")
        print(f"  {W}  Wins             : {G}{BOLD}{wins}{RST}")
        print(f"  {W}  Losses           : {R}{BOLD}{losses}{RST}")
        print(f"  {W}  Win Rate         : {wc}{BOLD}{wr:.1f}%{RST}")
        print(f"  {W}  Avg Return/Trade : {ac}{BOLD}{avg:+.2f}%{RST}")
        print(f"  {W}  Total Return     : {tc}{BOLD}{total:+.2f}%{RST}")
        print(f"  {'─'*50}")
        print(f"  {W}  TP hits : {G}{tp_cnt}{RST}  "
              f"SL hits : {R}{sl_cnt}{RST}  "
              f"Manual : {D}{to_cnt}{RST}")
        print(f"  {'─'*50}")

        # Per market breakdown
        if "Market" in done.columns:
            for mkt in done["Market"].unique():
                m_df  = done[done["Market"] == mkt]
                m_wr  = (m_df["Result"] == "WIN").mean() * 100
                m_avg = pd.to_numeric(
                    m_df["Return_%"], errors="coerce"
                ).mean()
                mc    = G if m_wr >= 55 else R
                print(f"  {D}  {mkt:<6} : "
                      f"{mc}{m_wr:.1f}% WR{RST}  "
                      f"Avg: {m_avg:+.2f}%  "
                      f"({len(m_df)} trades)")

        print(f"\n  {rel}")

        # Model verdict
        print(f"\n  {BOLD}  MODEL VERDICT:{RST}")
        if len(done) >= 20:
            if wr >= 60:
                print(f"  {G}{BOLD}  ✅ EXCELLENT! Real money consider பண்ணலாம்!{RST}")
            elif wr >= 55:
                print(f"  {G}  ✅ GOOD! Real money ready!{RST}")
            elif wr >= 50:
                print(f"  {Y}  ⚠️  AVERAGE. More paper trade continue.{RST}")
            else:
                print(f"  {R}  ❌ POOR. Strategy review வேணும்.{RST}")
        else:
            remaining = 20 - len(done)
            print(f"  {Y}  ⏳ Need {remaining} more trades to verdict!{RST}")

        print(f"  {'═'*50}\n")

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
          f"║     ALGO TRADING — Daily Signal Scanner v4       ║\n"
          f"║     {today}  {now.strftime('%I:%M %p')}  |  NSE + US       ║\n"
          f"╚══════════════════════════════════════════════════╝"
          f"{RST}\n")

    init_journal()
    show_open_trades()

    print(f"  {BOLD}{W}  Scanning {len(STOCKS)} stocks "
          f"(NSE: {len(NSE_STOCKS)} | US: {len(US_STOCKS)})...{RST}\n")

    nse_buy = []
    us_buy  = []
    watch   = []
    no_sig  = []
    errors  = []

    for ticker, name in STOCKS.items():
        market = "NSE 🇮🇳" if ticker.endswith(".NS") else \
                "FX 💱"   if "=X" in ticker else "US 🇺🇸"
        print(f"  {D}  [{market}] {name}...{RST}", end=" ", flush=True)

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
            curr  = "₹" if ticker.endswith(".NS") else \
                    ""   if "=X" in ticker else "$"


            if ok:
                sl = price * (1 - STOP_LOSS_PCT / 100)
                tp = price * (1 + TAKE_PROFIT_PCT / 100)
                signal = dict(
                    ticker=ticker, name=name,
                    price=price, sl=sl, tp=tp,
                    rsi=rsi, adx=adx, vol_r=vol_r,
                    rules=rules, last=last,
                    market="NSE" if ticker.endswith(".NS") else "US",
                    curr=curr
                )
                if ticker.endswith(".NS"):
                    nse_buy.append(signal)
                else:
                    us_buy.append(signal)
                print(f"{G}BUY SIGNAL! ({score}/5){RST}")
            elif score >= 3:
                waiting = [
                    k.split("[")[0].strip()
                    for k, v in rules.items() if not v
                ]
                watch.append(dict(
                    name=name, price=price,
                    rsi=rsi, score=score,
                    adx=adx, vol_r=vol_r,
                    waiting=waiting,
                    market="NSE" if ticker.endswith(".NS") else "US",
                    curr=curr
                ))
                print(f"{Y}Watch ({score}/5){RST}")
            else:
                no_sig.append(name)
                print(f"{D}No signal ({score}/5){RST}")

        except Exception as e:
            errors.append(name)
            print(f"{R}Error{RST}")

    all_buy = nse_buy + us_buy

    print()

    # ── BUY SIGNALS ──────────────────────────────────────

    if all_buy:
        print(f"  {BOLD}{G}"
              f"╔══════════════════════════════════════════════╗\n"
              f"  ║   ▲  BUY SIGNAL(S) FOUND  —  "
              f"{len(all_buy)} stock(s)          ║\n"
              f"  ╚══════════════════════════════════════════════╝"
              f"{RST}\n")

        for s in all_buy:
            sym     = s["ticker"].replace(".NS","")
            curr    = s["curr"]
            already = is_already_open(s["ticker"])

            mkt_label = (
                f"{B}[NSE 🇮🇳] Manual / Moneybhai{RST}"
                if s["market"] == "NSE"
                else f"{C}[US 🇺🇸] TradingView Paper Trade{RST}"
            )

            print(f"  {BOLD}{G}  ▲  {s['name']} ({sym}){RST}  {mkt_label}")
            print(f"  {'─'*55}")
            print(f"  {W}  Price  : {BOLD}{curr}{s['price']:.2f}{RST}")
            print(f"  {G}  Target : {BOLD}{curr}{s['tp']:.2f}{RST}"
                  f"  {D}(+{TAKE_PROFIT_PCT}%){RST}")
            print(f"  {R}  Stop L : {BOLD}{curr}{s['sl']:.2f}{RST}"
                  f"  {D}(-{STOP_LOSS_PCT}%){RST}")

            # ADX + Volume
            adx_icon = "✅" if s["adx"]   > 25 else "⚠️"
            vol_icon = "✅" if s["vol_r"] >  1 else "⚠️"
            print(f"\n  {D}  ADX: {s['adx']:.1f} {adx_icon}  "
                  f"Vol: {s['vol_r']:.2f}x {vol_icon}{RST}")

            # Core conditions
            print(f"\n  {BOLD}{W}  ★ 5 Core Conditions (All Pass!):{RST}")
            for rule, result in s["rules"].items():
                icon = f"{G}✔{RST}" if result else f"{R}✘{RST}"
                print(f"     {icon}  {W}{rule}{RST}")

            # Paper trade instructions
            print(f"\n  {BOLD}{Y}  ACTION:{RST}")
            if s["market"] == "NSE":
                print(f"  {W}  Platform : Moneybhai (NSE paper trade)")
                print(f"     Search  : {sym}")
                print(f"     BUY     : 1-5 shares")
            else:
                print(f"  {W}  Platform : TradingView Paper Trading")
                print(f"     Search  : {sym}")
                print(f"     BUY     : 10 shares")

            print(f"     SL Alert: {curr}{s['sl']:.0f}")
            print(f"     TP Alert: {curr}{s['tp']:.0f}{RST}")

            if already:
                print(f"\n  {Y}  ⚠️  Already have open trade for {s['name']}!")
                print(f"     Skip saving — already tracking!{RST}")
            print(f"  {'─'*55}\n")

        # Save to journal
        to_save = [s for s in all_buy if not is_already_open(s["ticker"])]
        if to_save:
            save = input(
                f"  {Y}  Save {len(to_save)} new trade(s) to journal? (y/n): {RST}"
            ).strip().lower()
            if save == "y":
                for s in to_save:
                    log_trade(
                        s["ticker"], s["name"],
                        s["price"], s["sl"], s["tp"],
                        s["adx"], s["vol_r"], s["market"]
                    )
                print(f"  {G}  ✔  Saved! {RST}\n")
        else:
            print(f"  {D}  All signals already in journal.{RST}\n")

    else:
        print(f"  {Y}  ◆  No BUY signals today.{RST}\n")

    # ── WATCH LIST ───────────────────────────────────────

    if watch:
        nse_w = [w for w in watch if w["market"] == "NSE"]
        us_w  = [w for w in watch if w["market"] == "US"]

        print(f"  {BOLD}{Y}  WATCH LIST — {len(watch)} stocks{RST}")
        print(f"  {'─'*55}")

        for w in watch:
            mkt = f"🇮🇳" if w["market"] == "NSE" else f"🇺🇸"
            adx_icon = "✅" if w["adx"]   > 25 else "⚠️"
            vol_icon = "✅" if w["vol_r"] >  1 else "⚠️"
            print(f"\n  {Y}  ◈ {mkt} {w['name']}{RST}"
                  f"  {w['curr']}{w['price']:.1f}"
                  f"  |  {w['score']}/5"
                  f"  |  RSI:{w['rsi']:.1f}"
                  f"  |  ADX:{w['adx']:.0f}{adx_icon}"
                  f"  |  Vol:{w['vol_r']:.1f}x{vol_icon}")
            print(f"     {R}  Waiting : "
                  f"{', '.join(w['waiting'])}{RST}")
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
  {BOLD}{W}  RULES{RST}
  {'─'*50}
  {B}  🇮🇳 NSE    → Moneybhai paper trade{RST}
  {C}  🇺🇸 US     → TradingView paper trade{RST}
  {'─'*50}
  {G}  ✔  All 5 core conditions = BUY signal{RST}
  {G}  ✔  SL hit  → Close immediately{RST}
  {G}  ✔  TP hit  → Close & celebrate{RST}
  {G}  ✔  10 days → Close regardless{RST}
  {R}  ✘  No signal → No trade!{RST}
  {'─'*50}
  {D}  Next scan : Tomorrow 9:20 AM
  Command   : python daily_scanner.py{RST}
""")

if __name__ == "__main__":
    main()
