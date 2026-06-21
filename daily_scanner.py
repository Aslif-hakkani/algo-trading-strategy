"""
╔══════════════════════════════════════════════════════════════╗
║           ALGO TRADING — Daily Signal Scanner                ║
║           Strategy: EMA + RSI + MACD + Bollinger Bands       ║
║           Stocks  : RELIANCE | TCS | INFY | HDFC | WIPRO     ║
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
    "RELIANCE.NS" : "Reliance",
    "TCS.NS"      : "TCS",
    "INFY.NS"     : "Infosys",
    "HDFCBANK.NS" : "HDFC Bank",
    "WIPRO.NS"    : "Wipro",
}

STOP_LOSS_PCT   = 2.0    # %
TAKE_PROFIT_PCT = 4.0    # %
HOLD_DAYS       = 10
JOURNAL_FILE    = "trade_journal.csv"
MIN_ROWS        = 210    # Need at least 210 days for EMA200

# ──────────────────────────────────────────────────────────────
# COLORS
# ──────────────────────────────────────────────────────────────

G = "\033[92m"   # Green
R = "\033[91m"   # Red
Y = "\033[93m"   # Yellow
B = "\033[94m"   # Blue
C = "\033[96m"   # Cyan
W = "\033[97m"   # White
D = "\033[90m"   # Dark/Gray
BOLD  = "\033[1m"
RESET = "\033[0m"

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
                start  = start.strftime("%Y-%m-%d"),
                end    = end.strftime("%Y-%m-%d"),
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

    # Fallback: shorter period
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
    d = df.copy()
    close = d["Close"].squeeze()
    high  = d["High"].squeeze()
    low   = d["Low"].squeeze()

    d["EMA20"]  = ta.trend.EMAIndicator(close, 20).ema_indicator()
    d["EMA50"]  = ta.trend.EMAIndicator(close, 50).ema_indicator()
    d["EMA200"] = ta.trend.EMAIndicator(close, 200).ema_indicator()
    d["RSI"]    = ta.momentum.RSIIndicator(close, 14).rsi()

    macd = ta.trend.MACD(close)
    d["MACD"]     = macd.macd()
    d["MACD_SIG"] = macd.macd_signal()

    bb = ta.volatility.BollingerBands(close)
    bh = bb.bollinger_hband()
    bl = bb.bollinger_lband()
    d["BB_POS"] = (close - bl) / (bh - bl + 1e-9)

    return d.dropna()

# ──────────────────────────────────────────────────────────────
# SIGNAL CHECK
# ──────────────────────────────────────────────────────────────

def check_signal(df):
    if len(df) < 5:
        return False, {}, None

    last = df.iloc[-1]

    rules = {
        "EMA50 > EMA200   [Uptrend Confirmed]"    : bool(last["EMA50"]  > last["EMA200"]),
        "RSI between 50-70  [Healthy Momentum]"   : bool(50 < last["RSI"] < 70),
        "MACD > Signal Line [Bullish Crossover]"  : bool(last["MACD"]  > last["MACD_SIG"]),
        "BB Position < 0.85 [Not Overbought]"     : bool(last["BB_POS"] < 0.85),
        "EMA20 > EMA50    [Short-Term Uptrend]"   : bool(last["EMA20"]  > last["EMA50"]),
    }

    return all(rules.values()), rules, last

# ──────────────────────────────────────────────────────────────
# JOURNAL
# ──────────────────────────────────────────────────────────────

def init_journal():
    if not os.path.exists(JOURNAL_FILE):
        with open(JOURNAL_FILE, "w", newline="") as f:
            csv.writer(f).writerow([
                "Date","Stock","Ticker",
                "Entry_Price","Stop_Loss","Take_Profit",
                "Exit_Date","Exit_Price","Exit_Type",
                "Return_%","Result","Notes"
            ])

def log_trade(ticker, name, price, sl, tp):
    with open(JOURNAL_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.today().strftime("%Y-%m-%d"),
            name, ticker,
            f"{price:.2f}", f"{sl:.2f}", f"{tp:.2f}",
            "","","","","OPEN",""
        ])

def show_open_trades():
    if not os.path.exists(JOURNAL_FILE):
        return
    df = pd.read_csv(JOURNAL_FILE)
    open_trades = df[df["Result"] == "OPEN"]
    if len(open_trades) == 0:
        return

    print(f"  {BOLD}{Y}  OPEN TRADES ({len(open_trades)}){RESET}")
    print(f"  {'─'*62}")
    print(f"  {D}  {'Date':<12} {'Stock':<12} {'Entry':>8} "
          f"{'SL':>8} {'TP':>8} {'Days':>5}{RESET}")
    print(f"  {'─'*62}")
    today = datetime.today().date()
    for _, row in open_trades.iterrows():
        try:
            days = (today - pd.to_datetime(row["Date"]).date()).days
            sl_color = R if days > 7 else W
            print(f"  {W}  {str(row['Date']):<12} "
                  f"{str(row['Stock']):<12} "
                  f"₹{float(row['Entry_Price']):>7.1f} "
                  f"{R}₹{float(row['Stop_Loss']):>7.1f}{RESET} "
                  f"{G}₹{float(row['Take_Profit']):>7.1f}{RESET} "
                  f"{sl_color}{days:>4}d{RESET}")
        except Exception:
            pass
    print()

def show_performance():
    if not os.path.exists(JOURNAL_FILE):
        return
    df   = pd.read_csv(JOURNAL_FILE)
    done = df[df["Result"].isin(["WIN","LOSS"])]
    if len(done) == 0:
        print(f"  {D}  No completed trades yet.{RESET}")
        return

    wins   = (done["Result"] == "WIN").sum()
    losses = (done["Result"] == "LOSS").sum()
    wr     = wins / len(done) * 100
    rets   = pd.to_numeric(done["Return_%"], errors="coerce")
    avg    = rets.mean()
    total  = rets.sum()
    wr_col = G if wr >= 50 else R
    av_col = G if avg > 0 else R

    print(f"  {BOLD}{C}  PERFORMANCE SUMMARY{RESET}")
    print(f"  {'─'*40}")
    print(f"  {W}  Total Trades  : {BOLD}{len(done)}{RESET}")
    print(f"  {G}  Wins          : {BOLD}{wins}{RESET}")
    print(f"  {R}  Losses        : {BOLD}{losses}{RESET}")
    print(f"  {W}  Win Rate      : {wr_col}{BOLD}{wr:.1f}%{RESET}")
    print(f"  {W}  Avg Return    : {av_col}{BOLD}{avg:+.2f}%{RESET}")
    print(f"  {W}  Total Return  : {av_col}{BOLD}{total:+.2f}%{RESET}")

# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    now   = datetime.now()
    today = now.strftime("%Y-%m-%d")

    print(f"\n{BOLD}{B}"
          f"╔══════════════════════════════════════════════════╗\n"
          f"║        ALGO TRADING — Daily Signal Scanner       ║\n"
          f"║        {today}  {now.strftime('%I:%M %p')}  |  NSE India              ║\n"
          f"╚══════════════════════════════════════════════════╝"
          f"{RESET}\n")

    init_journal()
    show_open_trades()

    print(f"  {BOLD}{W}  Scanning {len(STOCKS)} stocks...{RESET}\n")

    buy_signals = []
    watch_list  = []
    no_signals  = []
    errors      = []

    for ticker, name in STOCKS.items():
        print(f"  {D}  [{ticker.replace('.NS','')}] Fetching data...{RESET}",
              end="\r")

        df = download_data(ticker)

        if df.empty:
            errors.append(name)
            print(f"  {R}  [{ticker.replace('.NS','')}] "
                  f"Data unavailable — skipping{RESET}")
            continue

        try:
            df = add_indicators(df)
            ok, rules, last = check_signal(df)

            price      = float(last["Close"])
            rsi        = float(last["RSI"])
            ema_gap    = float(
                (last["EMA50"] - last["EMA200"])
                / last["EMA200"] * 100
            )
            score      = sum(rules.values())

            if ok:
                sl = price * (1 - STOP_LOSS_PCT   / 100)
                tp = price * (1 + TAKE_PROFIT_PCT / 100)
                buy_signals.append(dict(
                    ticker=ticker, name=name,
                    price=price, sl=sl, tp=tp,
                    rsi=rsi, ema_gap=ema_gap, rules=rules
                ))
            elif score >= 3:
                waiting = [
                    k.split("[")[0].strip().replace("  "," ")
                    for k, v in rules.items() if not v
                ]
                watch_list.append(dict(
                    name=name, price=price,
                    rsi=rsi, score=score,
                    rules=rules, waiting=waiting
                ))
            else:
                no_signals.append(dict(
                    name=name, price=price, score=score
                ))

        except Exception as e:
            errors.append(name)
            print(f"  {R}  [{name}] Error: {e}{RESET}")

    # ── BUY SIGNALS ──────────────────────────────────────

    print()
    if buy_signals:
        print(f"  {BOLD}{G}"
              f"╔══════════════════════════════════════════════╗\n"
              f"  ║   ▲  BUY SIGNAL(S) FOUND  —  "
              f"{len(buy_signals)} stock(s)          ║\n"
              f"  ╚══════════════════════════════════════════════╝"
              f"{RESET}\n")

        for s in buy_signals:
            sym = s['ticker'].replace('.NS','')
            print(f"  {BOLD}{G}  ▲  {s['name']}  ({sym}){RESET}")
            print(f"  {'─'*55}")
            print(f"  {W}  Current Price  : {BOLD}₹{s['price']:.2f}{RESET}")
            print(f"  {G}  Take Profit    : {BOLD}₹{s['tp']:.2f}{RESET}"
                  f"  {D}(+{TAKE_PROFIT_PCT}%){RESET}")
            print(f"  {R}  Stop Loss      : {BOLD}₹{s['sl']:.2f}{RESET}"
                  f"  {D}(-{STOP_LOSS_PCT}%){RESET}")
            print(f"  {W}  RSI            : {s['rsi']:.1f}"
                  f"   EMA Gap : {s['ema_gap']:+.1f}%")
            print(f"\n  {D}  Signal Conditions:{RESET}")
            for rule, result in s["rules"].items():
                icon = f"{G}  ✔{RESET}" if result else f"{R}  ✘{RESET}"
                print(f"     {icon}  {W}{rule}{RESET}")

            print(f"\n  {BOLD}{Y}  ACTION STEPS:{RESET}")
            print(f"  {W}  1. Open TradingView  →  Paper Trading")
            print(f"     2. Search : {BOLD}{sym}{RESET}")
            print(f"     3. BUY    : Market Order   (10 shares)")
            print(f"     4. Alert  : Price < ₹{s['sl']:.0f}  "
                  f"[Stop Loss]")
            print(f"     5. Alert  : Price > ₹{s['tp']:.0f}  "
                  f"[Take Profit]{RESET}")
            print(f"  {'─'*55}\n")

        save = input(
            f"  {Y}  Save to trade journal? (y/n) : {RESET}"
        ).strip().lower()

        if save == "y":
            for s in buy_signals:
                log_trade(s["ticker"], s["name"],
                          s["price"], s["sl"], s["tp"])
            print(f"\n  {G}  ✔  Saved to {JOURNAL_FILE}{RESET}\n")

    else:
        print(f"  {Y}  ◆  No BUY signals today.{RESET}\n")

    # ── WATCH LIST ───────────────────────────────────────

    if watch_list:
        print(f"  {BOLD}{Y}  WATCH LIST  —  "
              f"{len(watch_list)} stock(s) almost ready{RESET}")
        print(f"  {'─'*55}")
        for w in watch_list:
            print(f"\n  {Y}  ◈  {w['name']}{RESET}"
                  f"  —  ₹{w['price']:.2f}"
                  f"  |  {w['score']}/5 rules"
                  f"  |  RSI: {w['rsi']:.1f}")
            print(f"  {D}     Waiting for : "
                  f"{', '.join(w['waiting'])}{RESET}")
        print()

    # ── NO SIGNAL ────────────────────────────────────────

    if no_signals:
        names = "  |  ".join([s["name"] for s in no_signals])
        print(f"  {D}  ○  No Signal   :  {names}{RESET}\n")

    if errors:
        names = "  |  ".join(errors)
        print(f"  {R}  ⚠  Data Failed :  {names}  "
              f"(retry in 5 min){RESET}\n")

    # ── PERFORMANCE ──────────────────────────────────────

    show_performance()

    # ── FOOTER ───────────────────────────────────────────

    print(f"""
  {BOLD}{W}  TRADING RULES{RESET}
  {'─'*50}
  {G}  ✔  Stop Loss hit     →  Close immediately{RESET}
  {G}  ✔  Take Profit hit   →  Close & move on{RESET}
  {G}  ✔  After {HOLD_DAYS} days       →  Close regardless{RESET}
  {R}  ✘  Do NOT move Stop Loss on emotion{RESET}
  {R}  ✘  Do NOT trade without a signal{RESET}
  {'─'*50}
  {D}  Next scan  :  Tomorrow at 9:20 AM
  Command    :  python daily_scanner.py{RESET}
""")


if __name__ == "__main__":
    main()