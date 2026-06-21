"""
Daily Signal Scanner — GitHub Actions Version
No colors, no input() — pure text output
"""

import yfinance as yf
import pandas as pd
import ta
from datetime import datetime, timedelta
import warnings
import time
import sys
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

STOCKS = {
    "RELIANCE.NS" : "Reliance",
    "TCS.NS"      : "TCS",
    "INFY.NS"     : "Infosys",
    "HDFCBANK.NS" : "HDFC Bank",
    "WIPRO.NS"    : "Wipro",
}

STOP_LOSS_PCT   = 2.0
TAKE_PROFIT_PCT = 4.0

# ──────────────────────────────────────────────
# DATA DOWNLOAD
# ──────────────────────────────────────────────

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
            if len(df) >= 210:
                return df
        except Exception:
            time.sleep(2)

    try:
        df = yf.Ticker(ticker).history(period="2y")
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[["Open","High","Low","Close","Volume"]].copy()
        df.dropna(inplace=True)
        if len(df) >= 210:
            return df
    except Exception:
        pass

    return pd.DataFrame()

# ──────────────────────────────────────────────
# INDICATORS
# ──────────────────────────────────────────────

def add_indicators(df):
    d     = df.copy()
    close = d["Close"].squeeze()
    high  = d["High"].squeeze()
    low   = d["Low"].squeeze()

    d["EMA20"]  = ta.trend.EMAIndicator(close, 20).ema_indicator()
    d["EMA50"]  = ta.trend.EMAIndicator(close, 50).ema_indicator()
    d["EMA200"] = ta.trend.EMAIndicator(close, 200).ema_indicator()
    d["RSI"]    = ta.momentum.RSIIndicator(close, 14).rsi()

    macd          = ta.trend.MACD(close)
    d["MACD"]     = macd.macd()
    d["MACD_SIG"] = macd.macd_signal()

    bb          = ta.volatility.BollingerBands(close)
    d["BB_POS"] = (close - bb.bollinger_lband()) / \
                  (bb.bollinger_hband() - bb.bollinger_lband() + 1e-9)

    return d.dropna()

# ──────────────────────────────────────────────
# SIGNAL CHECK
# ──────────────────────────────────────────────

def check_signal(df):
    if len(df) < 5:
        return False, {}, None

    last = df.iloc[-1]
    rules = {
        "EMA50 > EMA200  [Uptrend]"        : bool(last["EMA50"]  > last["EMA200"]),
        "RSI 50-70       [Momentum]"       : bool(50 < last["RSI"] < 70),
        "MACD > Signal   [Bullish]"        : bool(last["MACD"]   > last["MACD_SIG"]),
        "BB Position < 0.85 [Not Overbought]": bool(last["BB_POS"] < 0.85),
        "EMA20 > EMA50   [Short Uptrend]"  : bool(last["EMA20"]  > last["EMA50"]),
    }
    return all(rules.values()), rules, last

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    now   = datetime.now()
    today = now.strftime("%Y-%m-%d")

    print("=" * 55)
    print("  ALGO TRADING — Daily Signal Scanner")
    print(f"  Date    : {today}")
    print(f"  Time    : {now.strftime('%I:%M %p')} IST")
    print(f"  Stocks  : {', '.join(STOCKS.values())}")
    print("=" * 55)
    print()

    buy_signals = []
    watch_list  = []
    no_signals  = []
    errors      = []

    for ticker, name in STOCKS.items():
        print(f"  Scanning {name}...", end=" ", flush=True)

        df = download_data(ticker)

        if df.empty:
            print("FAILED (no data)")
            errors.append(name)
            continue

        try:
            df = add_indicators(df)
            ok, rules, last = check_signal(df)

            price   = float(last["Close"])
            rsi     = float(last["RSI"])
            ema_gap = float((last["EMA50"] - last["EMA200"])
                            / last["EMA200"] * 100)
            score   = sum(rules.values())

            if ok:
                sl = price * (1 - STOP_LOSS_PCT   / 100)
                tp = price * (1 + TAKE_PROFIT_PCT / 100)
                buy_signals.append(dict(
                    name=name, price=price,
                    sl=sl, tp=tp, rsi=rsi,
                    ema_gap=ema_gap, rules=rules,
                    ticker=ticker
                ))
                print(f"BUY SIGNAL! ({score}/5 rules)")

            elif score >= 3:
                waiting = [
                    k.split("[")[0].strip()
                    for k, v in rules.items() if not v
                ]
                watch_list.append(dict(
                    name=name, price=price,
                    rsi=rsi, score=score,
                    waiting=waiting
                ))
                print(f"Watch ({score}/5 rules)")

            else:
                no_signals.append(name)
                print(f"No signal ({score}/5 rules)")

        except Exception as e:
            errors.append(name)
            print(f"Error: {e}")

    # ── RESULTS ────────────────────────────────────

    print()
    print("=" * 55)
    print("  SCAN RESULTS")
    print("=" * 55)

    # BUY SIGNALS
    if buy_signals:
        print(f"\n  *** {len(buy_signals)} BUY SIGNAL(S) FOUND ***\n")
        for s in buy_signals:
            sym = s["ticker"].replace(".NS","")
            print(f"  [BUY] {s['name']} ({sym})")
            print(f"  {'─'*45}")
            print(f"  Current Price  : Rs {s['price']:.2f}")
            print(f"  Take Profit    : Rs {s['tp']:.2f}  (+{TAKE_PROFIT_PCT}%)")
            print(f"  Stop Loss      : Rs {s['sl']:.2f}  (-{STOP_LOSS_PCT}%)")
            print(f"  RSI            : {s['rsi']:.1f}")
            print(f"  EMA Gap        : {s['ema_gap']:+.1f}%")
            print(f"\n  Signal Conditions:")
            for rule, result in s["rules"].items():
                icon = "PASS" if result else "FAIL"
                print(f"    [{icon}] {rule}")
            print(f"\n  ACTION:")
            print(f"    Open TradingView Paper Trading")
            print(f"    Search : {sym}")
            print(f"    BUY    : 10 shares at market price")
            print(f"    Set SL Alert : Rs {s['sl']:.0f}")
            print(f"    Set TP Alert : Rs {s['tp']:.0f}")
            print()
    else:
        print(f"\n  No BUY signals today.\n")

    # WATCH LIST
    if watch_list:
        print(f"  WATCH LIST ({len(watch_list)} stocks):")
        print(f"  {'─'*45}")
        for w in watch_list:
            print(f"  {w['name']:12s} | Rs {w['price']:8.2f} | "
                  f"{w['score']}/5 rules | RSI: {w['rsi']:.1f}")
            print(f"    Waiting for: {', '.join(w['waiting'])}")
        print()

    # NO SIGNALS
    if no_signals:
        print(f"  No Signal  : {' | '.join(no_signals)}")

    # ERRORS
    if errors:
        print(f"  Data Error : {' | '.join(errors)}")

    # SUMMARY
    print()
    print("=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print(f"  BUY Signals : {len(buy_signals)}")
    print(f"  Watch List  : {len(watch_list)}")
    print(f"  No Signal   : {len(no_signals)}")
    print(f"  Errors      : {len(errors)}")
    print()
    print("  RULES REMINDER:")
    print("  [1] Stop Loss hit     -> Close immediately")
    print("  [2] Take Profit hit   -> Close & celebrate")
    print("  [3] After 10 days     -> Close regardless")
    print("  [4] No signal today   -> No trade!")
    print()
    print(f"  Next scan: Tomorrow 9:20 AM (auto)")
    print("=" * 55)

    # Exit code
    sys.exit(0)

if __name__ == "__main__":
    main()
