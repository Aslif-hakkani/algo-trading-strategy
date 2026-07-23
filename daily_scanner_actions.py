"""
Daily Signal Scanner v6 — GitHub Actions Version
NSE India (10) + US Stocks (10) + Crypto (5) = 25 stocks
No colors, no input() — pure text output for GitHub logs
"""

import yfinance as yf
import pandas as pd
import ta
from datetime import datetime, timedelta
import warnings
import time
import sys
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────

STOCKS = {
    # 🇮🇳 NSE India (10) — Moneybhai Paper Trade
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

    # 🇺🇸 US Stocks (10) — TradingView Paper Trade
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

    # 🪙 Crypto (5) — Bybit Demo Trade
    "BTC-USD" : "Bitcoin",
    "ETH-USD" : "Ethereum",
    "BNB-USD" : "BNB",
    "SOL-USD" : "Solana",
    "ADA-USD" : "Cardano",
}

STOCK_SL  = 2.0
STOCK_TP  = 4.0
CRYPTO_SL = 5.0
CRYPTO_TP = 15.0
MIN_ROWS  = 210

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def get_market(ticker):
    if ticker.endswith(".NS"):
        return "NSE", "🇮🇳", "Rs"
    elif "-USD" in ticker:
        return "CRYPTO", "🪙", "$"
    else:
        return "US", "🇺🇸", "$"

def get_sl_tp(ticker, price):
    if "-USD" in ticker:
        sl = price * (1 - CRYPTO_SL / 100)
        tp = price * (1 + CRYPTO_TP / 100)
        return sl, tp, CRYPTO_SL, CRYPTO_TP
    else:
        sl = price * (1 - STOCK_SL / 100)
        tp = price * (1 + STOCK_TP / 100)
        return sl, tp, STOCK_SL, STOCK_TP

def get_platform(market):
    if market == "NSE":
        return "Moneybhai (NSE paper trade)"
    elif market == "CRYPTO":
        return "Bybit Demo (Crypto paper trade)"
    else:
        return "TradingView (US paper trade)"

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
# SIGNAL CHECK
# ──────────────────────────────────────────────────────────────

def check_signal(df):
    if len(df) < 5:
        return False, {}, None
    last  = df.iloc[-1]
    rules = {
        "EMA50 > EMA200   [Uptrend]"         : bool(last["EMA50"]  > last["EMA200"]),
        "RSI 50-70        [Momentum]"         : bool(50 < last["RSI"] < 70),
        "MACD > Signal    [Bullish]"          : bool(last["MACD"]   > last["MACD_SIG"]),
        "BB Position < 0.85 [Not Overbought]" : bool(last["BB_POS"] < 0.85),
        "EMA20 > EMA50    [Short Uptrend]"    : bool(last["EMA20"]  > last["EMA50"]),
    }
    return all(rules.values()), rules, last

# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    now   = datetime.now()
    today = now.strftime("%Y-%m-%d")

    nse_c    = sum(1 for k in STOCKS if k.endswith(".NS"))
    us_c     = sum(1 for k in STOCKS
                   if not k.endswith(".NS") and "-USD" not in k)
    crypto_c = sum(1 for k in STOCKS if "-USD" in k)

    print("=" * 60)
    print("  ALGO TRADING — Daily Signal Scanner v6")
    print(f"  Date   : {today}")
    print(f"  Time   : {now.strftime('%I:%M %p')} IST")
    print(f"  Stocks : NSE({nse_c}) + US({us_c}) + Crypto({crypto_c})"
          f" = {len(STOCKS)}")
    print("=" * 60)
    print()

    buy_signals = []
    watch_list  = []
    no_signals  = []
    errors      = []

    for ticker, name in STOCKS.items():
        market, flag, curr = get_market(ticker)
        print(f"  [{flag} {market}] {name}...", end=" ", flush=True)

        df = download_data(ticker)
        if df.empty:
            errors.append(name)
            print("FAILED")
            continue

        try:
            df = add_indicators(df)
            ok, rules, last = check_signal(df)

            price = float(last["Close"])
            rsi   = float(last["RSI"])
            adx   = float(last["ADX"])
            vol_r = float(last["VOL_RATIO"])
            score = sum(rules.values())
            sl, tp, sl_pct, tp_pct = get_sl_tp(ticker, price)

            if ok:
                buy_signals.append(dict(
                    ticker=ticker, name=name,
                    price=price, sl=sl, tp=tp,
                    sl_pct=sl_pct, tp_pct=tp_pct,
                    rsi=rsi, adx=adx, vol_r=vol_r,
                    rules=rules, market=market, curr=curr
                ))
                print(f"*** BUY SIGNAL! ({score}/5) ***")

            elif score >= 3:
                waiting = [k.split("[")[0].strip()
                           for k, v in rules.items() if not v]
                watch_list.append(dict(
                    name=name, price=price, curr=curr,
                    rsi=rsi, score=score, adx=adx,
                    vol_r=vol_r, waiting=waiting, market=market
                ))
                print(f"Watch ({score}/5)")
            else:
                no_signals.append(name)
                print(f"No signal ({score}/5)")

        except Exception as e:
            errors.append(name)
            print(f"Error: {str(e)[:30]}")

    # ── BUY SIGNALS ──────────────────────────────────────

    print()
    print("=" * 60)
    print("  SCAN RESULTS")
    print("=" * 60)

    if buy_signals:
        print(f"\n  *** {len(buy_signals)} BUY SIGNAL(S) FOUND ***\n")

        for s in buy_signals:
            sym  = s["ticker"].replace(".NS","").replace("-USD","")
            curr = s["curr"]
            ep   = s["price"]
            fmt  = ".0f" if ep > 100 else ".5f"
            adx_s = "STRONG" if s["adx"]   > 25 else "weak"
            vol_s = "HIGH"   if s["vol_r"] >  1 else "low"

            print(f"  {'='*55}")
            print(f"  [BUY] {s['name']} ({sym}) — {s['market']}")
            print(f"  {'='*55}")
            print(f"  Price      : {curr} {ep:{fmt}}")
            print(f"  Take Profit: {curr} {s['tp']:{fmt}}"
                  f"  (+{s['tp_pct']:.0f}%)")
            print(f"  Stop Loss  : {curr} {s['sl']:{fmt}}"
                  f"  (-{s['sl_pct']:.0f}%)")

            if s["market"] == "CRYPTO":
                print(f"  NOTE: Crypto wider SL/TP — higher volatility!")

            print(f"\n  --- EXTRA INFO ---")
            print(f"  ADX  : {s['adx']:.1f}  [{adx_s}]")
            print(f"  Vol  : {s['vol_r']:.2f}x  [{vol_s}]")
            print(f"  RSI  : {s['rsi']:.1f}")

            print(f"\n  --- 5 CORE CONDITIONS (All Pass!) ---")
            for rule, result in s["rules"].items():
                status = "PASS" if result else "FAIL"
                print(f"    [{status}] {rule}")

            print(f"\n  --- ACTION ---")
            print(f"  Platform : {get_platform(s['market'])}")
            print(f"  Search   : {sym}")
            print(f"  SL Alert : {curr} {s['sl']:{fmt}}")
            print(f"  TP Alert : {curr} {s['tp']:{fmt}}")
            print()

    else:
        print(f"\n  No BUY signals today.\n")

    # ── WATCH LIST ───────────────────────────────────────

    for market_name, label in [("NSE","NSE India 🇮🇳"),
                                 ("US","US Stocks 🇺🇸"),
                                 ("CRYPTO","Crypto 🪙")]:
        wl = [w for w in watch_list if w["market"] == market_name]
        if not wl:
            continue
        print(f"  [{label}] Watch List:")
        print(f"  {'─'*55}")
        for w in wl:
            adx_s = "STRONG" if w["adx"]   > 25 else "weak"
            vol_s = "HIGH"   if w["vol_r"] >  1 else "low"
            ep    = w["price"]
            fmt   = ".0f" if ep > 100 else ".5f"
            print(f"  {w['name']:14s} | {w['curr']}{ep:{fmt}}"
                  f" | {w['score']}/5"
                  f" | RSI:{w['rsi']:.1f}"
                  f" | ADX:{w['adx']:.0f}({adx_s})"
                  f" | Vol:{w['vol_r']:.1f}x({vol_s})")
            print(f"    Waiting: {', '.join(w['waiting'])}")
        print()

    # ── SUMMARY ──────────────────────────────────────────

    if no_signals:
        print(f"  No Signal : {' | '.join(no_signals)}")
    if errors:
        print(f"  Failed    : {' | '.join(errors)}")

    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Total Scanned : {len(STOCKS)}"
          f" (NSE:{nse_c} | US:{us_c} | Crypto:{crypto_c})")
    print(f"  BUY Signals   : {len(buy_signals)}")
    print(f"  Watch List    : {len(watch_list)}")
    print(f"  No Signal     : {len(no_signals)}")
    print(f"  Errors        : {len(errors)}")
    print()
    print("  PLATFORMS:")
    print(f"  NSE stocks → Moneybhai"
          f"  (SL:{STOCK_SL}% TP:{STOCK_TP}%)")
    print(f"  US stocks  → TradingView"
          f"  (SL:{STOCK_SL}% TP:{STOCK_TP}%)")
    print(f"  Crypto     → Bybit Demo"
          f"  (SL:{CRYPTO_SL}% TP:{CRYPTO_TP}%)")
    print()
    print("  RULES:")
    print("  [1] All 5 conditions = BUY signal")
    print("  [2] ADX > 25 = Stronger signal (bonus)")
    print("  [3] SL hit   -> Close immediately")
    print("  [4] TP hit   -> Close & celebrate")
    print("  [5] 10 days  -> Close regardless")
    print("  [6] No signal -> No trade!")
    print()
    print(f"  Next scan: Tomorrow 9:20 AM (auto)")
    print("=" * 60)

    sys.exit(0)


if __name__ == "__main__":
    main()