"""
Daily Signal Scanner v7 — GitHub Actions Version
NSE(10) + US(10) + Crypto(5) = 25 stocks
6 Conditions: 5 Technical + Candlestick Pattern
No colors, no input() — pure text for GitHub logs
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
        return "NSE", "Rs"
    elif "-USD" in ticker:
        return "CRYPTO", "$"
    else:
        return "US", "$"

def get_sl_tp(ticker, price):
    if "-USD" in ticker:
        return (price*(1-CRYPTO_SL/100),
                price*(1+CRYPTO_TP/100),
                CRYPTO_SL, CRYPTO_TP)
    else:
        return (price*(1-STOCK_SL/100),
                price*(1+STOCK_TP/100),
                STOCK_SL, STOCK_TP)

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
    d["BB_HIGH"]= bb.bollinger_hband()
    d["BB_LOW"] = bb.bollinger_lband()
    d["BB_POS"] = (close - bb.bollinger_lband()) / \
                  (bb.bollinger_hband() - bb.bollinger_lband() + 1e-9)

    adx        = ta.trend.ADXIndicator(high, low, close, 14)
    d["ADX"]   = adx.adx()

    d["VOL_MA20"]  = vol.rolling(20).mean()
    d["VOL_RATIO"] = vol / (d["VOL_MA20"] + 1)
    d["EMA_GAP"]   = (d["EMA50"] - d["EMA200"]) / d["EMA200"] * 100

    return d.dropna()

# ──────────────────────────────────────────────────────────────
# CANDLESTICK PATTERN DETECTION
# ──────────────────────────────────────────────────────────────

def detect_candle_pattern(df):
    if len(df) < 3:
        return False, "None", ""

    c0  = df.iloc[-3]
    c1  = df.iloc[-2]
    c2  = df.iloc[-1]

    o2  = float(c2["Open"]);  h2 = float(c2["High"])
    l2  = float(c2["Low"]);   c_2= float(c2["Close"])
    o1  = float(c1["Open"]);  h1 = float(c1["High"])
    l1  = float(c1["Low"]);   c_1= float(c1["Close"])
    o0  = float(c0["Open"]);  c_0= float(c0["Close"])

    total2   = h2 - l2 + 1e-9
    body2    = abs(c_2 - o2)
    lower_w2 = min(o2, c_2) - l2
    upper_w2 = h2 - max(o2, c_2)

    total1   = h1 - l1 + 1e-9
    body1    = abs(c_1 - o1)

    # 1. Hammer
    if (lower_w2 >= 2.0 * max(body2, total2*0.01) and
            upper_w2 <= 0.15 * total2 and
            body2 >= 0.05 * total2 and c_2 >= o2):
        return True, "Hammer", \
            "Long lower wick — buyers rejected lower prices!"

    # 2. Bullish Engulfing
    if (c_1 < o1 and c_2 > o2 and
            o2 <= c_1 and c_2 >= o1 and body2 > body1):
        return True, "Bullish Engulfing", \
            "Green candle engulfs previous red — bulls took over!"

    # 3. Morning Star
    if (c_0 < o0 and
            body1 <= 0.3*(h1-l1+1e-9) and
            c_2 > o2 and c_2 > (o0+c_0)/2):
        return True, "Morning Star", \
            "3-candle reversal — strong bullish signal!"

    # 4. Doji at Support
    if body2 <= 0.1 * total2:
        bb_low = float(c2["BB_LOW"]) if "BB_LOW" in c2 else 0
        ema50  = float(c2["EMA50"])  if "EMA50"  in c2 else 0
        if (abs(l2-bb_low) <= 0.02*c_2 or
                abs(l2-ema50) <= 0.02*c_2):
            return True, "Doji at Support", \
                "Indecision at support — reversal possible!"

    # 5. Bullish Marubozu
    if (c_2 > o2 and body2 >= 0.85*total2 and
            lower_w2 <= 0.05*total2 and upper_w2 <= 0.05*total2):
        return True, "Bullish Marubozu", \
            "Full green candle — strong buying pressure!"

    # 6. Piercing Line
    if (c_1 < o1 and c_2 > o2 and
            o2 < c_1 and c_2 > (o1+c_1)/2):
        return True, "Piercing Line", \
            "Green pierces red midpoint — bullish!"

    # 7. Three White Soldiers
    if (c_0>o0 and c_1>o1 and c_2>o2 and
            c_1>c_0 and c_2>c_1 and
            body2 >= 0.5*total2):
        return True, "Three White Soldiers", \
            "3 consecutive green candles — strong uptrend!"

    return False, "None", ""

# ──────────────────────────────────────────────────────────────
# SIGNAL CHECK — 6 CONDITIONS
# ──────────────────────────────────────────────────────────────

def check_signal(df):
    if len(df) < 5:
        return False, {}, None, False, "None", ""

    last = df.iloc[-1]

    core_rules = {
        "EMA50 > EMA200   [Uptrend]"         : bool(last["EMA50"]  > last["EMA200"]),
        "RSI 50-70        [Momentum]"         : bool(50 < last["RSI"] < 70),
        "MACD > Signal    [Bullish]"          : bool(last["MACD"]   > last["MACD_SIG"]),
        "BB Position < 0.85 [Not Overbought]" : bool(last["BB_POS"] < 0.85),
        "EMA20 > EMA50    [Short Uptrend]"    : bool(last["EMA20"]  > last["EMA50"]),
    }

    pat_found, pat_name, pat_desc = detect_candle_pattern(df)

    all_rules = dict(core_rules)
    all_rules[f"Bullish Candle [{pat_name}]"] = pat_found

    return all(all_rules.values()), all_rules, last, \
           pat_found, pat_name, pat_desc

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

    print("=" * 62)
    print("  ALGO TRADING — Daily Signal Scanner v7")
    print(f"  Date    : {today}")
    print(f"  Time    : {now.strftime('%I:%M %p')} IST")
    print(f"  Stocks  : NSE({nse_c}) + US({us_c})"
          f" + Crypto({crypto_c}) = {len(STOCKS)}")
    print("  Signal  : 6 Conditions (Technical + Candle)")
    print("=" * 62)
    print()

    buy_signals = []
    watch_list  = []
    no_signals  = []
    errors      = []

    for ticker, name in STOCKS.items():
        market, curr = get_market(ticker)
        flag = ("NSE" if market=="NSE" else
                "CRYPTO" if market=="CRYPTO" else "US")
        print(f"  [{flag}] {name}...", end=" ", flush=True)

        df = download_data(ticker)
        if df.empty:
            errors.append(name)
            print("FAILED")
            continue

        try:
            df = add_indicators(df)
            ok, rules, last, pat_found, pat_name, pat_desc = \
                check_signal(df)

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
                    rules=rules, market=market, curr=curr,
                    pattern=pat_name, pat_desc=pat_desc
                ))
                print(f"*** BUY! ({score}/6) [{pat_name}] ***")
            elif score >= 4:
                waiting = [k.split("[")[0].strip()
                           for k, v in rules.items() if not v]
                watch_list.append(dict(
                    name=name, price=price, curr=curr,
                    rsi=rsi, score=score, adx=adx,
                    vol_r=vol_r, waiting=waiting,
                    market=market,
                    pattern=pat_name if pat_found else "None"
                ))
                pat_txt = f" [{pat_name}]" if pat_found else ""
                print(f"Watch ({score}/6){pat_txt}")
            else:
                no_signals.append(name)
                print(f"No signal ({score}/6)")

        except Exception as e:
            errors.append(name)
            print(f"Error: {str(e)[:30]}")

    # ── RESULTS ──────────────────────────────────────────

    print()
    print("=" * 62)
    print("  SCAN RESULTS")
    print("=" * 62)

    if buy_signals:
        print(f"\n  *** {len(buy_signals)} BUY SIGNAL(S)"
              f" — 6/6 CONDITIONS PASS ***\n")

        for s in buy_signals:
            sym  = s["ticker"].replace(".NS","").replace("-USD","")
            curr = s["curr"]
            ep   = s["price"]
            fmt  = ".2f" if ep > 10 else ".6f"
            adx_s = "STRONG" if s["adx"]   > 25 else "weak"
            vol_s = "HIGH"   if s["vol_r"] >  1 else "low"

            print(f"  {'='*58}")
            print(f"  [BUY] {s['name']} ({sym}) — {s['market']}")
            print(f"  {'='*58}")
            print(f"  Price      : {curr} {ep:{fmt}}")
            print(f"  Take Profit: {curr} {s['tp']:{fmt}}"
                  f"  (+{s['tp_pct']:.0f}%)")
            print(f"  Stop Loss  : {curr} {s['sl']:{fmt}}"
                  f"  (-{s['sl_pct']:.0f}%)")

            print(f"\n  --- CANDLE PATTERN (6th Condition) ---")
            print(f"  Pattern : {s['pattern']}")
            print(f"  Meaning : {s['pat_desc']}")

            print(f"\n  --- EXTRA INFO ---")
            print(f"  ADX  : {s['adx']:.1f}  [{adx_s}]")
            print(f"  Vol  : {s['vol_r']:.2f}x  [{vol_s}]")
            print(f"  RSI  : {s['rsi']:.1f}")

            print(f"\n  --- 6 CONDITIONS (All Pass!) ---")
            for rule, result in s["rules"].items():
                status = "PASS" if result else "FAIL"
                print(f"    [{status}] {rule}")

            print(f"\n  --- ACTION ---")
            print(f"  Platform : {get_platform(s['market'])}")
            print(f"  Search   : {sym}")
            print(f"  SL Alert : {curr} {s['sl']:{fmt}}")
            print(f"  TP Alert : {curr} {s['tp']:{fmt}}")
            if s["market"] == "CRYPTO":
                print(f"  NOTE: Crypto wider SL/TP — high volatility!")
            print()

    else:
        print(f"\n  No BUY signals today.")
        print(f"  (Need all 6 conditions"
              f" including candle pattern)\n")

    # ── WATCH LIST ───────────────────────────────────────

    for mkt_name, label in [("NSE",    "NSE India"),
                              ("US",     "US Stocks"),
                              ("CRYPTO", "Crypto")]:
        wl = [w for w in watch_list if w["market"] == mkt_name]
        if not wl:
            continue
        print(f"  [{label}] Watch List:")
        print(f"  {'─'*58}")
        for w in wl:
            adx_s = "STRONG" if w["adx"]   > 25 else "weak"
            vol_s = "HIGH"   if w["vol_r"] >  1 else "low"
            ep    = w["price"]
            fmt   = ".2f" if ep > 10 else ".6f"
            pat   = (f" | Pattern:{w['pattern']}"
                     if w.get("pattern","None") != "None" else "")
            print(f"  {w['name']:14s}"
                  f" | {w['curr']}{ep:{fmt}}"
                  f" | {w['score']}/6"
                  f" | RSI:{w['rsi']:.1f}"
                  f" | ADX:{w['adx']:.0f}({adx_s})"
                  f" | Vol:{w['vol_r']:.1f}x({vol_s})"
                  f"{pat}")
            print(f"    Waiting: {', '.join(w['waiting'])}")
        print()

    # ── SUMMARY ──────────────────────────────────────────

    if no_signals:
        print(f"  No Signal : {' | '.join(no_signals)}")
    if errors:
        print(f"  Failed    : {' | '.join(errors)}")

    print()
    print("=" * 62)
    print("  SUMMARY")
    print("=" * 62)
    print(f"  Total    : {len(STOCKS)}"
          f" (NSE:{nse_c} | US:{us_c} | Crypto:{crypto_c})")
    print(f"  BUY      : {len(buy_signals)}")
    print(f"  Watch    : {len(watch_list)}")
    print(f"  No Signal: {len(no_signals)}")
    print(f"  Errors   : {len(errors)}")
    print()
    print("  6 CONDITIONS FOR BUY SIGNAL:")
    print("  [1] EMA50 > EMA200    [Uptrend]")
    print("  [2] RSI 50-70         [Momentum]")
    print("  [3] MACD > Signal     [Bullish]")
    print("  [4] BB Pos < 0.85     [Not Overbought]")
    print("  [5] EMA20 > EMA50     [Short Uptrend]")
    print("  [6] Candle Pattern    [Hammer/Engulfing/etc]")
    print()
    print("  PLATFORMS:")
    print(f"  NSE    -> Moneybhai  (SL:{STOCK_SL}% TP:{STOCK_TP}%)")
    print(f"  US     -> TradingView(SL:{STOCK_SL}% TP:{STOCK_TP}%)")
    print(f"  Crypto -> Bybit Demo (SL:{CRYPTO_SL}% TP:{CRYPTO_TP}%)")
    print()
    print("  RULES:")
    print("  [1] All 6 conditions = BUY signal")
    print("  [2] SL hit   -> Close immediately")
    print("  [3] TP hit   -> Close & celebrate")
    print("  [4] 10 days  -> Close regardless")
    print("  [5] No signal -> No trade!")
    print()
    print(f"  Next scan: Tomorrow 9:20 AM (auto)")
    print("=" * 62)

    sys.exit(0)


if __name__ == "__main__":
    main()
