"""
╔══════════════════════════════════════════════════════════════╗
║   ALGO TRADING — Daily Signal Scanner v7                     ║
║   NSE(10) + US(10) + Crypto(5) = 25 stocks                  ║
║   6 Conditions: 5 Technical + Candlestick Pattern            ║
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

JOURNAL_FILE = "trade_journal.csv"
MIN_ROWS     = 210

G    = "\033[92m"
R    = "\033[91m"
Y    = "\033[93m"
B    = "\033[94m"
C    = "\033[96m"
P    = "\033[95m"
W    = "\033[97m"
D    = "\033[90m"
BOLD = "\033[1m"
RST  = "\033[0m"

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def get_market(ticker):
    if ticker.endswith(".NS"):
        return "NSE", "🇮🇳", "₹"
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

    # Technical indicators
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
# CANDLESTICK PATTERN DETECTION (6th Condition!)
# ──────────────────────────────────────────────────────────────

def detect_candle_pattern(df):
    """
    Detect bullish candlestick patterns.
    Returns: (pattern_found, pattern_name, pattern_desc)
    """
    if len(df) < 3:
        return False, "None", ""

    # Last 3 candles
    c0 = df.iloc[-3]  # 3 candles ago
    c1 = df.iloc[-2]  # Previous candle
    c2 = df.iloc[-1]  # Current candle (today)

    o2 = float(c2["Open"])
    h2 = float(c2["High"])
    l2 = float(c2["Low"])
    c_2= float(c2["Close"])

    o1 = float(c1["Open"])
    h1 = float(c1["High"])
    l1 = float(c1["Low"])
    c_1= float(c1["Close"])

    o0 = float(c0["Open"])
    c_0= float(c0["Close"])

    total2  = h2 - l2 + 1e-9
    body2   = abs(c_2 - o2)
    lower_w2= min(o2, c_2) - l2
    upper_w2= h2 - max(o2, c_2)

    total1  = h1 - l1 + 1e-9
    body1   = abs(c_1 - o1)
    lower_w1= min(o1, c_1) - l1

    vol2    = float(c2["VOL_RATIO"]) if "VOL_RATIO" in c2 else 1.0
    vol1    = float(c1["VOL_RATIO"]) if "VOL_RATIO" in c1 else 1.0

    # ── 1. HAMMER 🔨 ─────────────────────────────────────
    # Long lower wick (2x body), small upper wick, green preferred
    is_hammer = (
        lower_w2 >= 2.0 * max(body2, total2 * 0.01) and
        upper_w2 <= 0.15 * total2 and
        body2    >= 0.05 * total2 and
        c_2 >= o2  # Green candle preferred
    )
    if is_hammer:
        return True, "Hammer 🔨", \
            "Long lower wick — buyers rejected lower prices!"

    # ── 2. BULLISH ENGULFING 📈 ───────────────────────────
    # Previous red candle, current green candle engulfs it
    prev_red   = c_1 < o1
    curr_green = c_2 > o2
    engulfs    = (o2 <= c_1) and (c_2 >= o1)
    is_bullish_engulfing = (
        prev_red and curr_green and engulfs and
        body2 > body1
    )
    if is_bullish_engulfing:
        return True, "Bullish Engulfing 📈", \
            "Green candle engulfs previous red — bulls took over!"

    # ── 3. MORNING STAR ⭐ ────────────────────────────────
    # 3 candles: Red → Small(doji) → Green
    prev2_red  = c_0 < o0
    prev1_small= body1 <= 0.3 * (h1 - l1 + 1e-9)
    curr_green2= c_2 > o2
    gap_down   = max(o1, c_1) < min(o0, c_0)
    is_morning_star = (
        prev2_red and prev1_small and
        curr_green2 and
        c_2 > (o0 + c_0) / 2
    )
    if is_morning_star:
        return True, "Morning Star ⭐", \
            "3-candle reversal pattern — strong bullish signal!"

    # ── 4. DOJI AT SUPPORT 十 ────────────────────────────
    # Open ≈ Close, near BB lower band or EMA
    is_doji = body2 <= 0.1 * total2
    near_support = False
    if "BB_LOW" in c2 and "EMA50" in c2:
        bb_low  = float(c2["BB_LOW"])
        ema50   = float(c2["EMA50"])
        near_support = (
            abs(l2 - bb_low) <= 0.02 * c_2 or
            abs(l2 - ema50)  <= 0.02 * c_2
        )
    is_doji_support = is_doji and near_support
    if is_doji_support:
        return True, "Doji at Support 十", \
            "Indecision at support — reversal possible!"

    # ── 5. BULLISH MARUBOZU 💪 ───────────────────────────
    # Strong green candle, almost no wicks
    is_marubozu = (
        c_2 > o2 and
        body2  >= 0.85 * total2 and
        lower_w2 <= 0.05 * total2 and
        upper_w2 <= 0.05 * total2
    )
    if is_marubozu:
        return True, "Bullish Marubozu 💪", \
            "Full green candle — very strong buying pressure!"

    # ── 6. PIERCING LINE 🗡️ ───────────────────────────────
    # Previous red, current green opens below and closes above midpoint
    prev_red2  = c_1 < o1
    opens_low  = o2 < l1
    mid_prev   = (o1 + c_1) / 2
    is_piercing= (
        prev_red2 and
        c_2 > o2 and
        c_2 > mid_prev and
        o2  < c_1
    )
    if is_piercing:
        return True, "Piercing Line 🗡️", \
            "Green candle pierces previous red midpoint — bullish!"

    # ── 7. THREE WHITE SOLDIERS 🪖 ────────────────────────
    # 3 consecutive green candles, each closing higher
    all_green = (c_0 > o0) and (c_1 > o1) and (c_2 > o2)
    ascending = (c_1 > c_0) and (c_2 > c_1)
    is_soldiers = all_green and ascending and \
                  body2 >= 0.5 * total2
    if is_soldiers:
        return True, "Three White Soldiers 🪖", \
            "3 consecutive green candles — strong uptrend!"

    return False, "None", ""

# ──────────────────────────────────────────────────────────────
# SIGNAL CHECK — 6 CONDITIONS
# ──────────────────────────────────────────────────────────────

def check_signal(df):
    if len(df) < 5:
        return False, {}, None, False, "None", ""

    last = df.iloc[-1]

    # 5 Technical Core Rules
    core_rules = {
        "EMA50 > EMA200   [Uptrend]"         : bool(last["EMA50"]  > last["EMA200"]),
        "RSI 50-70        [Momentum]"         : bool(50 < last["RSI"] < 70),
        "MACD > Signal    [Bullish]"          : bool(last["MACD"]   > last["MACD_SIG"]),
        "BB Position < 0.85 [Not Overbought]" : bool(last["BB_POS"] < 0.85),
        "EMA20 > EMA50    [Short Uptrend]"    : bool(last["EMA20"]  > last["EMA50"]),
    }

    # 6th Condition: Candlestick Pattern
    pattern_found, pattern_name, pattern_desc = detect_candle_pattern(df)

    all_rules = dict(core_rules)
    all_rules[f"Bullish Candle Pattern [{pattern_name}]"] = pattern_found

    all_pass = all(all_rules.values())

    return all_pass, all_rules, last, pattern_found, pattern_name, pattern_desc

# ──────────────────────────────────────────────────────────────
# JOURNAL
# ──────────────────────────────────────────────────────────────

def init_journal():
    if not os.path.exists(JOURNAL_FILE):
        with open(JOURNAL_FILE, "w", newline="") as f:
            csv.writer(f).writerow([
                "Date","Stock","Ticker","Market",
                "Entry_Price","Stop_Loss","Take_Profit",
                "SL_PCT","TP_PCT","ADX","Vol_Ratio","Pattern",
                "Exit_Date","Exit_Price","Exit_Type",
                "Return_%","Result","Notes"
            ])

def log_trade(ticker, name, price, sl, tp,
              sl_pct, tp_pct, adx, vol, market, pattern):
    with open(JOURNAL_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.today().strftime("%Y-%m-%d"),
            name, ticker, market,
            f"{price:.4f}", f"{sl:.4f}", f"{tp:.4f}",
            f"{sl_pct}", f"{tp_pct}",
            f"{adx:.1f}", f"{vol:.2f}", pattern,
            "","","","","OPEN",""
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
        df = pd.read_csv(JOURNAL_FILE)
        op = df[df["Result"] == "OPEN"]
        if len(op) == 0:
            return
        print(f"\n  {BOLD}{Y}  OPEN TRADES ({len(op)}){RST}")
        print(f"  {'─'*75}")
        print(f"  {D}  {'Date':<12} {'Stock':<14} {'Mkt':>6} "
              f"{'Entry':>10} {'SL':>10} {'TP':>10} {'Days':>5}{RST}")
        print(f"  {'─'*75}")
        today = datetime.today().date()
        for _, row in op.iterrows():
            try:
                days = (today - pd.to_datetime(row["Date"]).date()).days
                mkt  = str(row.get("Market","NSE"))
                curr = "₹" if mkt == "NSE" else "$"
                warn = f"{R}⚠️ {RST}" if days > 8 else "   "
                ep   = float(row["Entry_Price"])
                fmt  = ".0f" if ep > 10 else ".4f"
                print(f"  {W}  {str(row['Date']):<12} "
                      f"{str(row['Stock']):<14} "
                      f"{mkt:>6} "
                      f"{curr}{ep:>9{fmt}} "
                      f"{R}{curr}{float(row['Stop_Loss']):>9{fmt}}{RST} "
                      f"{G}{curr}{float(row['Take_Profit']):>9{fmt}}{RST} "
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
        done   = df[df["Result"].isin(["WIN","LOSS"])]
        open_t = df[df["Result"] == "OPEN"]

        print(f"\n  {BOLD}{C}  MODEL SUCCESS RATE TRACKER{RST}")
        print(f"  {'═'*52}")

        if len(done) == 0:
            print(f"  {D}  No completed trades yet.")
            print(f"  {D}  Need 20+ trades for verdict!{RST}\n")
            return

        wins  = (done["Result"] == "WIN").sum()
        loss  = (done["Result"] == "LOSS").sum()
        wr    = wins / len(done) * 100
        rets  = pd.to_numeric(done["Return_%"], errors="coerce")
        avg   = rets.mean()
        total = rets.sum()
        tp_c  = (done["Exit_Type"] == "TP").sum()
        sl_c  = (done["Exit_Type"] == "SL").sum()

        wc = G if wr >= 55 else Y if wr >= 50 else R
        ac = G if avg > 0 else R
        tc = G if total > 0 else R

        print(f"  {W}  Completed : {BOLD}{len(done)}{RST} "
              f"({D}Open: {len(open_t)}{RST})")
        print(f"  {G}  Wins      : {BOLD}{wins}{RST}  "
              f"{R}Losses: {BOLD}{loss}{RST}")
        print(f"  {W}  Win Rate  : {wc}{BOLD}{wr:.1f}%{RST}")
        print(f"  {W}  Avg Return: {ac}{BOLD}{avg:+.2f}%{RST}")
        print(f"  {W}  Total     : {tc}{BOLD}{total:+.2f}%{RST}")
        print(f"  {'─'*52}")
        print(f"  {W}  TP hits: {G}{tp_c}{RST}  SL hits: {R}{sl_c}{RST}")

        # Per market breakdown
        if "Market" in done.columns:
            print(f"  {'─'*52}")
            for mkt in ["NSE","US","CRYPTO"]:
                m = done[done["Market"] == mkt]
                if len(m) == 0:
                    continue
                m_wr = (m["Result"] == "WIN").mean() * 100
                m_av = pd.to_numeric(
                    m["Return_%"], errors="coerce").mean()
                mc   = G if m_wr >= 55 else R
                icon = ("🇮🇳" if mkt == "NSE" else
                        "🪙"  if mkt == "CRYPTO" else "🇺🇸")
                print(f"  {D}  {icon} {mkt:<7}: "
                      f"{mc}{m_wr:.1f}% WR{RST}  "
                      f"Avg:{m_av:+.2f}%  "
                      f"({len(m)} trades)")

        # Pattern success rate
        if "Pattern" in done.columns:
            with_pat = done[done["Pattern"] != "None"]
            if len(with_pat) > 0:
                pat_wr = (with_pat["Result"]=="WIN").mean()*100
                print(f"  {'─'*52}")
                pc = G if pat_wr >= 55 else R
                print(f"  {D}  🕯️ Pattern trades: "
                      f"{pc}{pat_wr:.1f}% WR{RST} "
                      f"({len(with_pat)} trades)")

        print(f"  {'─'*52}")
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

    nse_count    = sum(1 for k in STOCKS if k.endswith(".NS"))
    us_count     = sum(1 for k in STOCKS
                       if not k.endswith(".NS") and "-USD" not in k)
    crypto_count = sum(1 for k in STOCKS if "-USD" in k)

    print(f"\n{BOLD}{B}"
          f"╔══════════════════════════════════════════════════╗\n"
          f"║   ALGO TRADING — Daily Signal Scanner v7         ║\n"
          f"║   {today}  {now.strftime('%I:%M %p')}                     ║\n"
          f"║   NSE({nse_count}) + US({us_count}) + Crypto({crypto_count})"
          f" = {len(STOCKS)} stocks     ║\n"
          f"║   6 Conditions: Technical + Candle Pattern       ║\n"
          f"╚══════════════════════════════════════════════════╝"
          f"{RST}\n")

    init_journal()
    show_open_trades()

    print(f"  {BOLD}{W}  Scanning {len(STOCKS)} stocks...{RST}\n")

    nse_buy    = []
    us_buy     = []
    crypto_buy = []
    watch      = []
    no_sig     = []
    errors     = []

    for ticker, name in STOCKS.items():
        market, flag, curr = get_market(ticker)

        print(f"  {D}  [{flag} {market}] {name:14s}...{RST}",
              end=" ", flush=True)

        df = download_data(ticker)
        if df.empty:
            errors.append(name)
            print(f"{R}FAILED{RST}")
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
                sig = dict(
                    ticker=ticker, name=name,
                    price=price, sl=sl, tp=tp,
                    sl_pct=sl_pct, tp_pct=tp_pct,
                    rsi=rsi, adx=adx, vol_r=vol_r,
                    rules=rules,
                    pattern=pat_name, pat_desc=pat_desc,
                    market=market, curr=curr
                )
                if market == "NSE":
                    nse_buy.append(sig)
                elif market == "CRYPTO":
                    crypto_buy.append(sig)
                else:
                    us_buy.append(sig)
                print(f"{G}{BOLD}BUY SIGNAL! ({score}/6) "
                      f"[{pat_name}]{RST}")

            elif score >= 4:
                waiting = [k.split("[")[0].strip()
                           for k, v in rules.items() if not v]
                watch.append(dict(
                    name=name, price=price, curr=curr,
                    rsi=rsi, score=score, adx=adx,
                    vol_r=vol_r, waiting=waiting,
                    market=market,
                    pattern=pat_name if pat_found else "None"
                ))
                pat_txt = f" [{pat_name}]" if pat_found else ""
                print(f"{Y}Watch ({score}/6){pat_txt}{RST}")
            else:
                no_sig.append(name)
                print(f"{D}No signal ({score}/6){RST}")

        except Exception:
            errors.append(name)
            print(f"{R}Error{RST}")

    all_buy = nse_buy + us_buy + crypto_buy
    print()

    # ── BUY SIGNALS ──────────────────────────────────────

    if all_buy:
        print(f"\n  {BOLD}{G}"
              f"╔══════════════════════════════════════════════╗\n"
              f"  ║   ▲  BUY SIGNALS — {len(all_buy)} found"
              f"  (6/6 conditions!)   ║\n"
              f"  ╚══════════════════════════════════════════════╝"
              f"{RST}\n")

        for s in all_buy:
            sym     = s["ticker"].replace(".NS","").replace("-USD","")
            curr    = s["curr"]
            already = is_already_open(s["ticker"])
            adx_i   = "✅" if s["adx"]   > 25 else "⚠️"
            vol_i   = "✅" if s["vol_r"] >  1 else "⚠️"
            _, flag, _ = get_market(s["ticker"])
            platform   = get_platform(s["market"])

            mkt_color = B if s["market"]=="NSE" else \
                        P if s["market"]=="CRYPTO" else C

            print(f"  {BOLD}{G}  ▲  {s['name']} ({sym}){RST}"
                  f"  {mkt_color}[{flag} {s['market']}]{RST}")
            print(f"  {'─'*58}")

            ep  = s["price"]
            fmt = ".0f" if ep > 10 else ".5f"
            print(f"  {W}  Price  : {BOLD}{curr}{ep:{fmt}}{RST}")
            print(f"  {G}  Target : {BOLD}{curr}{s['tp']:{fmt}}{RST}"
                  f"  {D}(+{s['tp_pct']:.0f}%){RST}")
            print(f"  {R}  Stop L : {BOLD}{curr}{s['sl']:{fmt}}{RST}"
                  f"  {D}(-{s['sl_pct']:.0f}%){RST}")

            if s["market"] == "CRYPTO":
                print(f"\n  {P}  ⚡ CRYPTO: Wider SL/TP"
                      f" — Higher volatility!{RST}")

            # Candlestick Pattern highlight
            print(f"\n  {BOLD}{Y}  🕯️ CANDLE PATTERN: "
                  f"{s['pattern']}{RST}")
            print(f"  {D}     {s['pat_desc']}{RST}")

            print(f"\n  {D}  ADX:{s['adx']:.1f}{adx_i}"
                  f"  Vol:{s['vol_r']:.2f}x{vol_i}"
                  f"  RSI:{s['rsi']:.1f}{RST}")

            print(f"\n  {BOLD}{W}  ★ 6/6 Conditions Pass:{RST}")
            for rule, result in s["rules"].items():
                icon = f"{G}✔{RST}" if result else f"{R}✘{RST}"
                # Highlight pattern condition
                if "Pattern" in rule:
                    print(f"     {icon}  {Y}{rule}{RST}")
                else:
                    print(f"     {icon}  {W}{rule}{RST}")

            print(f"\n  {BOLD}{Y}  ACTION:{RST}")
            print(f"  {W}  Platform: {platform}")
            print(f"     Search : {sym}")
            buy_qty = ("small amount" if s["market"]=="CRYPTO"
                       else "1-5 shares" if s["market"]=="NSE"
                       else "5-10 shares")
            print(f"     BUY    : {buy_qty}")
            print(f"     SL Alert: {curr}{s['sl']:{fmt}}")
            print(f"     TP Alert: {curr}{s['tp']:{fmt}}{RST}")

            if already:
                print(f"\n  {Y}  ⚠️  Already tracking {s['name']}!"
                      f" Skip save.{RST}")
            print(f"  {'─'*58}\n")

        to_save = [s for s in all_buy
                   if not is_already_open(s["ticker"])]
        if to_save:
            save = input(
                f"  {Y}  Save {len(to_save)} new trade(s)? (y/n): {RST}"
            ).strip().lower()
            if save == "y":
                for s in to_save:
                    log_trade(
                        s["ticker"], s["name"],
                        s["price"], s["sl"], s["tp"],
                        s["sl_pct"], s["tp_pct"],
                        s["adx"], s["vol_r"],
                        s["market"], s["pattern"]
                    )
                print(f"  {G}  ✔  Saved!{RST}\n")
        else:
            print(f"  {D}  All signals already in journal.{RST}\n")

    else:
        print(f"  {Y}  ◆  No BUY signals today.{RST}")
        print(f"  {D}  (Need all 6 conditions including"
              f" candle pattern){RST}\n")

    # ── WATCH LIST ───────────────────────────────────────

    if watch:
        nse_w    = [w for w in watch if w["market"] == "NSE"]
        us_w     = [w for w in watch if w["market"] == "US"]
        crypto_w = [w for w in watch if w["market"] == "CRYPTO"]

        print(f"  {BOLD}{Y}  WATCH LIST — {len(watch)} stocks{RST}")
        print(f"  {'─'*60}")

        def print_watch(wlist, label, color):
            if not wlist:
                return
            print(f"\n  {color}  {label}{RST}")
            for w in wlist:
                ai  = "✅" if w["adx"]   > 25 else "⚠️"
                vi  = "✅" if w["vol_r"] >  1 else "⚠️"
                ep  = w["price"]
                fmt = ".0f" if ep > 10 else ".5f"
                pat = (f" | 🕯️{w['pattern']}"
                       if w.get("pattern","None") != "None" else "")
                print(f"  {Y}  ◈ {w['name']:14s}{RST}"
                      f"  {w['curr']}{ep:{fmt}}"
                      f"  {w['score']}/6"
                      f"  RSI:{w['rsi']:.1f}"
                      f"  ADX:{w['adx']:.0f}{ai}"
                      f"  Vol:{w['vol_r']:.1f}x{vi}"
                      f"{G}{pat}{RST}")
                print(f"     {R}  → {', '.join(w['waiting'])}{RST}")

        print_watch(nse_w,    "NSE India 🇮🇳", B)
        print_watch(us_w,     "US Stocks 🇺🇸",  C)
        print_watch(crypto_w, "Crypto 🪙",       P)
        print()

    # ── NO SIGNAL ────────────────────────────────────────

    if no_sig:
        print(f"  {D}  ○  No Signal : {' | '.join(no_sig)}{RST}\n")
    if errors:
        print(f"  {R}  ⚠  Failed   : {' | '.join(errors)}{RST}\n")

    # ── SUCCESS RATE ─────────────────────────────────────

    show_success_rate()

    # ── FOOTER ───────────────────────────────────────────

    nse_c    = nse_count
    us_c     = us_count
    crypto_c = crypto_count

    print(f"""
  {BOLD}{W}  STRATEGY v7 — 6 Conditions{RST}
  {'─'*55}
  {W}  Technical (5):{RST}
  {G}  ✔  EMA50 > EMA200   [Uptrend]{RST}
  {G}  ✔  RSI 50-70         [Momentum]{RST}
  {G}  ✔  MACD > Signal     [Bullish]{RST}
  {G}  ✔  BB Pos < 0.85     [Not Overbought]{RST}
  {G}  ✔  EMA20 > EMA50     [Short Uptrend]{RST}
  {Y}  ✔  Candle Pattern    [Hammer/Engulfing/etc]{RST}
  {'─'*55}
  {B}  🇮🇳 NSE    → Moneybhai   SL:{STOCK_SL}% TP:{STOCK_TP}%{RST}
  {C}  🇺🇸 US     → TradingView SL:{STOCK_SL}% TP:{STOCK_TP}%{RST}
  {P}  🪙 Crypto  → Bybit Demo  SL:{CRYPTO_SL}% TP:{CRYPTO_TP}%{RST}
  {'─'*55}
  {G}  ✔  SL hit  → Close immediately
  ✔  TP hit  → Close & celebrate
  ✔  10 days → Close regardless{RST}
  {R}  ✘  No signal → No trade!{RST}
  {'─'*55}
  {D}  Stocks : NSE({nse_c}) + US({us_c}) + Crypto({crypto_c})
  Next   : Tomorrow 9:20 AM
  Command: python daily_scanner.py{RST}
""")


if __name__ == "__main__":
    main()
