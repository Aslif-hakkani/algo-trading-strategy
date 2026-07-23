"""
AlgoView Risk Management Layer
--------------------------------
Plug this into your existing scanner (GitHub Actions job) right after
a signal is generated and before you log/place the paper trade.

Covers:
1. Fixed-fractional position sizing
2. Crypto correlation exposure check
3. Portfolio drawdown circuit breaker

Assumes your existing SL% by asset class:
    stocks -> SL 2%, TP 4%
    crypto -> SL 5%, TP 15%
"""

import json
import os
from datetime import datetime, timezone

# ---------------------------------------------------------------
# CONFIG - tune these
# ---------------------------------------------------------------
RISK_PER_TRADE_PCT = 0.015          # risk 1.5% of portfolio per trade
MAX_CRYPTO_SAME_DIRECTION = 2        # max concurrent same-direction crypto positions
DRAWDOWN_PAUSE_PCT = 0.10            # pause new trades if portfolio down 10% from peak
STATE_FILE   = "risk_state.json"     # persisted between GitHub Actions runs (drawdown peak)
JOURNAL_FILE = "trade_journal.csv"   # your existing scanner's journal - source of truth for open positions

SL_PCT = {
    "stock": 0.02,
    "crypto": 0.05,
}

# matches your scanner's crypto tickers (yfinance format)
CRYPTO_ASSETS = {"BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD"}


# ---------------------------------------------------------------
# 1. POSITION SIZING (fixed-fractional)
# ---------------------------------------------------------------
def calculate_position_size(portfolio_value: float, entry_price: float,
                             asset_class: str) -> dict:
    """
    Returns position size in units + capital allocated, based on
    risking a fixed % of portfolio on the stop-loss distance.
    """
    sl_pct = SL_PCT.get(asset_class)
    if sl_pct is None:
        raise ValueError(f"Unknown asset_class: {asset_class}")

    stop_loss_price = entry_price * (1 - sl_pct)
    risk_amount = portfolio_value * RISK_PER_TRADE_PCT
    price_risk_per_unit = entry_price - stop_loss_price

    units = risk_amount / price_risk_per_unit
    capital_allocated = units * entry_price

    return {
        "units": round(units, 6),
        "capital_allocated": round(capital_allocated, 2),
        "risk_amount": round(risk_amount, 2),
        "stop_loss_price": round(stop_loss_price, 4),
    }


# ---------------------------------------------------------------
# 2. CRYPTO CORRELATION / EXPOSURE CHECK
# ---------------------------------------------------------------
def load_open_crypto_count(exclude_ticker: str = None) -> int:
    """
    Reads your existing trade_journal.csv and counts how many crypto
    trades currently have Result == 'OPEN'. All your signals are
    long-only (BUY), so we just count open crypto positions overall.
    """
    if not os.path.exists(JOURNAL_FILE):
        return 0
    try:
        import pandas as pd
        df = pd.read_csv(JOURNAL_FILE)
        open_df = df[df["Result"] == "OPEN"]
        if exclude_ticker:
            open_df = open_df[open_df["Ticker"] != exclude_ticker]
        crypto_open = open_df[open_df["Ticker"].isin(CRYPTO_ASSETS)]
        return len(crypto_open)
    except Exception:
        return 0


def check_crypto_exposure(new_ticker: str) -> dict:
    """
    Blocks/reduces a new crypto trade if too many crypto positions
    are already open (crypto pairs tend to move together, so stacking
    BTC + ETH + SOL longs at once is concentrated risk, not diversification).
    """
    if new_ticker not in CRYPTO_ASSETS:
        return {"allowed": True, "size_multiplier": 1.0, "reason": "not crypto, no correlation check needed"}

    open_crypto = load_open_crypto_count(exclude_ticker=new_ticker)

    if open_crypto >= MAX_CRYPTO_SAME_DIRECTION:
        return {
            "allowed": False,
            "size_multiplier": 0.0,
            "reason": f"{open_crypto} crypto positions already open, limit is {MAX_CRYPTO_SAME_DIRECTION}",
        }
    elif open_crypto == MAX_CRYPTO_SAME_DIRECTION - 1:
        return {
            "allowed": True,
            "size_multiplier": 0.5,
            "reason": f"{open_crypto} crypto position(s) open already, sizing down 50%",
        }

    return {"allowed": True, "size_multiplier": 1.0, "reason": "within exposure limits"}


# ---------------------------------------------------------------
# 3. DRAWDOWN CIRCUIT BREAKER
# ---------------------------------------------------------------
class DrawdownCircuitBreaker:
    """
    Tracks portfolio peak value and pauses new trade entries
    if drawdown exceeds DRAWDOWN_PAUSE_PCT. Existing positions'
    SL/TP still execute as normal; this only blocks NEW entries.
    """

    def __init__(self, state_file: str = STATE_FILE):
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                return json.load(f)
        return {"peak_value": None, "paused": False, "open_positions": []}

    def _save_state(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def update(self, current_portfolio_value: float) -> dict:
        peak = self.state.get("peak_value")
        if peak is None or current_portfolio_value > peak:
            peak = current_portfolio_value

        drawdown = (peak - current_portfolio_value) / peak if peak else 0
        paused = drawdown >= DRAWDOWN_PAUSE_PCT

        self.state["peak_value"] = peak
        self.state["paused"] = paused
        self.state["last_checked"] = datetime.now(timezone.utc).isoformat()
        self._save_state()

        return {
            "peak_value": round(peak, 2),
            "current_value": round(current_portfolio_value, 2),
            "drawdown_pct": round(drawdown * 100, 2),
            "trading_paused": paused,
        }


# ---------------------------------------------------------------
# EXAMPLE: wiring it into your daily scanner
# ---------------------------------------------------------------
if __name__ == "__main__":
    portfolio_value = 10000.0

    # Step 1: check circuit breaker first
    breaker = DrawdownCircuitBreaker()
    dd_status = breaker.update(portfolio_value)
    print("Drawdown check:", dd_status)

    if dd_status["trading_paused"]:
        print("Trading paused due to drawdown. Skipping new entries.")
    else:
        # Step 2: signal generated by your existing 5-condition logic
        ticker = "BTC-USD"
        entry_price = 62500.0
        asset_class = "crypto"

        exposure = check_crypto_exposure(ticker)
        print("Exposure check:", exposure)

        if exposure["allowed"]:
            sizing = calculate_position_size(portfolio_value, entry_price, asset_class)
            multiplier = exposure.get("size_multiplier", 1.0)
            sizing["units"] = round(sizing["units"] * multiplier, 6)
            sizing["capital_allocated"] = round(sizing["capital_allocated"] * multiplier, 2)
            print("Final position sizing:", sizing)
        else:
            print(f"Trade skipped: {exposure['reason']}")
