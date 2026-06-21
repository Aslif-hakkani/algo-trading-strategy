"""
Trade Journal Update Tool
Usage: python update_trade.py
"""

import pandas as pd
import csv
import os
from datetime import datetime

JOURNAL_FILE = "trade_journal.csv"
GREEN  = "\033[92m"; RED = "\033[91m"
YELLOW = "\033[93m"; BOLD = "\033[1m"; RESET = "\033[0m"

def update_trade():
    if not os.path.exists(JOURNAL_FILE):
        print(f"{RED}Journal file not found!{RESET}")
        return

    df = pd.read_csv(JOURNAL_FILE)
    open_trades = df[df["Result"] == "OPEN"]

    if len(open_trades) == 0:
        print(f"{YELLOW}No open trades to update.{RESET}")
        return

    print(f"\n{BOLD}Open Trades:{RESET}")
    for i, (idx, row) in enumerate(open_trades.iterrows()):
        print(f"  {i+1}. {row['Stock']} | Entry: ₹{row['Entry_Price']} | "
              f"SL: ₹{row['Stop_Loss']} | TP: ₹{row['Take_Profit']}")

    try:
        choice = int(input(f"\n{YELLOW}Which trade to close? (number): {RESET}")) - 1
        idx    = open_trades.index[choice]
        row    = open_trades.iloc[choice]
    except Exception:
        print(f"{RED}Invalid choice.{RESET}"); return

    print(f"\n  Closing: {row['Stock']}")
    exit_price = float(input(f"  Exit price (₹): "))
    exit_type  = input(f"  Exit type (TP/SL/manual): ").strip().upper()
    notes      = input(f"  Notes (optional): ").strip()

    entry_price = float(row["Entry_Price"])
    ret_pct     = (exit_price - entry_price) / entry_price * 100
    result      = "WIN" if ret_pct > 0 else "LOSS"
    today       = datetime.today().strftime("%Y-%m-%d")

    df.loc[idx, "Exit_Date"]  = today
    df.loc[idx, "Exit_Price"] = f"{exit_price:.2f}"
    df.loc[idx, "Exit_Type"]  = exit_type
    df.loc[idx, "Return_%"]   = f"{ret_pct:.2f}"
    df.loc[idx, "Result"]     = result
    df.loc[idx, "Notes"]      = notes

    df.to_csv(JOURNAL_FILE, index=False)

    color = GREEN if result == "WIN" else RED
    print(f"\n  {color}{BOLD}{'🎉 WIN!' if result=='WIN' else '📉 LOSS'}{RESET}")
    print(f"  Return: {color}{ret_pct:+.2f}%{RESET}")
    print(f"  Saved to {JOURNAL_FILE} ✅")

    # Show updated stats
    df2   = pd.read_csv(JOURNAL_FILE)
    done  = df2[df2["Result"].isin(["WIN","LOSS"])]
    if len(done) > 0:
        wins = (done["Result"] == "WIN").sum()
        wr   = wins / len(done) * 100
        avg  = pd.to_numeric(done["Return_%"], errors="coerce").mean()
        print(f"\n  Overall: {len(done)} trades | "
              f"WR: {GREEN if wr>=50 else RED}{wr:.1f}%{RESET} | "
              f"Avg: {GREEN if avg>0 else RED}{avg:+.2f}%{RESET}")

if __name__ == "__main__":
    update_trade()
