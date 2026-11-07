"""Whipsaw filters: buy/sell wait-day grid and a +-band around the 200MA (no hedge, fee 0.1%)."""
import sys, pandas as pd
sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import backtest as bt, backtest_long as bl
p10 = bt.load(); pL = bl.px
def row(px, **k):
    s, log = bt.run(px, FEE=0.001, **k); st = bt.stats(s)
    return st["CAGR"], st["MDD"], sum(1 for x in log if x[1] == "BUY")
print("buy/sell wait grid (fee 0.1%): long CAGR | MDD | trades || 10y CAGR | MDD")
for b in [1, 2, 3, 4, 5]:
    for se in [1, 2, 3, 4, 5]:
        L = row(pL, buy_n=b, sell_n=se); T = row(p10, buy_n=b, sell_n=se)
        print(f"buy {b} sell {se}: {L[0]:6.1%} {L[1]:6.1%} {L[2]:3d} || {T[0]:6.1%} {T[1]:6.1%} {T[2]:3d}")
print("\nband filter (buy 3/sell 2 + MA +-band)")
for band in [0.0, 0.01, 0.02, 0.03, 0.05]:
    L = row(pL, band=band); T = row(p10, band=band)
    print(f"band {band:.0%}: {L[0]:6.1%} {L[1]:6.1%} {L[2]:3d} || {T[0]:6.1%} {T[1]:6.1%} {T[2]:3d}")
