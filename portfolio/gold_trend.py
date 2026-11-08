"""Does trend following help on gold?

1) Daily 1999-01 ~ 2026-11 (gold chain from portfolio_long.py), idle cash earns T-bills, fee 0.1%.
2) Monthly 1971 ~ 2026 (monthly average gold price, datasets/gold-prices), 10/12-month SMA,
   cash earns 13-week T-bill (^IRX). Includes the 1980~1999 gold bear market.
3) P1 portfolio (long and last 10y) with the gold sleeve replaced by gold trend following.
"""
import os
import urllib.request
import io
import pandas as pd
import yfinance as yf
import portfolio_long as pb

HERE = os.path.dirname(os.path.abspath(__file__))
FEE = 0.001


def trend(price, cash_r, n, band=0.0, buy_n=1, sell_n=1, fee=FEE, ppy=252):
    ma = price.rolling(n).mean()
    pos, up, dn, held = 0, 0, 0, []
    for p, m in zip(price.values, ma.values):
        if m == m:
            up = up + 1 if p > m * (1 + band) else 0
            dn = dn + 1 if p < m * (1 - band) else 0
            if pos == 0 and up >= buy_n:
                pos = 1
            elif pos == 1 and dn >= sell_n:
                pos = 0
        held.append(pos)
    held = pd.Series(held, index=price.index)
    prev = held.shift().fillna(0)
    r = price.pct_change().fillna(0) * prev + (1 - prev) * cash_r - (held != prev) * fee
    return (1 + r).cumprod(), int(((held == 1) & (prev == 0)).sum()), held.mean()


def stats(s, ppy=252):
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    r = s.pct_change().dropna()
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1
    dd = (s / s.cummax() - 1).min()
    return cagr, dd, r.mean() / r.std() * ppy ** 0.5


def show(label, s, n=None, ex=None, ppy=252):
    c, d, sh = stats(s, ppy)
    extra = "" if n is None else f"  buys {n:3d}  in-market {ex:4.0%}"
    print(f"  {label:28s} CAGR {c:6.1%}  MDD {d:6.1%}  Sharpe {sh:5.2f}{extra}")


if __name__ == "__main__":
    # 1) daily
    stock = pb.stock
    gold = (1 + pb.gold_r).cumprod()
    cash = pb.cash_r
    variants = {
        "MA200": dict(n=200), "MA200 band1%": dict(n=200, band=0.01),
        "MA200 band1% buy3/sell2": dict(n=200, band=0.01, buy_n=3, sell_n=2),
        "MA120 band1%": dict(n=120, band=0.01), "MA100 band1%": dict(n=100, band=0.01),
        "MA50 band1%": dict(n=50, band=0.01),
    }
    for a, b in [("1999-01-04", None), ("2011-01-01", "2015-12-31"), ("2016-11-07", None)]:
        print(f"\nDaily gold {a} ~ {b or 'now'}")
        g, c_ = gold.loc[:b], cash.loc[:b]
        show("Gold B&H", g.loc[a:] / g.loc[a:].iloc[0])
        for k, v in variants.items():
            s, n, ex = trend(g, c_, **v)
            s = s.loc[a:]
            show(k, s / s.iloc[0], n, ex)

    # 2) monthly 1971~
    raw = urllib.request.urlopen(
        "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv", timeout=30).read()
    gm = pd.read_csv(io.BytesIO(raw))
    gm.index = pd.to_datetime(gm["Date"]) + pd.offsets.MonthEnd(0)
    gm = gm["Price"].loc["1970":]
    irx = yf.download("^IRX", start="1960-01-01", progress=False)["Close"].squeeze()
    cm = (irx.resample("ME").mean() / 100 / 12).reindex(gm.index).ffill().fillna(0.0)
    for a, b in [("1971-12-31", None), ("1980-01-31", "1999-12-31"), ("2000-01-31", None)]:
        print(f"\nMonthly gold {a[:7]} ~ {(b or 'now')[:7]}")
        g = gm.loc[:b]
        show("Gold B&H", g.loc[a:] / g.loc[a:].iloc[0], ppy=12)
        for n in (10, 12):
            s, k, ex = trend(g, cm.loc[:b], n, ppy=12)
            s = s.loc[a:]
            show(f"{n}-month SMA", s / s.iloc[0], k, ex, ppy=12)

    # 3) P1 with gold trend sleeve
    gt, _, _ = trend(gold, cash, n=200, band=0.01)
    pb.sleeves["GOLD_T"] = gt.reindex(pb.sleeves.index).ffill() / gt.reindex(pb.sleeves.index).ffill().iloc[0]
    P1 = pb.PORTS["P1 TQQQ40/Gold30/BTC20/SGOV10"]
    swap = lambda w: {("GOLD_T" if k == "GLD" else k): v for k, v in w.items()}
    curves = pd.DataFrame({"P1 gold B&H": pb.portfolio(*P1), "P1 gold MA200 band1%": pb.portfolio(*map(swap, P1)),
                           "P2": pb.portfolio(*pb.PORTS["P2 TQQQ50/BTC30/SGOV20"])})
    for a in ("1999-01-04", "2016-11-07", "2018-01-01"):
        print(f"\nPortfolio from {a}")
        for c in curves:
            st = pb.stats(curves[c].loc[a:])
            print(f"  {c:22s} CAGR {st['CAGR']:6.1%}  MDD {st['MDD']:6.1%}  Sharpe {st['Sharpe']:5.2f}  Calmar {st['Calmar']:5.2f}")
