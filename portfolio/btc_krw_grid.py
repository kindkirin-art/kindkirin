"""BTC trend-following optimisation on BTC/KRW (daily, 7 days/week).

Price: Upbit KRW-BTC closes (09:00 KST) from 2017-09-25 (btc_120ma/upbit_krw_btc.csv), yfinance BTC-KRW
before (scaled to join). Fee 0.05% (Upbit), idle cash earns the KRW short rate (data_krw).
Rule: MA (SMA/EMA, N days); buy after k closes above MA*(1+b), sell after k closes below MA*(1-b).
Evaluated (a) BTC alone: 2015~, 2015~2020, 2021~ and (b) P1 in KRW (10y from 2016-11, BTC from start,
gold held) with the BTC sleeve replaced.
"""
import os
import itertools
import pandas as pd
import yfinance as yf
import gold_krw as K
import gold_regime_switch as rs
import portfolio_long as pb

HERE = os.path.dirname(os.path.abspath(__file__))
FEE = 0.0005
up = pd.read_csv(os.path.join(HERE, "..", "btc_120ma", "upbit_krw_btc.csv"), index_col=0, parse_dates=True)["close"]
up = up[up.index < pb.END]
yk = yf.download("BTC-KRW", start="2014-01-01", end=pb.END, auto_adjust=True, progress=False)["Close"].squeeze().dropna()
t0 = up.index[0]
px = pd.concat([yk[yk.index < t0] * up.iloc[0] / yk.loc[t0], up])
ret = px.pct_change().fillna(0)
cash = (K.rate_m / 100 / 365).reindex(px.index.union(K.rate_m.index)).ffill().reindex(px.index).fillna(0)


def run(kind, n, band, k):
    ma = px.rolling(n).mean() if kind == "sma" else px.ewm(span=n, adjust=False, min_periods=n).mean()
    pos, upc, dnc, held = 0, 0, 0, []
    for p, m in zip(px.values, ma.values):
        if m == m:
            upc = upc + 1 if p > m * (1 + band) else 0
            dnc = dnc + 1 if p < m * (1 - band) else 0
            if pos == 0 and upc >= k:
                pos = 1
            elif pos == 1 and dnc >= k:
                pos = 0
        held.append(pos)
    held = pd.Series(held, index=px.index)
    prev = held.shift().fillna(0)
    eq = (1 + ret * prev + (1 - prev) * cash - (held != prev) * FEE).cumprod()
    return eq, int(((held == 1) & (prev == 0)).loc["2015":].sum())


def cm(s, a, b=None):
    s = s.loc[a:b]
    s = s / s.iloc[0]
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return s.iloc[-1] ** (1 / yrs) - 1, (s / s.cummax() - 1).min()


def p1(eq):
    sl = K.SLK.copy()
    e = eq.reindex(sl.index).ffill().fillna(1.0)          # BTC unused before 2016-11
    sl["BTC"] = e / e.iloc[0]
    rs.SL = sl
    x = rs.fixed(*K.P1, "2016-11-07", "2016-11-07")
    return pb.stats(x), pb.stats(x.loc["2018-01-01":]), pb.stats(x.loc["2021-01-01":])


if __name__ == "__main__":
    pd.set_option("display.width", 260)
    bh = px / px.iloc[0]
    for lab, a, b in [("2015~", "2015-01-15", None), ("15~20", "2015-01-15", "2020-12-31"), ("21~", "2021-01-01", None)]:
        c, d = cm(bh, a, b)
        print(f"BTC/KRW B&H {lab}: {c:.1%} / {d:.1%}")
    rows = []
    grid = [("sma", n, b, k) for n in (20, 30, 50, 60, 90, 100, 120, 150, 200) for b in (0, .01, .02, .03, .05) for k in (1, 2, 3)]
    grid += [("ema", n, b, k) for n in (20, 50, 100, 120, 200) for b in (0, .01, .02, .03, .05) for k in (1, 2, 3)]
    for rule in grid:
        eq, buys = run(*rule)
        a = cm(eq, "2015-01-15"); e1 = cm(eq, "2015-01-15", "2020-12-31"); e2 = cm(eq, "2021-01-01")
        s10, s18, s21 = p1(eq)
        rows.append({"rule": f"{rule[0]}{rule[1]} b{rule[2]:.0%} k{rule[3]}", "buys": buys,
                     "BTC CAGR": a[0], "BTC MDD": a[1], "15~20": e1[0], "15~20 MDD": e1[1], "21~": e2[0], "21~ MDD": e2[1],
                     "P1 10y": s10["CAGR"], "P1 10y MDD": s10["MDD"], "P1 Sharpe": s10["Sharpe"],
                     "P1 18~": s18["CAGR"], "P1 18~ MDD": s18["MDD"], "P1 21~": s21["CAGR"]})
    df = pd.DataFrame(rows).set_index("rule")
    df.to_csv(os.path.join(HERE, "btc_krw_grid.csv"), float_format="%.4f")
    fmt = {c: (lambda x: f"{x:.1%}") for c in df.columns if c not in ("buys", "P1 Sharpe")}
    fmt["P1 Sharpe"] = "{:.2f}".format
    print(df.sort_values("P1 10y", ascending=False).head(40).to_string(formatters=fmt))
    print("\nBaseline rows:")
    print(df.loc[["sma120 b0% k1", "sma120 b1% k1"]].to_string(formatters=fmt))
