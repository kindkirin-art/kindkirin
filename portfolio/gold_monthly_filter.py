"""P1 with a slow gold filter: monthly 10-month (and 12-month) SMA.

Signal at the last US trading day of each month: gold month-end close vs SMA of the last
N month-end closes. Above -> hold gold next month, below -> T-bills next month.
1998 month values come from gold_monthly_1998_2000.csv for the warm-up. Fee 0.1% per switch.
"""
import os
import pandas as pd
import gold_trend as g
import portfolio_long as pb

HERE = os.path.dirname(os.path.abspath(__file__))
gold = (1 + pb.gold_r).cumprod()
stock = pb.stock


def monthly_filter(n, band=0.0):
    me = gold.groupby([gold.index.year, gold.index.month]).tail(1)          # month-end closes
    pre = pd.read_csv(os.path.join(HERE, "gold_monthly_1998_2000.csv"))
    pre.index = pd.to_datetime(pre["Date"]) + pd.offsets.MonthEnd(0)
    pre = pre["Price"].loc[:"1998-12-31"] / pre["Price"].loc["1999-01-31"] * me.iloc[0]
    m = pd.concat([pre, me])
    sma = m.rolling(n).mean()
    sig = (m > sma * (1 + band)).astype(int).loc[me.index]                  # decided at month end
    held = sig.reindex(stock).ffill().fillna(1)                             # held after that close
    prev = held.shift().fillna(1)
    r = pb.gold_r * prev + (1 - prev) * pb.cash_r - (held != prev) * g.FEE
    return (1 + r).cumprod(), int(((held == 1) & (prev == 0)).sum()), held.mean()


if __name__ == "__main__":
    sl = pb.sleeves
    idx = sl.index
    print("Gold alone, 1999~2026")
    g.show("Gold B&H", gold)
    sleeves = {}
    for name, kw in [("10M SMA", dict(n=10)), ("12M SMA", dict(n=12)), ("10M SMA band1%", dict(n=10, band=0.01))]:
        s, k, ex = monthly_filter(**kw)
        g.show(name, s, k, ex)
        sleeves[name] = s
    gt, _, _ = g.trend(gold, pb.cash_r, n=200, band=0.01)
    sleeves["daily MA200 band1%"] = gt

    P1 = pb.PORTS["P1 TQQQ40/Gold30/BTC20/SGOV10"]
    curves = {"P1 gold B&H": pb.portfolio(*P1)}
    for name, s in sleeves.items():
        col = "G_" + name
        sl[col] = s.reindex(idx).ffill() / s.reindex(idx).ffill().iloc[0]
        swap = lambda w: {(col if k == "GLD" else k): v for k, v in w.items()}
        curves[f"P1 gold {name}"] = pb.portfolio(*map(swap, P1))
    curves["P2"] = pb.portfolio(*pb.PORTS["P2 TQQQ50/BTC30/SGOV20"])
    curves = pd.DataFrame(curves)
    for a, b in [("1999-01-04", None), ("1999-01-04", "2002-12-31"), ("2003-01-01", "2009-12-31"),
                 ("2010-01-01", "2017-12-31"), ("2018-01-01", None), ("2016-11-07", None)]:
        print(f"\nPortfolio {a} ~ {b or 'now'}")
        for c in curves:
            st = pb.stats(curves[c].loc[a:b])
            print(f"  {c:30s} CAGR {st['CAGR']:6.1%}  MDD {st['MDD']:6.1%}  Sharpe {st['Sharpe']:5.2f}  Calmar {st['Calmar']:5.2f}")
    c2 = curves.loc[curves.index.isin(stock)]
    dd = c2 / c2.cummax() - 1
    print()
    for c in curves:
        t = dd[c].idxmin()
        print(f"{c:30s} MDD {dd[c].min():6.1%}  {c2[c][:t].idxmax().date()} -> {t.date()}")
    yr = pd.concat([c2.iloc[:1], c2.resample("YE").last()]).pct_change().dropna()
    yr.index = yr.index.year
    print("\nYearly %\n", (yr[["P1 gold B&H", "P1 gold 10M SMA"]] * 100).round(1).T.to_string())
