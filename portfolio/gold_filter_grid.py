"""Search for a monthly gold filter that avoids long gold bear markets at the lowest cost to P1.

Signal (evaluated at each month end, applied for the next month):
  kind "sma"/"ema": price vs N-month moving average, band b (buy above MA*(1+b), sell below MA*(1-b)),
                    k consecutive months needed to switch.
  kind "mom":       price vs price N months ago (time-series momentum), same band/confirm rules.
A) Gold alone, monthly 1972~2026 (monthly average prices, gold_monthly_1960_2026.csv), cash = T-bill.
   Key window: 1980-01 ~ 1999-12 gold bear market.
B) P1 (long test 1999~, BTC from 2018) with the gold sleeve filtered on month-end closes of the daily gold chain.
"""
import os
import itertools
import pandas as pd
import yfinance as yf
import portfolio_long as pb
import gold_regime_switch as rs

HERE = os.path.dirname(os.path.abspath(__file__))
FEE = 0.001

gm = pd.read_csv(os.path.join(HERE, "gold_monthly_1960_2026.csv"))
gm.index = pd.to_datetime(gm["Date"]) + pd.offsets.MonthEnd(0)
gm = gm["Price"]
irx = yf.download("^IRX", start="1960-01-01", progress=False)["Close"].squeeze()
cash_m = (irx.resample("ME").mean() / 100 / 12).reindex(gm.index).ffill().fillna(0.0)


def signal(m, kind, n, band=0.0, k=1):
    """m: month-end price series. Returns held (1/0) decided at each month end."""
    if kind == "sma":
        ref = m.rolling(n).mean()
    elif kind == "ema":
        ref = m.ewm(span=n, adjust=False, min_periods=n).mean()
    else:
        ref = m.shift(n)
    pos, up, dn, out = 1, 0, 0, []
    for p, r in zip(m.values, ref.values):
        if r == r:
            up = up + 1 if p > r * (1 + band) else 0
            dn = dn + 1 if p < r * (1 - band) else 0
            if pos == 0 and up >= k:
                pos = 1
            elif pos == 1 and dn >= k:
                pos = 0
        out.append(pos)
    return pd.Series(out, index=m.index)


def cagr_mdd(s):
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1, (s / s.cummax() - 1).min()


def monthly_alone(held):
    prev = held.shift().fillna(1)
    r = gm.pct_change().fillna(0) * prev + (1 - prev) * cash_m - (held != prev) * FEE
    return (1 + r).cumprod(), int((held.diff() == 1).sum())


# daily gold chain month-end closes, with monthly averages before 1999 for warm-up
gold_d = (1 + pb.gold_r).cumprod()
me = gold_d.groupby([gold_d.index.year, gold_d.index.month]).tail(1)
pre = gm.loc[:"1998-12-31"] / gm.loc["1999-01-31"] * me.iloc[0]
m_daily = pd.concat([pre, me])
P1 = pb.PORTS["P1 TQQQ40/Gold30/BTC20/SGOV10"]


def p1_with(held_m, start="1999-01-04", switch=pb.SWITCH):
    held = held_m.loc[me.index].reindex(pb.stock).ffill().fillna(1)
    prev = held.shift().fillna(1)
    r = pb.gold_r * prev + (1 - prev) * pb.cash_r - (held != prev) * FEE
    s = (1 + r).cumprod().reindex(pb.sleeves.index).ffill()
    pb.sleeves["GF"] = s / s.iloc[0]
    rs.SL = pb.sleeves
    sw = lambda w: {("GF" if k == "GLD" else k): v for k, v in w.items()}
    return rs.fixed(sw(P1[0]), sw(P1[1]), start, switch)


if __name__ == "__main__":
    base_long = rs.fixed(*P1)
    base_10 = rs.fixed(*P1, "2016-11-07", "2016-11-07")
    bl = pb.stats(base_long)
    b10 = pb.stats(base_10)
    a_all = cagr_mdd(gm.loc["1972":])
    a_bear = cagr_mdd(gm.loc["1980-01":"1999-12"])
    print(f"Gold B&H: 1972~ {a_all[0]:.1%}/{a_all[1]:.1%}  1980~99 {a_bear[0]:.1%}/{a_bear[1]:.1%}")
    print(f"P1 gold B&H: long {bl['CAGR']:.2%} / {bl['MDD']:.1%} Sharpe {bl['Sharpe']:.2f} | "
          f"10y {b10['CAGR']:.2%} / {b10['MDD']:.1%}")
    grid = [("sma", n, b, k) for n in (6, 8, 10, 12, 15, 18, 24) for b in (0.0, 0.02, 0.05) for k in (1, 2)]
    grid += [("ema", n, b, k) for n in (10, 12, 18, 24) for b in (0.0, 0.02) for k in (1, 2)]
    grid += [("mom", n, b, k) for n in (6, 12, 18, 24) for b in (0.0, 0.05) for k in (1, 2)]
    rows = []
    for kind, n, b, k in grid:
        s, buys = monthly_alone(signal(gm, kind, n, b, k))
        c_all, d_all = cagr_mdd(s.loc["1972":])
        c_bear, d_bear = cagr_mdd(s.loc["1980-01":"1999-12"])
        c_new, d_new = cagr_mdd(s.loc["2000-01":])
        hm = signal(m_daily, kind, n, b, k)
        pl = pb.stats(p1_with(hm))
        p10 = pb.stats(p1_with(hm, "2016-11-07", "2016-11-07"))
        rows.append({"rule": f"{kind}{n} b{b:.0%} k{k}", "buys72": buys,
                     "G 72~": c_all, "G MDD 72~": d_all, "G 80~99": c_bear, "G MDD 80~99": d_bear,
                     "G 00~": c_new, "G MDD 00~": d_new,
                     "P1 dCAGR": pl["CAGR"] - bl["CAGR"], "P1 MDD": pl["MDD"], "P1 Sharpe": pl["Sharpe"],
                     "P1 10y dCAGR": p10["CAGR"] - b10["CAGR"], "P1 10y MDD": p10["MDD"]})
    df = pd.DataFrame(rows).set_index("rule")
    df.to_csv(os.path.join(HERE, "gold_filter_grid.csv"), float_format="%.4f")
    pct = lambda x: f"{x:.1%}"
    fmt = {c: pct for c in df.columns if c not in ("buys72", "P1 Sharpe")}
    fmt["P1 Sharpe"] = "{:.2f}".format
    pd.set_option("display.width", 250)
    print(df.sort_values("P1 dCAGR", ascending=False).to_string(formatters=fmt))
