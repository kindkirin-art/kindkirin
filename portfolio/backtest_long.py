"""Long portfolio backtest from the dot-com bubble (1999-01 ~ 2026-11).

Same sleeves/rules as backtest.py (TQQQ switching 1% band, BTC 120MA 1% band, gold, SGOV,
year-end rebalancing, 0.1% trading cost inside sleeves, idle cash earns T-bills).
Before 2018 the BTC weight is given to the TQQQ switching sleeve:
  P1: 1999~2017 TQQQ 60 / Gold 30 / SGOV 10  -> 2018~ TQQQ 40 / Gold 30 / BTC 20 / SGOV 10
  P2: 1999~2017 TQQQ 80 / SGOV 20            -> 2018~ TQQQ 50 / BTC 30 / SGOV 20
(switch at the 2017 year-end rebalance)

Data splices
- TQQQ/SQQQ/SCHD: synthetic before 2010/2011 (see tqqq_200ma/backtest_long.py)
- Gold: monthly gold price (datasets/gold-prices, interpolated daily) until 2000-08-30,
  gold futures GC=F until 2004-11-18, GLD after.
- Cash/SGOV: 13-week T-bill (^IRX) until 2007-05-30, BIL until 2020-05-26, SGOV after.
"""
import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tqqq_200ma"))
import backtest as bt
import backtest_long as bl

START, END, FEE, BAND, SWITCH = "1999-01-04", bt.END, 0.001, 0.01, "2017-12-01"
PORTS = {
    "P1 TQQQ40/Gold30/BTC20/SGOV10": ({"TQQQ_SW": 0.6, "GLD": 0.3, "SGOV": 0.1},
                                      {"TQQQ_SW": 0.4, "GLD": 0.3, "BTC": 0.2, "SGOV": 0.1}),
    "P2 TQQQ50/BTC30/SGOV20": ({"TQQQ_SW": 0.8, "SGOV": 0.2},
                               {"TQQQ_SW": 0.5, "BTC": 0.3, "SGOV": 0.2}),
}

px = yf.download(["GLD", "GC=F", "SGOV", "BIL", "^IRX", "BTC-USD"], start="1997-01-01", end=END,
                 auto_adjust=True, progress=False)["Close"]
stock = bl.px.index                                   # US trading days from 1999-01-04
s = px.reindex(stock)


def chain(*series):
    """Daily returns from the first series that has data on both days."""
    out = pd.Series(np.nan, index=stock)
    for x in series:
        out = out.fillna(x.pct_change().where(x.shift().notna()))
    return out.fillna(0.0)


irx = (px["^IRX"].ffill() / 100 / 252).reindex(stock).ffill()
cash_r = chain(s["SGOV"], s["BIL"]).where(lambda r: s["BIL"].shift().notna() | s["SGOV"].shift().notna(), irx)

gm = pd.read_csv(os.path.join(HERE, "gold_monthly_1998_2000.csv"))
gm.index = pd.to_datetime(gm["Date"]) + pd.offsets.MonthEnd(0)
gm = np.exp(np.log(gm["Price"]).reindex(gm.index.union(stock)).interpolate("time")).reindex(stock)
gold_r = chain(s["GLD"], s["GC=F"].ffill(), gm)

# TQQQ switching sleeve (synthetic TQQQ before 2010)
tq_eq, _ = bt.run(bl.px, FEE=FEE, band=BAND, cash_r=cash_r)

# BTC sleeve
btc = px["BTC-USD"].dropna()
ma = btc.rolling(120).mean()
pos, held = 0, []
for p, m in zip(btc.values, ma.values):
    if pos == 0 and p > m * (1 + BAND):
        pos = 1
    elif pos == 1 and p < m * (1 - BAND):
        pos = 0
    held.append(pos)
held = pd.Series(held, index=btc.index)
prev = held.shift().fillna(0)
btc_eq = (1 + btc.pct_change().fillna(0) * prev
          + (1 - prev) * cash_r.reindex(btc.index).fillna(0) - (held != prev) * FEE).cumprod()

cal = stock.union(btc.index)
cal = cal[(cal >= START) & (cal < END)]
sleeves = pd.DataFrame({
    "TQQQ_SW": tq_eq.reindex(cal).ffill(),
    "GLD": (1 + gold_r).cumprod().reindex(cal).ffill(),
    "BTC": btc_eq.reindex(cal).ffill().fillna(1.0),     # unused before 2018
    "SGOV": (1 + cash_r).cumprod().reindex(cal).ffill(),
    "QQQ": bl.px["QQQ"].reindex(cal).ffill(),
    "TQQQ_BH": bl.px["TQQQ"].reindex(cal).ffill(),
}).dropna()
sleeves = sleeves / sleeves.iloc[0]
rebal = set(pd.Series(stock[stock >= START]).groupby(stock[stock >= START].year).max())


def portfolio(w_pre, w_post):
    w = w_pre
    units = {k: w[k] / sleeves.iloc[0][k] for k in w}
    out = []
    for d, row in sleeves.iterrows():
        val = sum(units[k] * row[k] for k in units)
        out.append(val)
        if d in rebal:
            w = w_post if d >= pd.Timestamp(SWITCH) else w_pre
            units = {k: val * w[k] / row[k] for k in w}
    return pd.Series(out, index=sleeves.index)


def stats(x):
    x = x.loc[x.index.isin(stock)]
    yrs = (x.index[-1] - x.index[0]).days / 365.25
    r = x.pct_change().dropna()
    cagr = (x.iloc[-1] / x.iloc[0]) ** (1 / yrs) - 1
    dd = (x / x.cummax() - 1).min()
    return {"Final x": x.iloc[-1] / x.iloc[0], "CAGR": cagr, "MDD": dd, "Vol": r.std() * 252 ** 0.5,
            "Sharpe": r.mean() / r.std() * 252 ** 0.5, "Calmar": cagr / -dd}


if __name__ == "__main__":
    curves = pd.DataFrame({n: portfolio(*w) for n, w in PORTS.items()})
    for k, lab in [("TQQQ_SW", "TQQQ switching only"), ("GLD", "Gold only"),
                   ("QQQ", "QQQ B&H"), ("TQQQ_BH", "TQQQ B&H")]:
        curves[lab] = sleeves[k]
    curves.to_csv(os.path.join(HERE, "equity_curves_long.csv"), float_format="%.6f")
    fmt = {"Final x": "{:,.1f}".format, "CAGR": "{:.1%}".format, "MDD": "{:.1%}".format,
           "Vol": "{:.1%}".format, "Sharpe": "{:.2f}".format, "Calmar": "{:.2f}".format}
    print(f"Period {curves.index[0].date()} ~ {curves.index[-1].date()}, fee {FEE:.1%}, band {BAND:.0%}")
    print(pd.DataFrame({c: stats(curves[c]) for c in curves}).T.to_string(formatters=fmt))
    for a, b_ in [("1999-01-04", "2002-12-31"), ("2003-01-01", "2009-12-31"), ("2010-01-01", "2017-12-31"),
                  ("2018-01-01", END)]:
        print(f"\n{a} ~ {b_}")
        print(pd.DataFrame({c: stats(curves[c].loc[a:b_]) for c in curves}).T.to_string(formatters=fmt))
    c2 = curves.loc[curves.index.isin(stock)]
    yr = pd.concat([c2.iloc[:1], c2.resample("YE").last()]).pct_change().dropna()
    yr.index = yr.index.year
    print("\nYearly returns %\n", (yr * 100).round(1).to_string())
    dd = c2 / c2.cummax() - 1
    for c in PORTS:
        t = dd[c].idxmin()
        print(f"{c}: MDD {dd[c].min():.1%} peak {c2[c][:t].idxmax().date()} trough {t.date()}")
    # sanity: data sources
    print("\ncash_r ann. by year (%):", (cash_r.groupby(cash_r.index.year).sum() * 100).round(1).to_dict())
