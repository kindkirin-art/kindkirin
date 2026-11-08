"""Portfolio backtest (last 10 years): TQQQ switching (1% band) + gold + BTC 120MA (1% band) + SGOV.

Sleeves
- TQQQ switching: TQQQ 200MA, buy after 3 closes above MA*1.01, sell after 2 closes below
  MA*0.99 (sell TQQQ+SCHD). +20/50/100/200% -> 25/25/50/50% of TQQQ moved to SCHD.
- BTC: 120-day SMA, no wait, buy close > MA*1.01, sell close < MA*0.99.
- Gold: GLD.  SGOV (BIL before 2020-05-26).
- Idle cash inside the TQQQ and BTC sleeves earns SGOV/BIL.
- Trading cost 0.1% per trade inside the sleeves. Rebalanced to target weights at the
  last US trading day of each year (no cost on rebalancing). No taxes.
"""
import os
import sys
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tqqq_200ma"))
import backtest as bt

START, END, FEE, BAND = "2016-11-07", bt.END, 0.001, 0.01
PORTS = {
    "P1 TQQQ40/Gold30/BTC20/SGOV10": {"TQQQ_SW": 0.4, "GLD": 0.3, "BTC": 0.2, "SGOV": 0.1},
    "P2 TQQQ50/BTC30/SGOV20": {"TQQQ_SW": 0.5, "BTC": 0.3, "SGOV": 0.2},
}

px = yf.download(["GLD", "SGOV", "BIL", "BTC-USD", "QQQ"], start="2015-01-01", end=END,
                 auto_adjust=True, progress=False)["Close"]
stock = px["GLD"].dropna().index
cash_px = px.loc[stock]
cash_r = cash_px["SGOV"].pct_change().where(cash_px["SGOV"].shift().notna(),
                                            cash_px["BIL"].pct_change()).fillna(0.0)

# TQQQ switching sleeve
tq = bt.load()
tq_eq, tq_log = bt.run(tq, FEE=FEE, band=BAND, cash_r=cash_r)

# BTC sleeve (daily incl. weekends)
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
r_btc = (btc.pct_change().fillna(0) * prev + (1 - prev) * cash_r.reindex(btc.index).fillna(0)
         - (held != prev) * FEE)
btc_eq = (1 + r_btc).cumprod()

cal = btc.loc[START:].index
sleeves = pd.DataFrame({
    "TQQQ_SW": tq_eq.reindex(cal).ffill(),
    "GLD": px["GLD"].reindex(cal).ffill(),
    "BTC": btc_eq.reindex(cal),
    "SGOV": (1 + cash_r).cumprod().reindex(cal).ffill(),
    "QQQ": px["QQQ"].reindex(cal).ffill(),
}).dropna()
sleeves = sleeves / sleeves.iloc[0]
rebal = set(pd.Series(stock[stock >= START]).groupby(stock[stock >= START].year).max())


def portfolio(w):
    val, units, out = 1.0, None, []
    base = sleeves.iloc[0]
    units = {k: w[k] / base[k] for k in w}
    for d, row in sleeves.iterrows():
        val = sum(units[k] * row[k] for k in w)
        out.append(val)
        if d in rebal:
            units = {k: val * w[k] / row[k] for k in w}
    return pd.Series(out, index=sleeves.index)


def stats(s):
    s = s.loc[s.index.isin(stock)]          # sample on US trading days
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    r = s.pct_change().dropna()
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1
    dd = (s / s.cummax() - 1).min()
    return {"Final x": s.iloc[-1] / s.iloc[0], "CAGR": cagr, "MDD": dd, "Vol": r.std() * 252 ** 0.5,
            "Sharpe": r.mean() / r.std() * 252 ** 0.5, "Calmar": cagr / -dd}


if __name__ == "__main__":
    curves = pd.DataFrame({n: portfolio(w) for n, w in PORTS.items()})
    for k, lab in [("TQQQ_SW", "TQQQ switching only"), ("BTC", "BTC 120MA only"), ("GLD", "Gold only"),
                   ("QQQ", "QQQ B&H")]:
        curves[lab] = sleeves[k]
    curves.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "equity_curves.csv"),
                  float_format="%.6f")
    fmt = {"Final x": "{:.2f}".format, "CAGR": "{:.1%}".format, "MDD": "{:.1%}".format,
           "Vol": "{:.1%}".format, "Sharpe": "{:.2f}".format, "Calmar": "{:.2f}".format}
    print(f"Period {curves.index[0].date()} ~ {curves.index[-1].date()}, fee {FEE:.1%}, band {BAND:.0%}")
    print(pd.DataFrame({c: stats(curves[c]) for c in curves}).T.to_string(formatters=fmt))
    c2 = curves.loc[curves.index.isin(stock)]
    print("\nFrom 2018-01-01 (excluding the 2017 BTC bubble year)")
    print(pd.DataFrame({c: stats(curves[c].loc["2018-01-01":]) for c in curves}).T.to_string(formatters=fmt))
    yr = pd.concat([c2.iloc[:1], c2.resample("YE").last()]).pct_change().dropna()
    yr.index = yr.index.year
    print("\nYearly returns %\n", (yr * 100).round(1).to_string())
    dd = c2 / c2.cummax() - 1
    for c in list(PORTS):
        t = dd[c].idxmin()
        print(f"\n{c}: MDD {dd[c].min():.1%} peak {c2[c][:t].idxmax().date()} trough {t.date()}")
