"""Whipsaw filter: fixed % band vs ATR band around TQQQ 200MA (buy 3 / sell 2, no hedge, fee 0.1%).

ATR = Wilder ATR on TQQQ daily OHLC (real TQQQ from 2010-02).
Before 2010 (long test): ATR% of ^NDX/QQQ * 3, times the synthetic TQQQ price.
Band: buy above MA + k*ATR, sell below MA - k*ATR.
"""
import pandas as pd
import yfinance as yf
import backtest as bt
import backtest_long as bl

FEE = 0.001
d = yf.download(["^NDX", "TQQQ"], start="1985-01-01", end=bt.END, auto_adjust=True, progress=False)


def tr_pct(t):
    h, l, c = d["High"][t], d["Low"][t], d["Close"][t]
    pc = c.shift()
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1) / pc


trp = (tr_pct("^NDX") * 3).where(d["Close"]["TQQQ"].isna(), tr_pct("TQQQ")).dropna()


def with_atr(px, n):
    atrp = trp.ewm(alpha=1 / n, adjust=False).mean()
    px = px.copy()
    px["ATR"] = atrp.reindex(px.index) * px["TQQQ"]
    return px


tests = [("no filter", {}),
         ("band 1%", {"band": 0.01}), ("band 2%", {"band": 0.02})]
for n in (14, 20):
    for k in (0.25, 0.5, 0.75, 1.0, 1.5):
        tests.append((f"ATR{n} x{k}", {"atr_k": k, "_n": n}))

periods = {"1999~2026": bl.px, "1999~2002": bl.px.loc[:"2002"], "2003~2009": bl.px.loc["2003":"2009"],
           "2010~2016.10": bl.px.loc["2010":"2016-11-06"], "10y": bt.load()}
rows = []
for name, kw in tests:
    kw = dict(kw); n = kw.pop("_n", 14)
    row = {"filter": name}
    for pname, px in periods.items():
        s, log = bt.run(with_atr(px, n), FEE=FEE, **kw)
        st = bt.stats(s)
        row[pname] = f"{st['CAGR']:.1%} / {st['MDD']:.0%}"
        if pname in ("1999~2026", "10y"):
            row[pname + " buys"] = sum(1 for x in log if x[1] == "BUY")
    rows.append(row)
print("CAGR / MDD  (fee 0.1%)")
print(pd.DataFrame(rows).set_index("filter").to_string())
atr = with_atr(bl.px, 14)
print("\nTQQQ ATR14 as % of 200MA, yearly median:")
print((atr.ATR / atr.MA200 * 100).groupby(atr.index.year).median().round(1).to_string())
