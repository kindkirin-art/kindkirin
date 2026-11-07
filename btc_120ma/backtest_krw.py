"""BTC 120-day SMA trend (no wait), no band vs 1% band, on KRW prices.

- Upbit KRW-BTC daily closes (09:00 KST) from 2017-09-25 (upbit_krw_btc.csv, run fetch_upbit.py to refresh).
- Before that: yfinance BTC-KRW (USD price x FX, no kimchi premium), scaled to join Upbit.
- Compared with the USD (BTC-USD) result over the same periods.
"""
import pandas as pd
import yfinance as yf
import backtest as b

up = pd.read_csv(__file__.replace("backtest_krw.py", "upbit_krw_btc.csv"),
                 index_col=0, parse_dates=True)["close"]
up = up[up.index < b.END]                       # drop today's unfinished candle
yk = yf.download("BTC-KRW", start="2014-01-01", end=b.END, auto_adjust=True,
                 progress=False)["Close"].squeeze().dropna()
t0 = up.index[0]
krw = pd.concat([yk[yk.index < t0] * up.iloc[0] / yk.loc[t0], up])
krw_ma = krw.rolling(120).mean()

# kimchi premium vs USD price x FX
prem = (up / yk.reindex(up.index) - 1).dropna()


def table(fee, start, end=None):
    out = []
    for cur, px, ma in [("KRW", krw, krw_ma), ("USD", b.btc, b.sma)]:
        px, ma = px.loc[:end], ma.loc[:end]
        for name, band in [("no band", 0.0), ("band 1%", 0.01)]:
            s, _, _ = b.run(band, fee, px.loc[ma.notna()], ma.dropna())
            c, d, sh, ca = b.stats(s, start)
            held = b.run(band, 0.0, px.loc[start:], ma.loc[start:])[1]
            out.append((cur, name, c, d, sh, held))
        bh = px.loc[start:] / px.loc[start:].iloc[0]
        c, d, sh, _ = b.stats(bh)
        out.append((cur, "B&H", c, d, sh, None))
    return out


if __name__ == "__main__":
    print("Kimchi premium (Upbit vs BTC-USD x FX): median "
          f"{prem.median():+.1%}, max {prem.max():+.1%} ({prem.idxmax().date()}), "
          f"min {prem.min():+.1%} ({prem.idxmin().date()})")
    for fee in (0.0005, 0.001):
        for start, label in [("2016-11-07", "last 10y (pre-2017-09 = BTC-KRW converted)"),
                             ("2018-01-23", "Upbit-only (after 120-day warm-up)")]:
            print(f"\n{label}, from {start}, fee {fee:.2%}")
            print(" cur  strategy     CAGR     MDD  Sharpe  buys")
            for cur, name, c, d, sh, n in table(fee, start):
                print(f" {cur}  {name:8s} {c:7.1%} {d:7.1%} {sh:6.2f}  {'' if n is None else n}")
    s0, _, _ = b.run(0.0, 0.0005, krw[krw_ma.notna()], krw_ma.dropna())
    s1, _, _ = b.run(0.01, 0.0005, krw[krw_ma.notna()], krw_ma.dropna())
    cur = pd.DataFrame({"no_band": s0, "band_1pct": s1}).loc["2016-11-07":]
    cur = cur / cur.iloc[0]
    yr = pd.concat([cur.iloc[:1], cur.resample("YE").last()]).pct_change().dropna()
    yr.index = yr.index.year
    print("\nKRW yearly returns % (fee 0.05%)\n", (yr * 100).round(1).to_string())
