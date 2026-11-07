"""BTC 120-day SMA trend following (no wait) with/without a 1% band.

- Daily BTC-USD closes (7 days/week), SMA over the last 120 daily bars.
- No band : close > SMA -> buy at that close, close < SMA -> sell at that close.
- Band b  : buy when close > SMA*(1+b), sell when close < SMA*(1-b), else hold.
- Cash earns 0%. Per-trade cost FEE applied on each buy/sell.
"""
import pandas as pd
import yfinance as yf

END = "2026-11-07"
btc = yf.download("BTC-USD", start="2014-01-01", end=END, auto_adjust=True,
                  progress=False)["Close"].squeeze().dropna()
sma = btc.rolling(120).mean()
btc, sma = btc[sma.notna()], sma.dropna()
ret = btc.pct_change().fillna(0.0)


def run(band=0.0, fee=0.0, px=btc, ma=sma):
    pos, held, trades = 0, [], 0
    for p, m in zip(px.values, ma.values):
        if pos == 0 and p > m * (1 + band):
            pos, trades = 1, trades + 1
        elif pos == 1 and p < m * (1 - band):
            pos = 0
        held.append(pos)
    held = pd.Series(held, index=px.index)
    prev = held.shift().fillna(0)
    r = px.pct_change().fillna(0.0) * prev - (held != prev) * fee   # position from yesterday close
    return (1 + r).cumprod(), trades, held.mean()


def stats(s, start=None):
    s = s.loc[start:] / s.loc[start:].iloc[0]
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    dd = (s / s.cummax() - 1).min()
    r = s.pct_change().dropna()
    cagr = s.iloc[-1] ** (1 / yrs) - 1
    return cagr, dd, r.mean() / r.std() * 365 ** 0.5, cagr / -dd


if __name__ == "__main__":
    print(f"Period {btc.index[0].date()} ~ {btc.index[-1].date()}")
    for fee in (0.0, 0.001, 0.002):
        print(f"\nfee {fee:.1%}   CAGR    MDD  Sharpe Calmar  buys  in-market | last 10y CAGR  MDD")
        for name, b in [("no band", 0.0), ("band 1%", 0.01), ("band 2%", 0.02), ("band 3%", 0.03)]:
            s, n, ex = run(b, fee)
            c, d, sh, ca = stats(s)
            c10, d10, *_ = stats(s, "2016-11-07")
            print(f"{name:8s} {c:7.1%} {d:6.1%} {sh:6.2f} {ca:6.2f} {n:5d} {ex:9.0%} | {c10:12.1%} {d10:6.1%}")
        bh = btc / btc.iloc[0]
        c, d, sh, ca = stats(bh); c10, d10, *_ = stats(bh, "2016-11-07")
        print(f"{'BTC B&H':8s} {c:7.1%} {d:6.1%} {sh:6.2f} {ca:6.2f}     -      100% | {c10:12.1%} {d10:6.1%}")

    A, nA, _ = run(0.0, 0.001); B, nB, _ = run(0.01, 0.001)
    cur = pd.DataFrame({"no_band": A, "band_1pct": B, "BTC_BH": btc / btc.iloc[0]})
    cur.to_csv(__file__.replace("backtest.py", "equity_curves.csv"), float_format="%.6f")
    yr = pd.concat([cur.iloc[:1], cur.resample("YE").last()]).pct_change().dropna()
    yr.index = yr.index.year
    print("\nYearly returns % (fee 0.1%)\n", (yr * 100).round(1).to_string())
