"""Current signals for the P1 rules from fresh data (completed daily/monthly closes only)."""
import os
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
TODAY_UTC = pd.Timestamp.utcnow().tz_localize(None).normalize()


def state(price, n, band, buy_n=1, sell_n=1):
    ma = price.rolling(n).mean()
    pos, up, dn, last = 0, 0, 0, None
    for d, p, m in zip(price.index, price.values, ma.values):
        if m != m:
            continue
        up = up + 1 if p > m * (1 + band) else 0
        dn = dn + 1 if p < m * (1 - band) else 0
        if pos == 0 and up >= buy_n:
            pos, last = 1, (d, "매수")
        elif pos == 1 and dn >= sell_n:
            pos, last = 0, (d, "매도")
    m = ma.iloc[-1]
    return dict(pos=pos, last=last, date=price.index[-1], close=price.iloc[-1], ma=m,
                buy=m * (1 + band), sell=m * (1 - band), up=up, dn=dn)


if __name__ == "__main__":
    t = yf.download("TQQQ", start="2010-01-01", auto_adjust=False, progress=False)["Close"].squeeze().dropna()
    t = t[t.index < TODAY_UTC]                              # completed US sessions
    s = state(t, 200, 0.01, 3, 2)
    print("TQQQ", s)
    up = pd.read_csv(os.path.join(HERE, "..", "btc_120ma", "upbit_krw_btc.csv"), index_col=0, parse_dates=True)["close"]
    up = up[up.index < TODAY_UTC]                           # completed 09:00 KST candles
    for n, b in [(30, 0.02), (120, 0.01)]:
        print(f"BTC SMA{n}", state(up, n, b))
    g = yf.download(["GLD", "KRW=X"], start="2023-01-01", auto_adjust=True, progress=False)["Close"].ffill().dropna()
    gk = (g["GLD"] * g["KRW=X"])
    gk = gk[gk.index < TODAY_UTC]
    me = gk.groupby([gk.index.year, gk.index.month]).tail(1)
    me = me[me.index < TODAY_UTC.replace(day=1)]          # completed months
    sma = me.rolling(12).mean()
    print("Gold KRW month-end vs SMA12:")
    for d in me.index[-6:]:
        print(f"  {d.date()} ratio {me[d] / sma[d] - 1:+.1%}")
    print("Gold KRW latest daily vs last SMA12:", gk.index[-1].date(), f"{gk.iloc[-1] / sma.iloc[-1] - 1:+.1%}")
