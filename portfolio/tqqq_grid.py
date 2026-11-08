"""TQQQ switching optimisation (long test 1999~, synthetic TQQQ before 2010).

Same mechanics as tqqq_200ma/backtest.py run(): buy after `buy_n` closes above MA*(1+band), sell (TQQQ+SCHD)
after `sell_n` closes below MA*(1-band); +20/50/100/200% -> 25/25/50/50% of TQQQ to SCHD; fee 0.1%;
idle cash earns T-bills. Signal source: TQQQ itself or QQQ; MA: SMA or EMA.
"""
import os
import numpy as np
import pandas as pd
import portfolio_long as pb

HERE = os.path.dirname(os.path.abspath(__file__))
PX = pb.bl.px
T, S = PX["TQQQ"].values, PX["SCHD"].values
CASH = pb.cash_r.reindex(PX.index).fillna(0).values
LEVELS = [(0.20, 0.25), (0.50, 0.25), (1.00, 0.50), (2.00, 0.50)]
FEE = 0.001


def ma_of(src, kind, n):
    full = pb.bl.raw  # long history for warm-up
    s = {"TQQQ": pb.bl.tqqq_syn, "QQQ": (1 + pb.bl.r).cumprod()}[src]
    if src == "TQQQ":
        s = pb.bl.splice(pb.bl.tqqq_syn, full["TQQQ"])
    m = s.rolling(n).mean() if kind == "sma" else s.ewm(span=n, adjust=False, min_periods=n).mean()
    return s.reindex(PX.index).values, m.reindex(PX.index).values


def fast_run(src="TQQQ", kind="sma", n=200, band=0.0, buy_n=3, sell_n=2, ladder=True):
    sig, ma = ma_of(src, kind, n)
    cash, st, ss = 1.0, 0.0, 0.0
    inv, up, dn, entry, hit = False, 0, 0, 0.0, 0
    eq = np.empty(len(T))
    buys = 0
    for i in range(len(T)):
        if cash > 0:
            cash *= 1 + CASH[i]
        p, m = sig[i], ma[i]
        up = up + 1 if p > m * (1 + band) else 0
        dn = dn + 1 if p < m * (1 - band) else 0
        t, s = T[i], S[i]
        if not inv and up >= buy_n:
            amt = cash / (1 + FEE)
            st, cash = amt / t, 0.0
            inv, entry, hit, buys = True, t, 0, buys + 1
        elif inv and dn >= sell_n:
            v = st * t + ss * s
            cash += v * (1 - FEE)
            st = ss = 0.0
            inv = False
        elif inv and ladder:
            ret = t / entry - 1
            while hit < 4 and ret >= LEVELS[hit][0]:
                v = st * t * LEVELS[hit][1]
                st -= v / t
                ss += v * (1 - FEE) / (1 + FEE) / s
                cash -= v * FEE - v * FEE  # proceeds fully reinvested (fees applied above)
                hit += 1
        eq[i] = cash + st * t + ss * s
    return pd.Series(eq, index=PX.index), buys


def st(s, a=None, b=None):
    x = s.loc[a:b]
    x = x / x.iloc[0]
    yrs = (x.index[-1] - x.index[0]).days / 365.25
    r = x.pct_change().dropna()
    c = x.iloc[-1] ** (1 / yrs) - 1
    d = (x / x.cummax() - 1).min()
    return c, d, r.mean() / r.std() * 252 ** 0.5


if __name__ == "__main__":
    import backtest as bt   # tqqq_200ma/backtest.py, for validation
    ref, _ = bt.run(PX, FEE=FEE, band=0.01, cash_r=pb.cash_r)
    me, _ = fast_run(band=0.01)
    print("validation final (bt.run vs fast):", round(ref.iloc[-1], 2), round(me.iloc[-1], 2))
    rows = []
    for src in ("TQQQ", "QQQ"):
        for kind in ("sma", "ema"):
            for n in (100, 150, 200, 250):
                for band in (0.0, 0.01, 0.02, 0.03):
                    for bn in (1, 2, 3, 5):
                        for sn in (1, 2, 3):
                            s, buys = fast_run(src, kind, n, band, bn, sn)
                            a, h1, h2, t10 = st(s), st(s, None, "2012-12-31"), st(s, "2013-01-01"), st(s, "2016-11-07")
                            rows.append({"rule": f"{src} {kind}{n} b{band:.0%} buy{bn} sell{sn}", "buys": buys,
                                         "CAGR": a[0], "MDD": a[1], "Sharpe": a[2], "99~12": h1[0], "99~12 MDD": h1[1],
                                         "13~": h2[0], "13~ MDD": h2[1], "10y": t10[0], "10y MDD": t10[1]})
    df = pd.DataFrame(rows).set_index("rule")
    df.to_csv(os.path.join(HERE, "tqqq_grid.csv"), float_format="%.4f")
    print(len(df), "rules")
