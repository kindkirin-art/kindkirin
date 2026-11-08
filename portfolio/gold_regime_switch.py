"""P1 with gold-regime switching.

Gold above its monthly 10-month SMA (month-end decision) -> P1 weights, else P2 weights:
  2018~      : gold ON  TQQQ_SW 40 / Gold 30 / BTC 20 / SGOV 10
               gold OFF TQQQ_SW 50 / BTC 30 / SGOV 20
  1999~2017  : BTC weight goes to TQQQ_SW -> ON 60/30/0/10, OFF 80/0/0/20
Rebalanced at every regime change (month-end close, 0.1% on turnover) and at each year end
(no cost, as in the other portfolio tests).
"""
import pandas as pd
import portfolio_long as pb
import gold_monthly_filter as gm

FEE = 0.001
SL = pb.sleeves
ON_PRE, ON_POST = pb.PORTS["P1 TQQQ40/Gold30/BTC20/SGOV10"]
OFF_PRE, OFF_POST = pb.PORTS["P2 TQQQ50/BTC30/SGOV20"]


def regime_portfolio(held, start="1999-01-04", switch=pb.SWITCH):
    sl = SL.loc[start:]
    sl = sl / sl.iloc[0]
    reg = held.reindex(sl.index, method="ffill").fillna(1).astype(int)
    post = lambda d: d >= pd.Timestamp(switch)
    target = lambda d, r: (ON_POST if r else OFF_POST) if post(d) else (ON_PRE if r else OFF_PRE)
    w = target(sl.index[0], reg.iloc[0])
    units = {k: w[k] / sl.iloc[0][k] for k in w}
    out, cur, switches = [], reg.iloc[0], 0
    for d, row in sl.iterrows():
        val = sum(units[k] * row[k] for k in units)
        r = reg.loc[d]
        change = r != cur and d in pb.stock
        if change or d in pb.rebal:
            new = target(d, r)
            if change:
                old = {k: units[k] * row[k] / val for k in units}
                turnover = sum(abs(new.get(k, 0) - old.get(k, 0)) for k in set(new) | set(old)) / 2
                val *= 1 - turnover * FEE
                cur, switches = r, switches + 1
            units = {k: val * new[k] / row[k] for k in new}
        out.append(val)
    return pd.Series(out, index=sl.index), switches


def fixed(w_pre, w_post, start="1999-01-04", switch=pb.SWITCH):
    on = pd.Series(1, index=pb.stock)
    global ON_PRE, ON_POST
    saved = ON_PRE, ON_POST
    ON_PRE, ON_POST = w_pre, w_post
    s, _ = regime_portfolio(on, start, switch)
    ON_PRE, ON_POST = saved
    return s


if __name__ == "__main__":
    held = gm.gold_held(10)
    print(f"gold OFF share of days 1999~: {1 - held.mean():.0%}")
    for label, start, switch in [("Long 1999~ (BTC from 2018)", "1999-01-04", pb.SWITCH),
                                 ("10y 2016-11~ (BTC from start, like portfolio_10y.py)", "2016-11-07", "2016-11-07")]:
        sw, n = regime_portfolio(held, start, switch)
        curves = pd.DataFrame({
            "P1 (gold B&H)": fixed(ON_PRE, ON_POST, start, switch),
            "P1 gold-off -> 50/30/20": sw,
            "P2": fixed(OFF_PRE, OFF_POST, start, switch),
        })
        print(f"\n=== {label}: regime switches {n}")
        periods = [(start, None)] + ([("1999-01-04", "2002-12-31"), ("2003-01-01", "2009-12-31"),
                                      ("2010-01-01", "2017-12-31"), ("2018-01-01", None)]
                                     if start < "2000" else [])
        for a, b in periods:
            print(f" {a} ~ {b or 'now'}")
            for c in curves:
                st = pb.stats(curves[c].loc[a:b])
                print(f"   {c:26s} x{st['Final x']:8.1f}  CAGR {st['CAGR']:6.1%}  MDD {st['MDD']:6.1%}  "
                      f"Sharpe {st['Sharpe']:5.2f}  Calmar {st['Calmar']:5.2f}")
        c2 = curves.loc[curves.index.isin(pb.stock)]
        dd = c2 / c2.cummax() - 1
        for c in curves:
            t = dd[c].idxmin()
            print(f"   MDD {c:26s} {dd[c].min():6.1%}  {c2[c][:t].idxmax().date()} -> {t.date()}")
        yr = pd.concat([c2.iloc[:1], c2.resample("YE").last()]).pct_change().dropna()
        yr.index = yr.index.year
        print((yr * 100).round(1).T.to_string())
