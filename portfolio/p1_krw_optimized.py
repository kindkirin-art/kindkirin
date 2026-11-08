"""P1 in KRW with the optimised rules.

- TQQQ switching: unchanged (TQQQ SMA200, 1% band, buy 3 / sell 2), converted to KRW.
- BTC: BTC/KRW (Upbit) SMA30, 2% band, 1 day (fee 0.05%, KRW cash) vs SMA120 1% band on BTC/KRW.
- Gold: KRW gold, monthly SMA12, 5% band, 2-month confirmation (KRW cash when out) vs gold held.
"""
import pandas as pd
import gold_krw as K
import gold_filter_grid as G
import btc_krw_grid as BG
import gold_regime_switch as rs
import portfolio_long as pb
import compare_three as C

GOLD_RULE = ("sma", 12, 0.05, 2)
btc30, _ = BG.run("sma", 30, 0.02, 1)
btc120, _ = BG.run("sma", 120, 0.01, 1)
gold_h = G.signal(K.m_daily_k, *GOLD_RULE)


def curve(btc=None, gold=False, start="1999-01-04", switch=pb.SWITCH):
    sl = K.SLK.copy()
    if btc is not None:
        e = btc.reindex(sl.index).ffill().fillna(1.0)
        sl["BTC"] = e / e.iloc[0]
    w = K.P1
    if gold:
        held = gold_h.loc[K.me_k.index].reindex(pb.stock).ffill().fillna(1)
        prev = held.shift().fillna(1)
        r = K.gold_krw_r * prev + (1 - prev) * K.cash_krw_d - (held != prev) * G.FEE
        s = (1 + r).cumprod().reindex(sl.index).ffill()
        sl["GF"] = s / s.iloc[0]
        w = tuple({("GF" if k == "GLD" else k): v for k, v in x.items()} for x in K.P1)
    rs.SL = sl
    return rs.fixed(*w, start, switch)


VARIANTS = {
    "A original (BTC USD120 b1, gold hold)": dict(),
    "B + gold SMA12 b5%": dict(gold=True),
    "C BTC KRW120 b1, gold hold": dict(btc=btc120),
    "D BTC KRW30 b2%, gold hold": dict(btc=btc30),
    "E optimised (BTC KRW30 b2% + gold SMA12 b5%)": dict(btc=btc30, gold=True),
}

if __name__ == "__main__":
    pd.set_option("display.width", 250)
    fmt = {k: "{:.1%}".format for k in ("CAGR", "MDD", "Vol", "Worst yr", "Worst 1y roll", "Worst day")}
    fmt.update({"Sharpe": "{:.2f}".format, "Calmar": "{:.2f}".format})
    for label, start, switch in [("LONG 1999~ (BTC from 2018)", "1999-01-04", pb.SWITCH),
                                 ("10Y 2016-11~ (BTC from start)", "2016-11-07", "2016-11-07")]:
        curves = pd.DataFrame({n: curve(**kw, start=start, switch=switch) for n, kw in VARIANTS.items()})
        print(f"\n===== {label}")
        print(pd.DataFrame({c: C.metrics(curves[c]) for c in curves}).T.to_string(formatters=fmt))
        per = [("1999-01-04", "2002-12-31"), ("2003-01-01", "2009-12-31"), ("2010-01-01", "2017-12-31"),
               ("2018-01-01", None), ("2021-01-01", None)] if start < "2000" else [("2018-01-01", None), ("2021-01-01", None)]
        for a, b in per:
            print(f"  {a[:4]}~{(b or 'now')[:4]}: " + " | ".join(
                f"{c.split(' ')[0]} {pb.stats(curves[c].loc[a:b])['CAGR']:.1%}/{pb.stats(curves[c].loc[a:b])['MDD']:.1%}" for c in curves))
        c2 = curves.loc[curves.index.isin(pb.stock)]
        dd = c2 / c2.cummax() - 1
        for c in curves:
            t = dd[c].idxmin()
            print(f"  MDD {c.split(' ')[0]}: {dd[c].min():.1%}  {c2[c][:t].idxmax().date()} -> {t.date()}")
        yr = pd.concat([c2.iloc[:1], c2.resample("YE").last()]).pct_change().dropna()
        yr.index = yr.index.year
        yr.columns = [c.split(" ")[0] for c in yr.columns]
        print((yr.loc[2016:] * 100).round(1).T.to_string())
