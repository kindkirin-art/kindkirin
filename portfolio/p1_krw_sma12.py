"""P1 in KRW: gold B&H vs gold filtered by monthly SMA12, 5% band, 2-month confirmation (KRW gold signal)."""
import pandas as pd
import gold_krw as K
import gold_filter_grid as G
import portfolio_long as pb
import compare_three as C

RULE = ("sma", 12, 0.05, 2)

if __name__ == "__main__":
    hm = G.signal(K.m_daily_k, *RULE)
    ch = hm.loc["1999":][hm.loc["1999":].diff() != 0]
    print("Gold signal changes (month-end, KRW):", " ".join(f"{d:%Y-%m}{'buy' if v else 'sell'}" for d, v in ch.items()))
    fmt = {k: "{:.1%}".format for k in ("CAGR", "MDD", "Vol", "Worst yr", "Worst 1y roll", "Worst day")}
    fmt.update({"Sharpe": "{:.2f}".format, "Calmar": "{:.2f}".format})
    for label, start, switch in [("LONG 1999~ (BTC from 2018)", "1999-01-04", pb.SWITCH),
                                 ("10Y 2016-11~ (BTC from start)", "2016-11-07", "2016-11-07")]:
        curves = pd.DataFrame({"P1 KRW gold B&H": K.p1_krw(None, start, switch),
                               "P1 KRW gold SMA12 b5% 2M": K.p1_krw(hm, start, switch)})
        print(f"\n===== {label}")
        print(pd.DataFrame({c: C.metrics(curves[c]) for c in curves}).T.to_string(formatters=fmt))
        if start < "2000":
            for a, b in [("1999-01-04", "2002-12-31"), ("2003-01-01", "2009-12-31"), ("2010-01-01", "2017-12-31"),
                         ("2018-01-01", None), ("2016-11-07", None)]:
                row = "  ".join(f"{c}: {pb.stats(curves[c].loc[a:b])['CAGR']:.1%} / {pb.stats(curves[c].loc[a:b])['MDD']:.1%}"
                                for c in curves)
                print(f"  {a[:7]}~{(b or 'now')[:4]}  {row}")
        c2 = curves.loc[curves.index.isin(pb.stock)]
        dd = c2 / c2.cummax() - 1
        for c in curves:
            t = dd[c].idxmin()
            print(f"  MDD {c}: {dd[c].min():.1%}  {c2[c][:t].idxmax().date()} -> {t.date()}")
        yr = pd.concat([c2.iloc[:1], c2.resample("YE").last()]).pct_change().dropna()
        yr.index = yr.index.year
        print((yr * 100).round(1).T.to_string())
