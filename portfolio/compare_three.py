"""Compare P1, P1 with monthly 10-month SMA gold filter, and P2 (long 1999~ and 10y with BTC from start)."""
import pandas as pd
import portfolio_long as pb
import gold_monthly_filter as gm
import gold_regime_switch as rs

g10, _, _ = gm.monthly_filter(10)
idx = pb.sleeves.index
pb.sleeves["G10"] = g10.reindex(idx).ffill() / g10.reindex(idx).ffill().iloc[0]
rs.SL = pb.sleeves
P1 = pb.PORTS["P1 TQQQ40/Gold30/BTC20/SGOV10"]
P2 = pb.PORTS["P2 TQQQ50/BTC30/SGOV20"]
sw = lambda w: {("G10" if k == "GLD" else k): v for k, v in w.items()}
NAMES = {"P1": P1, "P1 gold 10M": tuple(map(sw, P1)), "P2": P2}


def metrics(x):
    x = x.loc[x.index.isin(pb.stock)]
    st = pb.stats(x)
    yr = pd.concat([x.iloc[:1], x.resample("YE").last()]).pct_change().dropna()
    under = x < x.cummax()
    grp = (~under).cumsum()
    longest = max(((s.index[-1] - s.index[0]).days for _, s in x[under].groupby(grp[under])), default=0)
    r = x.pct_change().dropna()
    roll = x.pct_change(252).dropna()
    return {"CAGR": st["CAGR"], "MDD": st["MDD"], "Vol": st["Vol"], "Sharpe": st["Sharpe"], "Calmar": st["Calmar"],
            "Worst yr": yr.min(), "Neg yrs": f"{(yr < 0).sum()}/{len(yr)}", "Longest DD (days)": longest,
            "Worst 1y roll": roll.min(), "Worst day": r.min()}


if __name__ == "__main__":
    fmt = {k: "{:.1%}".format for k in ("CAGR", "MDD", "Vol", "Worst yr", "Worst 1y roll", "Worst day")}
    fmt.update({"Sharpe": "{:.2f}".format, "Calmar": "{:.2f}".format})
    for label, start, switch in [("LONG 1999-01 ~ (BTC from 2018)", "1999-01-04", pb.SWITCH),
                                 ("10Y 2016-11 ~ (BTC from start)", "2016-11-07", "2016-11-07")]:
        curves = pd.DataFrame({n: rs.fixed(*w, start, switch) for n, w in NAMES.items()})
        print(f"\n===== {label}")
        print(pd.DataFrame({c: metrics(curves[c]) for c in curves}).T.to_string(formatters=fmt))
        if start < "2000":
            for a, b in [("1999-01-04", "2002-12-31"), ("2003-01-01", "2009-12-31"),
                         ("2010-01-01", "2017-12-31"), ("2018-01-01", None)]:
                row = {c: f"{pb.stats(curves[c].loc[a:b])['CAGR']:.1%} / {pb.stats(curves[c].loc[a:b])['MDD']:.1%}"
                       for c in curves}
                print(f"  {a[:4]}~{(b or 'now')[:4]}: {row}")
        c2 = curves.loc[curves.index.isin(pb.stock)]
        dd = c2 / c2.cummax() - 1
        for c in curves:
            t = dd[c].idxmin()
            print(f"  MDD {c:12s} {dd[c].min():.1%}  {c2[c][:t].idxmax().date()} -> {t.date()}")
        yr = pd.concat([c2.iloc[:1], c2.resample("YE").last()]).pct_change().dropna()
        yr.index = yr.index.year
        print((yr * 100).round(1).T.to_string())
        wins = (yr["P1 gold 10M"] > yr["P1"]).sum()
        print(f"  P1 gold 10M beats P1 in {wins}/{len(yr)} years; P2 beats P1 in {(yr['P2'] > yr['P1']).sum()}/{len(yr)}")
