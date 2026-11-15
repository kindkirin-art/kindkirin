"""Managed futures (DBMF + KMLM) in the SGOV 10% slot of P1 (KRW).

DBMF real from 2019-05-08, KMLM from 2020-12-02. Before that a proxy from a basket of managed-futures mutual funds
(AQMIX, ASFYX, MFTFX, RYMFX; equal weight, from 2010-08): r = cash + beta * (basket - cash), beta fitted weekly on the
overlap, alpha set to 0 (the fitted alpha is printed). No usable data before 2010-08 -> tests start 2011.
  python3 mf_test.py         actual history
  python3 mf_test.py boot    BTC-era block bootstrap (2014-09~, 182-day blocks)
"""
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "actual"
sys.argv = sys.argv[:1]
import numpy as np
import pandas as pd
import yfinance as yf
import sgov_alt_test as A

D, pb, K, rs, C = A.D, A.pb, A.K, A.rs, A.C
stock = pb.stock
FUNDS = ["AQMIX", "ASFYX", "MFTFX", "RYMFX"]
raw = yf.download(["DBMF", "KMLM"] + FUNDS, start="2007-01-01", end=pb.END, auto_adjust=True, progress=False)["Close"].reindex(stock)
P0 = "2010-08-04"
basket = raw[FUNDS].pct_change(fill_method=None).loc[P0:].mean(axis=1)
cash = pb.cash_r
FIT = {}


def spliced(t):
    ex = (basket - cash).dropna()
    y = (raw[t].pct_change(fill_method=None) - cash).dropna()
    w = pd.concat([(1 + y).resample("W").prod() - 1, (1 + ex).resample("W").prod() - 1], axis=1, keys=["y", "x"]).loc[y.index[0]:].dropna()
    X = np.column_stack([np.ones(len(w)), w["x"]])
    c, *_ = np.linalg.lstsq(X, w["y"].values, rcond=None)
    FIT[t] = (c[0] * 52, c[1], np.corrcoef(w["y"], w["x"])[0, 1])
    proxy = cash + c[1] * (basket - cash)
    return raw[t].pct_change(fill_method=None).fillna(proxy).fillna(cash).reindex(stock).fillna(0.0)


RET = {"DBMF": spliced("DBMF"), "KMLM": spliced("KMLM"), "DBCF": A.RET["DBCF"], "IEF": A.RET["IEF"]}
MIX = {"DBMF 5 + KMLM 5": {"DBMF": .05, "KMLM": .05}, "DBMF 10": {"DBMF": .1}, "KMLM 10": {"KMLM": .1},
       "DBMF 2.5 + KMLM 2.5 + SGOV 5": {"DBMF": .025, "KMLM": .025}, "DBC 250d 10": {"DBCF": .1},
       "DBMF 5 + DBC 250d 5": {"DBMF": .05, "DBCF": .05}}


def actual():
    print("proxy fit vs fund basket, weekly (alpha/yr, beta, corr):", {k: tuple(round(float(x), 3) for x in v) for k, v in FIT.items()})
    lvl = {k: (1 + r).cumprod() for k, r in {"SGOV": cash, **RET}.items()}
    print("\nAlone, USD (CAGR / MDD / vol):")
    for k in ("SGOV", "DBMF", "KMLM"):
        f = lambda a: "%5.1f%% / %6.1f%% / %4.1f%%" % tuple(100 * pb.stats(lvl[k].loc[a:])[x] for x in ("CAGR", "MDD", "Vol"))
        print(f"  {k:5s} 2011~ (proxy) {f('2011-01-03')} | 2018~ {f('2018-01-01')} | 2020-12~ (real) {f('2020-12-02')}")
    yr = pd.DataFrame({k: v.resample("YE").last().pct_change() for k, v in lvl.items() if k in ("SGOV", "DBMF", "KMLM")}).loc["2011":]
    yr.index = yr.index.year
    print((yr * 100).round(1).T.to_string())
    sl = D.base_sleeves()
    for k, r in RET.items():
        D.add(sl, k, (1 + r).cumprod())
    rs.SL = sl
    m = sl.loc[sl.index.isin(stock)].resample("ME").last().pct_change()
    for lab, a in [("2011~", "2011-01"), ("2020-12~ real", "2020-12")]:
        print(f"\nMonthly return correlation in KRW, {lab}:")
        print(m.loc[a:].corr().loc[["SGOV", "DBMF", "KMLM"], ["TQQQ_SW", "GLD", "BTC", "SGOV", "DBMF", "KMLM"]].round(2).to_string())
    import dbc_corr as X
    t, b = X.tq_in == 0, X.btc_in == 0
    for lab, out, a in [("TQQQ & BTC out, 2015~", t & b, "2015-01-02"), ("TQQQ out, 2011~", t, "2011-01-03")]:
        prev = out.shift().fillna(False).astype(bool).loc[a:]
        print(f"  {lab} ({prev.mean():.0%} of days): " + "  ".join(
            f"{k} {(1 + RET[k].loc[a:][prev]).prod() ** (252 / prev.sum()) - 1:+.1%}/yr (else {(1 + RET[k].loc[a:][~prev]).prod() ** (252 / (~prev).sum()) - 1:+.1%})"
            for k in ("DBMF", "KMLM")))
    rows = {}
    for name, mix in {"SGOV (base P1)": {}, **MIX}.items():
        w = A.wts(mix)
        c11, c18, ty, re = (rs.fixed(*w, s, s) for s in ("2011-01-03", "2018-01-02", "2016-11-07", "2020-12-02"))
        a, b, t, r = (pb.stats(x) for x in (c11, c18, ty, re))
        x = c11.loc[c11.index.isin(stock)]
        y = pd.concat([x.iloc[:1], x.resample("YE").last()]).pct_change().dropna()
        y.index = y.index.year
        rows[name] = {"11~ CAGR": a["CAGR"], "11~ MDD": a["MDD"], "11~ Sh": a["Sharpe"], "18~ CAGR": b["CAGR"], "18~ MDD": b["MDD"],
                      "18~ Sh": b["Sharpe"], "10y CAGR": t["CAGR"], "10y MDD": t["MDD"], "10y Sh": t["Sharpe"],
                      "20-12~ CAGR": r["CAGR"], "20-12~ MDD": r["MDD"], "20-12~ Sh": r["Sharpe"],
                      "2018": y[2018], "2022": y[2022], "2023": y[2023], "2025": y[2025], "2026": y[2026]}
    print()
    D.show(rows).to_csv("mf_grid.csv", float_format="%.4f")


def boot():
    import bootstrap as S
    R = S.R.copy()
    ff = lambda r: (1 + r).cumprod().reindex(S.cal.union(stock)).ffill().reindex(S.cal).pct_change().fillna(0.0).values
    for k in ("DBMF", "KMLM"):
        R[k] = ff(RET[k])
    R["DBC"] = ff(D.dbc_r)
    cols_ = list(R.columns)
    S.RA = R.values
    wmap = {n: {"TQ": .4, "GOLD": .3, "BTC": .2, **({"SGOV": .1 - sum(m.values())} if sum(m.values()) < .1 - 1e-9 else {}), **m}
            for n, m in {"SGOV (base)": {}, **MIX}.items()}
    n = int((S.YEARS + S.WARM) * 365)
    rng = np.random.default_rng(42)
    res = {k: [] for k in wmap}
    for _ in range(S.NPATH):
        P = S.sample_path(rng, n)
        g = {c: P[:, i] for i, c in enumerate(cols_)}
        lvl = lambda r: np.cumprod(1 + r)
        fx, bk, d = lvl(g["FX"]), lvl(g["BTCK"]), lvl(g["DBC"])
        st = g["STOCK"] > 0
        ii = np.where(st)[0]
        h = np.ones(n)
        for a, v in zip(ii, S.trend_pos(d[ii], S.sma(d[ii], 250), 0.02, 1, start_pos=1)):
            h[a:] = v
        prev = np.r_[1.0, h[:-1]]
        cols = {"TQ": S.sleeve_tqqq(lvl(g["TQQQ"]), lvl(g["SCHD"]), g["CASHU"], g["STOCK"]) * fx, "GOLD": lvl(g["GOLD"]) * fx,
                "BTC": S.sleeve_trend(g["BTCK"], bk, S.sma(bk, 30), 0.02, 1, g["CASHK"], 0.0005), "SGOV": lvl(g["CASHU"]) * fx,
                "DBMF": lvl(g["DBMF"]) * fx, "KMLM": lvl(g["KMLM"]) * fx,
                "DBCF": np.cumprod(1 + g["DBC"] * prev + (1 - prev) * g["CASHU"] - (h != prev) * 0.001) * fx}
        yrs = pd.date_range("2000-01-01", periods=n, freq="D").year.values
        reb = np.zeros(n, bool)
        for y in np.unique(yrs):
            j = np.where((yrs == y) & st)[0]
            if len(j):
                reb[j[-1]] = True
        w0 = int(S.WARM * 365)
        out = A.combine(cols, wmap, reb, w0)
        for k in res:
            res[k].append(S.metrics(out[k], st[w0:]))
    A.report({k: np.array(v) for k, v in res.items()}, "SGOV (base)",
             f"BTC-era bootstrap 2014-09~: block {S.BLOCK} days, {S.NPATH} paths x {S.YEARS}y, KRW, TQQQ 40 / gold 30 / BTC 20 / slot 10")


if __name__ == "__main__":
    pd.set_option("display.width", 280)
    {"actual": actual, "boot": boot}[MODE]()
