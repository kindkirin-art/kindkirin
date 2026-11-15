"""Where the TQQQ switching sleeve waits while it is out: SGOV (current) vs DBMF and others (rules unchanged).

On the TQQQ sell signal the whole sleeve (TQQQ + SCHD) goes into the parking ETF instead of SGOV and comes back
on the buy signal (extra 0.1% each way). The fixed SGOV 10% of P1 stays SGOV.
DBMF/KMLM before listing: mutual-fund proxy from 2010-08 (mf_test.py) -> engine restarted 2011-01-03.
  python3 park_test.py         actual history (sleeve USD, P1 KRW)
  python3 park_test.py boot    BTC-era block bootstrap (2014-09~)
"""
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "actual"
sys.argv = sys.argv[:1]
import numpy as np
import pandas as pd
import mf_test as M
import schd_exit_test as E

A, D, pb, K, rs = M.A, M.D, M.pb, M.K, M.rs
idx, FEE, T0 = E.PX.index, 0.001, "2011-01-03"
al = lambda r: r.reindex(idx).fillna(0.0)
PARK = {"SGOV (current)": al(pb.cash_r), "DBMF": al(M.RET["DBMF"]), "KMLM": al(M.RET["KMLM"]),
        "DBMF 50 + KMLM 50": al((M.RET["DBMF"] + M.RET["KMLM"]) / 2), "DBMF 50 + SGOV 50": al((M.RET["DBMF"] + pb.cash_r) / 2),
        "IEF": al(A.RET["IEF"]), "GLD": al(pb.gold_r), "DBC 250d": al(A.RET["DBCF"]), "BTAL": al(A.RET["BTAL"])}


def sleeve(park, start=T0):
    i = idx.get_loc(pd.Timestamp(start))
    fee = 0.0 if park is PARK["SGOV (current)"] else FEE
    return pd.Series(E.engine(E.T[i:], E.S[i:], park.values[i:], E.MA[i:], park_fee=fee), index=idx[i:])


def p1(e, start):
    sl = D.base_sleeves()
    s = (e.reindex(sl.index).ffill() * K.fx_cal).bfill()
    sl["TQQQ_SW"] = s / s.iloc[0]
    rs.SL = sl
    return rs.fixed(*K.P1, start, start)


def actual():
    import dbc_corr as X
    out = (X.tq_in == 0).loc[T0:]
    grp = (out != out.shift()).cumsum()
    lv = {k: (1 + v).cumprod() for k, v in PARK.items()}
    rows = []
    for _, s in out[out].groupby(grp[out]):
        a, i = s.index[0], idx.get_loc(s.index[-1])
        n = idx[min(i + 1, len(idx) - 1)]
        rows.append({"TQQQ sell": a.date(), "buy": n.date(), "days": (n - a).days,
                     **{k: lv[k][n] / lv[k][a] - 1 for k in ("SGOV (current)", "DBMF", "KMLM", "IEF", "GLD")},
                     "QQQ": E.PX["QQQ"][n] / E.PX["QQQ"][a] - 1})
    ev = pd.DataFrame(rows)
    f = {c: "{:+.1%}".format for c in ev.columns[3:]}
    print(f"TQQQ-out periods since 2011 ({out.mean():.0%} of days); return of each parking asset while out")
    print(ev.to_string(index=False, formatters=f))
    print("  DBMF beat SGOV in %d of %d periods; sum of DBMF - SGOV: %+.1f%%p; periods >= 60 days: %s" % (
        (ev["DBMF"] > ev["SGOV (current)"]).sum(), len(ev), 100 * (ev["DBMF"] - ev["SGOV (current)"]).sum(),
        ", ".join(f"{d}: {x:+.1%}" for d, x in zip(ev[ev.days >= 60]["TQQQ sell"], (ev["DBMF"] - ev["SGOV (current)"])[ev.days >= 60]))))
    prev = out.shift().fillna(False).astype(bool)
    print("  annualised while TQQQ is out: " + ", ".join(f"{k} {(1 + v.loc[T0:][prev]).prod() ** (252 / prev.sum()) - 1:+.1%}" for k, v in PARK.items()))
    rows = {}
    for k, park in PARK.items():
        e = sleeve(park)
        a, b, c = (pb.stats(e.loc[s:]) for s in (T0, "2018-01-01", "2020-12-02"))
        pa, pt, pc = (pb.stats(p1(e, s)) for s in (T0, "2016-11-07", "2020-12-02"))
        x = p1(e, T0)
        x = x.loc[x.index.isin(pb.stock)]
        y = pd.concat([x.iloc[:1], x.resample("YE").last()]).pct_change().dropna()
        y.index = y.index.year
        rows[k] = {"11~ CAGR": a["CAGR"], "11~ MDD": a["MDD"], "11~ Sh": a["Sharpe"], "18~ CAGR": b["CAGR"], "18~ MDD": b["MDD"],
                   "20-12~ CAGR": c["CAGR"], "20-12~ MDD": c["MDD"],
                   "P1 11~ CAGR": pa["CAGR"], "P1 11~ MDD": pa["MDD"], "P1 11~ Sh": pa["Sharpe"],
                   "P1 10y CAGR": pt["CAGR"], "P1 10y MDD": pt["MDD"], "P1 10y Sh": pt["Sharpe"],
                   "P1 20-12~ CAGR": pc["CAGR"], "P1 20-12~ MDD": pc["MDD"], "P1 2018": y[2018], "P1 2022": y[2022], "P1 2025": y[2025]}
    print("\nTQQQ switching sleeve (USD) and P1 (KRW)")
    D.show(rows).to_csv("park_grid.csv", float_format="%.4f")


def boot():
    import bootstrap as S
    R = S.R.copy()
    ff = lambda r: (1 + r).cumprod().reindex(S.cal.union(pb.stock)).ffill().reindex(S.cal).pct_change().fillna(0.0).values
    names = [k for k in PARK if k not in ("SGOV (current)", "GLD")]
    for k in names:
        R["P_" + k] = ff(PARK[k])
    cols_ = list(R.columns)
    S.RA = R.values
    n, w0 = int((S.YEARS + S.WARM) * 365), int(S.WARM * 365)
    rng = np.random.default_rng(42)
    res = {k: [] for k in ["SGOV (current)"] + names}
    sres = {k: [] for k in res}
    for _ in range(S.NPATH):
        P = S.sample_path(rng, n)
        g = {c: P[:, i] for i, c in enumerate(cols_)}
        lvl = lambda r: np.cumprod(1 + r)
        fx, bk = lvl(g["FX"]), lvl(g["BTCK"])
        st = g["STOCK"] > 0
        ii = np.where(st)[0]
        t, s = lvl(g["TQQQ"])[ii], lvl(g["SCHD"])[ii]
        mt = S.sma(t, 200)
        gold, sgov = lvl(g["GOLD"]) * fx, lvl(g["CASHU"]) * fx
        btc = S.sleeve_trend(g["BTCK"], bk, S.sma(bk, 30), 0.02, 1, g["CASHK"], 0.0005)
        yrs = pd.date_range("2000-01-01", periods=n, freq="D").year.values
        reb = np.zeros(n, bool)
        for y in np.unique(yrs):
            j = np.where((yrs == y) & st)[0]
            if len(j):
                reb[j[-1]] = True
        for k in res:
            park = g["CASHU"] if k == "SGOV (current)" else g["P_" + k]
            cp = np.add.reduceat(np.log1p(park), np.r_[0, ii[:-1] + 1])            # returns accrued since the previous trading day
            e = E.engine(t, s, np.expm1(cp), mt, park_fee=0.0 if k == "SGOV (current)" else FEE)
            full = np.ones(n)
            for a, v in zip(ii, e):
                full[a:] = v
            sres[k].append(S.metrics(full[w0:] / full[w0], st[w0:]))
            out = A.combine({"TQ": full * fx, "GOLD": gold, "BTC": btc, "SGOV": sgov}, {"x": {"TQ": .4, "GOLD": .3, "BTC": .2, "SGOV": .1}}, reb, w0)
            res[k].append(S.metrics(out["x"], st[w0:]))
    for lab, r in [("TQQQ switching sleeve alone (USD)", sres), ("P1 TQQQ 40 / gold 30 / BTC 20 / SGOV 10 (KRW)", res)]:
        A.report({k: np.array(v) for k, v in r.items()}, "SGOV (current)",
                 f"\n{lab}: BTC-era bootstrap 2014-09~, block {S.BLOCK} days, {S.NPATH} paths x {S.YEARS}y")


if __name__ == "__main__":
    pd.set_option("display.width", 300)
    {"actual": actual, "boot": boot}[MODE]()
