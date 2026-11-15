"""TQQQ switching sleeve parked in a Nasdaq-100 inverse ETF (PSQ -1x, QID -2x, SQQQ -3x) instead of SGOV while out.

On the TQQQ sell signal the whole sleeve (TQQQ + SCHD) moves into the parking mix and returns on the buy signal
(extra 0.1% each way). Mixes with SGOV are rebalanced daily. The fixed SGOV 10% of P1 stays SGOV.
Before listing (PSQ/QID 2006-07, SQQQ 2010-02): synthetic -k x QQQ + (1+k) x T-bill - 0.95%/yr, spliced to the real ETFs.
  python3 short_park_test.py         actual history 1999~ (sleeve USD, P1 KRW)
  python3 short_park_test.py boot    long block bootstrap 1999~ (KRW, TQQQ 60 / gold 30 / SGOV 10)
"""
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "actual"
sys.argv = sys.argv[:1]
import numpy as np
import pandas as pd
import yfinance as yf
import schd_exit_test as E

D, pb, K, rs, C = E.D, E.pb, E.K, E.rs, E.C
bl = pb.bl
idx, FEE = E.PX.index, 0.001
raw = yf.download(["PSQ", "QID"], start="2006-01-01", end=pb.END, auto_adjust=True, progress=False)["Close"]
INV = {"PSQ": bl.splice(bl.lev(-1, -2 * bl.rf + bl.ER), raw["PSQ"]), "QID": bl.splice(bl.lev(-2, -3 * bl.rf + bl.ER), raw["QID"]),
       "SQQQ": E.PX["SQQQ"]}
INV = {k: v.reindex(idx).pct_change().fillna(0.0) for k, v in INV.items()}
cash = pb.cash_r.reindex(idx).fillna(0.0)
PARK = {"SGOV (current)": cash}
for k, ws in [("PSQ", (1.0, 0.5, 0.25)), ("QID", (1.0, 0.5, 0.25)), ("SQQQ", (1.0, 0.33, 0.1))]:
    for w in ws:
        PARK[f"{k} {w:.0%}" + ("" if w == 1 else f" + SGOV {1 - w:.0%}")] = w * INV[k] + (1 - w) * cash


def sleeve(k):
    return pd.Series(E.engine(E.T, E.S, PARK[k].values, E.MA, park_fee=0.0 if k.startswith("SGOV") else FEE), index=idx)


def p1(e, start, switch):
    sl = D.base_sleeves()
    s = e.reindex(sl.index).ffill() * K.fx_cal
    sl["TQQQ_SW"] = s / s.iloc[0]
    rs.SL = sl
    return rs.fixed(*K.P1, start, switch)


def actual():
    import dbc_corr as X
    out = X.tq_in == 0
    prev = out.shift().fillna(False).astype(bool)
    print(f"TQQQ out {prev.mean():.0%} of days since 1999. Annualised while out: QQQ {(1 + E.PX['QQQ'].pct_change().fillna(0)[prev]).prod() ** (252 / prev.sum()) - 1:+.1%}, "
          + ", ".join(f"{k} {(1 + PARK[k][prev]).prod() ** (252 / prev.sum()) - 1:+.1%}" for k in ("SGOV (current)", "PSQ 100%", "QID 100%", "SQQQ 100%")))
    grp = (out != out.shift()).cumsum()
    lv = {k: (1 + PARK[k]).cumprod() for k in ("SGOV (current)", "PSQ 100%", "QID 100%", "SQQQ 100%")}
    rows = []
    for _, s in out[out].groupby(grp[out]):
        a, i = s.index[0], idx.get_loc(s.index[-1])
        n = idx[min(i + 1, len(idx) - 1)]
        rows.append({"TQQQ sell": a.date(), "buy": n.date(), "days": (n - a).days, "QQQ": E.PX["QQQ"][n] / E.PX["QQQ"][a] - 1,
                     **{k.split()[0]: v[n] / v[a] - 1 for k, v in lv.items()}})
    ev = pd.DataFrame(rows)
    print(f"{len(ev)} out periods: QQQ rose in {(ev.QQQ > 0).sum()}, PSQ beat SGOV in {(ev.PSQ > ev.SGOV).sum()}; "
          f"median QQQ {ev.QQQ.median():+.1%}, median PSQ {ev.PSQ.median():+.1%}")
    print(ev[ev.days >= 30].to_string(index=False, formatters={c: "{:+.1%}".format for c in ev.columns[3:]}))
    rows = {}
    for k in PARK:
        e = sleeve(k)
        a, b, c, d = (pb.stats(e.loc[s:t]) for s, t in (("1999-01-04", None), ("1999-01-04", "2009-12-31"), ("2010-01-01", None), ("2018-01-01", None)))
        lg, ty = p1(e, "1999-01-04", pb.SWITCH), p1(e, "2016-11-07", "2016-11-07")
        m, t = C.metrics(lg), pb.stats(ty)
        x = lg.loc[lg.index.isin(pb.stock)]
        y = pd.concat([x.iloc[:1], x.resample("YE").last()]).pct_change().dropna()
        y.index = y.index.year
        rows[k] = {"99~ CAGR": a["CAGR"], "99~ MDD": a["MDD"], "99~ Sh": a["Sharpe"], "99-09 CAGR": b["CAGR"], "10~ CAGR": c["CAGR"],
                   "10~ MDD": c["MDD"], "18~ CAGR": d["CAGR"], "18~ MDD": d["MDD"],
                   "P1 CAGR": m["CAGR"], "P1 MDD": m["MDD"], "P1 Sh": m["Sharpe"], "P1 10y CAGR": t["CAGR"], "P1 10y MDD": t["MDD"],
                   "P1 10y Sh": t["Sharpe"], "P1 2002": y[2002], "P1 2008": y[2008], "P1 2022": y[2022], "P1 2023": y[2023]}
    print("\nTQQQ switching sleeve (USD) and P1 (KRW)")
    df = pd.DataFrame(rows).T
    df.to_csv("short_park_grid.csv", float_format="%.4f")
    fm = {c: "{:.1%}".format for c in df.columns if "Sh" not in c}
    fm.update({c: "{:.2f}".format for c in df.columns if "Sh" in c})
    print(df.to_string(formatters=fm))


def boot():
    import bootstrap_long as B
    from bootstrap import sma
    R = B.R.copy()
    R["QQQ"] = E.PX["QQQ"].reindex(B.days).pct_change().fillna(0).values
    for k, v in INV.items():
        R["I_" + k] = v.reindex(B.days).fillna(0).values
    cols, RA = list(R.columns), R.values
    ci = {c: i for i, c in enumerate(cols)}
    n, w0 = int((B.YEARS + B.WARM) * B.TD), int(B.WARM * B.TD)
    rng = np.random.default_rng(7)
    res, sres = {k: [] for k in PARK}, {k: [] for k in PARK}
    for _ in range(B.NPATH):
        ii = []
        while len(ii) < n:
            a = rng.integers(0, len(RA) - B.BLOCK)
            ii.extend(range(a, a + B.BLOCK))
        P = RA[np.array(ii[:n])]
        lvl = lambda c: np.cumprod(1 + P[:, ci[c]])
        t, s, fx, cu = lvl("TQQQ"), lvl("SCHD"), lvl("FX"), P[:, ci["CASHU"]]
        gold, sgov, mt = lvl("GOLD") * fx, np.cumprod(1 + cu) * fx, sma(t, 200)
        yrs = pd.bdate_range("2000-01-03", periods=n).year.values
        reb = np.r_[yrs[1:] != yrs[:-1], False]
        for k in PARK:
            if k.startswith("SGOV"):
                park, fee = cu, 0.0
            else:
                w = 1.0 if "+" not in k else float(k.split()[1].strip("%")) / 100
                park, fee = w * P[:, ci["I_" + k.split()[0]]] + (1 - w) * cu, FEE
            e = E.engine(t, s, park, mt, park_fee=fee)
            sres[k].append(B.metrics(e[w0:] / e[w0], B.YEARS))
            sl = np.column_stack([e * fx, gold, sgov])[w0:]
            sl = sl / sl[0]
            w = np.array([.6, .3, .1])
            units, v = w / sl[0], np.empty(len(sl))
            for i in range(len(sl)):
                v[i] = units @ sl[i]
                if reb[w0 + i]:
                    units = v[i] * w / sl[i]
            res[k].append(B.metrics(v, B.YEARS))
    for lab, r in [("TQQQ switching sleeve alone (USD)", sres), ("P1 form TQQQ 60 / gold 30 / SGOV 10 (KRW)", res)]:
        r = {k: np.array(v) for k, v in r.items()}
        b = r["SGOV (current)"]
        print(f"\n{lab}: block {B.BLOCK} trading days, {B.NPATH} paths x {B.YEARS}y")
        print(f"{'':22s} CAGR med    5%    MDD med    5%  P(MDD<-40) Sharpe | vs SGOV: CAGR> MDD> Sharpe>  dCAGR med   5%~95%")
        for k, v in r.items():
            d = v[:, 0] - b[:, 0]
            print(f"{k:22s} {np.median(v[:,0]):7.1%} {np.percentile(v[:,0],5):6.1%} {np.median(v[:,1]):8.1%} {np.percentile(v[:,1],5):6.1%}"
                  f" {(v[:,1]<-.4).mean():9.0%} {np.median(v[:,2]):8.2f} |      {(d>0).mean():5.0%} {(v[:,1]>b[:,1]).mean():5.0%}"
                  f" {(v[:,2]>b[:,2]).mean():5.0%}   {np.median(d):+.2%}  [{np.percentile(d,5):+.2%}, {np.percentile(d,95):+.2%}]")


if __name__ == "__main__":
    pd.set_option("display.width", 300)
    {"actual": actual, "boot": boot}[MODE]()
