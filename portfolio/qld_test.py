"""QLD (2x Nasdaq-100) instead of TQQQ in the switching sleeve, signalled by QLD's own SMA or by the Nasdaq Composite (^IXIC) SMA.

Everything else as now: buy after 3 closes above SMA*(1+band), sell after 2 closes below SMA*(1-band), profit ladder
+20/50/100/200% (on the traded ETF's entry price) -> 25/25/50/50% into SCHD, SGOV when out, fee 0.1%.
QLD before 2006-06: synthetic 2 x QQQ - T-bill - 0.95%/yr.
  python3 qld_test.py         actual history 1999~ (sleeve USD, P1 KRW)
  python3 qld_test.py boot    long block bootstrap 1999~ (KRW, sleeve 60 / gold 30 / SGOV 10)
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
idx = E.PX.index
raw = yf.download(["QLD", "^IXIC"], start="1985-01-01", end=pb.END, auto_adjust=True, progress=False)["Close"]
FULL = {"TQQQ": bl.splice(bl.tqqq_syn, bl.raw["TQQQ"]), "QLD": bl.splice(bl.lev(2, bl.rf + bl.ER), raw["QLD"]),
        "IXIC": raw["^IXIC"].reindex(bl.raw.index).ffill(), "QQQ": (1 + bl.r).cumprod()}
LV = {k: v.reindex(idx).values for k, v in FULL.items()}
ma = lambda k, n: FULL[k].rolling(n).mean().reindex(idx).values

VARS = [("TQQQ", "TQQQ", 200, 0.01)]
VARS += [("QLD", "QLD", n, b) for n in (100, 150, 200, 250) for b in (0.0, 0.01, 0.02)]
VARS += [("QLD", "IXIC", n, b) for n in (100, 150, 200, 250) for b in (0.0, 0.005, 0.01, 0.02)]
VARS += [("TQQQ", "IXIC", 200, 0.005), ("TQQQ", "IXIC", 200, 0.01), ("TQQQ", "QLD", 200, 0.01), ("QLD", "TQQQ", 200, 0.01)]
name = lambda v: f"{v[0]} / {v[1]} {v[2]}d {v[3]:.1%}"


def sleeve(v, info=None):
    tr, sg, n, b = v
    return pd.Series(E.engine(LV[tr], E.S, E.CASH, ma(sg, n), sig=LV[sg], band=b, info=info), index=idx)


def trades(v):
    s, m, b = LV[v[1]], ma(v[1], v[2]), v[3]
    pos = up = dn = n = 0
    held = np.zeros(len(s))
    for i in range(len(s)):
        up = up + 1 if s[i] > m[i] * (1 + b) else 0
        dn = dn + 1 if s[i] < m[i] * (1 - b) else 0
        if not pos and up >= 3:
            pos, n = 1, n + 1
        elif pos and dn >= 2:
            pos = 0
        held[i] = pos
    return n, held.mean()


def p1(e, start, switch):
    sl = D.base_sleeves()
    s = e.reindex(sl.index).ffill() * K.fx_cal
    sl["TQQQ_SW"] = s / s.iloc[0]
    rs.SL = sl
    return rs.fixed(*K.P1, start, switch)


def actual():
    print("Buy & hold 1999~ (USD): " + ", ".join(
        f"{k} {pb.stats(pd.Series(LV[k], index=idx))['CAGR']:.1%} / {pb.stats(pd.Series(LV[k], index=idx))['MDD']:.1%}" for k in ("QQQ", "QLD", "TQQQ")))
    rows = {}
    for v in VARS:
        e = sleeve(v)
        a, b, c, d = (pb.stats(e.loc[s:t]) for s, t in (("1999-01-04", None), ("1999-01-04", "2009-12-31"), ("2010-01-01", None), ("2018-01-01", None)))
        lg, ty = p1(e, "1999-01-04", pb.SWITCH), p1(e, "2016-11-07", "2016-11-07")
        m, t = C.metrics(lg), pb.stats(ty)
        x = lg.loc[lg.index.isin(pb.stock)]
        y = pd.concat([x.iloc[:1], x.resample("YE").last()]).pct_change().dropna()
        y.index = y.index.year
        n, inm = trades(v)
        rows[name(v)] = {"buys": n, "in%": inm, "99~ CAGR": a["CAGR"], "99~ MDD": a["MDD"], "99~ Sh": a["Sharpe"], "99-09 CAGR": b["CAGR"],
                         "10~ CAGR": c["CAGR"], "10~ MDD": c["MDD"], "18~ CAGR": d["CAGR"], "18~ MDD": d["MDD"],
                         "P1 CAGR": m["CAGR"], "P1 MDD": m["MDD"], "P1 Sh": m["Sharpe"], "P1 worst 1y": m["Worst 1y roll"],
                         "P1 10y CAGR": t["CAGR"], "P1 10y MDD": t["MDD"], "P1 10y Sh": t["Sharpe"], "P1 2022": y[2022], "P1 2026": y[2026]}
    df = pd.DataFrame(rows).T
    df.to_csv("qld_grid.csv", float_format="%.4f")
    fm = {c: "{:.1%}".format for c in df.columns if "Sh" not in c}
    fm.update({c: "{:.2f}".format for c in df.columns if "Sh" in c})
    fm["buys"] = "{:.0f}".format
    print("traded / signal SMA band   (sleeve USD, P1 KRW)")
    print(df.to_string(formatters=fm))


def boot():
    import bootstrap_long as B
    from bootstrap import sma
    sel = [("TQQQ", "TQQQ", 200, 0.01), ("QLD", "QLD", 200, 0.01), ("QLD", "QLD", 200, 0.0), ("QLD", "QLD", 150, 0.01), ("QLD", "QLD", 250, 0.01),
           ("QLD", "IXIC", 200, 0.01), ("QLD", "IXIC", 200, 0.005), ("QLD", "IXIC", 200, 0.0), ("QLD", "IXIC", 150, 0.005),
           ("QLD", "IXIC", 250, 0.005), ("QLD", "IXIC", 250, 0.0), ("TQQQ", "IXIC", 200, 0.01), ("QLD", "TQQQ", 200, 0.01)]
    R = B.R.copy()
    for k in ("QLD", "IXIC"):
        R[k] = FULL[k].reindex(B.days).pct_change().fillna(0).values
    cols, RA = list(R.columns), R.values
    ci = {c: i for i, c in enumerate(cols)}
    n, w0 = int((B.YEARS + B.WARM) * B.TD), int(B.WARM * B.TD)
    rng = np.random.default_rng(7)
    res, sres = {name(v): [] for v in sel}, {name(v): [] for v in sel}
    for _ in range(B.NPATH):
        ii = []
        while len(ii) < n:
            a = rng.integers(0, len(RA) - B.BLOCK)
            ii.extend(range(a, a + B.BLOCK))
        P = RA[np.array(ii[:n])]
        lvl = lambda c: np.cumprod(1 + P[:, ci[c]])
        L = {"TQQQ": lvl("TQQQ"), "QLD": lvl("QLD"), "IXIC": lvl("IXIC")}
        s, fx, cu = lvl("SCHD"), lvl("FX"), P[:, ci["CASHU"]]
        gold, sgov = lvl("GOLD") * fx, np.cumprod(1 + cu) * fx
        yrs = pd.bdate_range("2000-01-03", periods=n).year.values
        reb = np.r_[yrs[1:] != yrs[:-1], False]
        mas = {}
        for v in sel:
            key = (v[1], v[2])
            if key not in mas:
                mas[key] = sma(L[v[1]], v[2])
            e = E.engine(L[v[0]], s, cu, mas[key], sig=L[v[1]], band=v[3])
            sres[name(v)].append(B.metrics(e[w0:] / e[w0], B.YEARS))
            sl = np.column_stack([e * fx, gold, sgov])[w0:]
            sl = sl / sl[0]
            w = np.array([.6, .3, .1])
            units, val = w / sl[0], np.empty(len(sl))
            for i in range(len(sl)):
                val[i] = units @ sl[i]
                if reb[w0 + i]:
                    units = val[i] * w / sl[i]
            res[name(v)].append(B.metrics(val, B.YEARS))
    for lab, r in [("Switching sleeve alone (USD)", sres), ("P1 form sleeve 60 / gold 30 / SGOV 10 (KRW)", res)]:
        r = {k: np.array(v) for k, v in r.items()}
        b = r[name(sel[0])]
        print(f"\n{lab}: block {B.BLOCK} trading days, {B.NPATH} paths x {B.YEARS}y")
        print(f"{'':26s} CAGR med    5%    MDD med    5%  P(MDD<-40) Sharpe | vs current: CAGR> MDD> Sharpe>  dCAGR med   5%~95%")
        for k, v in r.items():
            d = v[:, 0] - b[:, 0]
            print(f"{k:26s} {np.median(v[:,0]):7.1%} {np.percentile(v[:,0],5):6.1%} {np.median(v[:,1]):8.1%} {np.percentile(v[:,1],5):6.1%}"
                  f" {(v[:,1]<-.4).mean():9.0%} {np.median(v[:,2]):8.2f} |         {(d>0).mean():5.0%} {(v[:,1]>b[:,1]).mean():5.0%}"
                  f" {(v[:,2]>b[:,2]).mean():5.0%}   {np.median(d):+.2%}  [{np.percentile(d,5):+.2%}, {np.percentile(d,95):+.2%}]")


if __name__ == "__main__":
    pd.set_option("display.width", 320)
    {"actual": actual, "boot": boot}[MODE]()
