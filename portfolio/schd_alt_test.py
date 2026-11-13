"""Replace SCHD as the profit-taking destination of the TQQQ switching sleeve (all rules unchanged).

Ladder: +20/50/100/200% -> 25/25/50/50% of TQQQ into X; X is sold with TQQQ on the sell signal.
Long test 1999~ only for candidates with history or a proxy:
  SCHD <- VEIPX, IEF <- VFITX regression, TLT <- VUSTX, QLD <- synthetic 2x QQQ, GLD <- gold chain, DBC <- GSCI proxy,
  SPY, QQQ, DIA, XLP, XLU, XLV real (1999~), SGOV = T-bill chain.
All candidates (incl. USMV, SPLV, VIG, VYM, HDV, DVY, RSP, BTAL): engine restarted 2011-10-20 (real SCHD from then).
  python3 schd_alt_test.py        actual history (sleeve USD, P1 KRW)
  python3 schd_alt_test.py boot   long block bootstrap (long-history candidates)
"""
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "actual"
sys.argv = sys.argv[:1]
import numpy as np
import pandas as pd
import yfinance as yf
import schd_exit_test as E
import sgov_alt_test as A

D, pb, K, rs, C = E.D, E.pb, E.K, E.rs, E.C
bl = pb.bl
idx = E.PX.index
T0 = "2011-10-20"
tick = ["SPY", "QQQ", "DIA", "XLP", "XLU", "XLV", "QLD", "TLT", "VUSTX", "USMV", "SPLV", "VIG", "VYM", "HDV", "DVY", "RSP", "BTAL", "VTV"]
raw = yf.download(tick, start="1998-01-01", end=pb.END, auto_adjust=True, progress=False)["Close"]
lv = lambda r: (1 + r.reindex(idx).fillna(0)).cumprod()
LONG = {
    "SCHD (current)": E.PX["SCHD"], "SGOV": lv(pb.cash_r), "IEF": lv(A.RET["IEF"]),
    "TLT": bl.splice(raw["VUSTX"].ffill(), raw["TLT"]).reindex(idx), "GLD": lv(pb.gold_r), "DBC": lv(D.dbc_r),
    "SPY": raw["SPY"].reindex(idx), "DIA": raw["DIA"].reindex(idx), "QQQ": E.PX["QQQ"],
    "QLD": bl.splice(bl.lev(2, bl.rf + bl.ER), raw["QLD"]).reindex(idx),
    "XLP": raw["XLP"].reindex(idx), "XLU": raw["XLU"].reindex(idx), "XLV": raw["XLV"].reindex(idx),
}
SHORT = {k: raw[k].reindex(idx) for k in ("USMV", "SPLV", "VIG", "VYM", "HDV", "DVY", "VTV", "RSP", "BTAL")}
ALL = {**LONG, **SHORT}


def sleeve(level, start=None):
    i = 0 if start is None else idx.get_loc(pd.Timestamp(start))
    e = E.engine(E.T[i:], level.values[i:], E.CASH[i:], E.MA[i:])
    return pd.Series(e, index=idx[i:])


def p1(e, start, switch):
    sl = D.base_sleeves()
    s = (e.reindex(sl.index).ffill() * K.fx_cal).bfill()
    sl["TQQQ_SW"] = s / s.iloc[0]
    rs.SL = sl
    return rs.fixed(*K.P1, start, switch)


def actual():
    rows = {}
    base_m = E.PX["TQQQ"].resample("ME").last().pct_change()
    for k, level in ALL.items():
        r = {}
        mret = level.resample("ME").last().pct_change()
        if k in LONG:
            e = sleeve(level)
            a = pb.stats(e)
            lg = p1(e, "1999-01-04", pb.SWITCH)
            m = C.metrics(lg)
            x = lg.loc[lg.index.isin(pb.stock)]
            yr = pd.concat([x.iloc[:1], x.resample("YE").last()]).pct_change().dropna()
            yr.index = yr.index.year
            r.update({"99~ CAGR": a["CAGR"], "99~ MDD": a["MDD"], "99~ Sh": a["Sharpe"],
                      "P1 CAGR": m["CAGR"], "P1 MDD": m["MDD"], "P1 Sh": m["Sharpe"], "P1 2008": yr[2008], "P1 2022": yr[2022]})
        e2 = sleeve(level, T0)
        b, c = pb.stats(e2), pb.stats(e2.loc["2018-01-01":])
        t = pb.stats(p1(e2, "2016-11-07", "2016-11-07"))
        al = pb.stats(level.loc[T0:])
        r.update({"11~ CAGR": b["CAGR"], "11~ MDD": b["MDD"], "11~ Sh": b["Sharpe"], "18~ CAGR": c["CAGR"], "18~ MDD": c["MDD"],
                  "P1 10y CAGR": t["CAGR"], "P1 10y MDD": t["MDD"], "P1 10y Sh": t["Sharpe"],
                  "X alone 11~": al["CAGR"], "X MDD 11~": al["MDD"], "corr TQQQ": mret.loc[T0:].corr(base_m.loc[T0:])})
        rows[k] = r
    df = pd.DataFrame(rows).T
    df.to_csv("schd_alt_grid.csv", float_format="%.4f")
    fm = {c: "{:.1%}".format for c in df.columns if "Sh" not in c and "corr" not in c}
    fm.update({c: "{:.2f}".format for c in df.columns if "Sh" in c or "corr" in c})
    print(df.to_string(formatters=fm, na_rep="-"))


def boot():
    import bootstrap_long as B
    from bootstrap import sma
    R = B.R.copy()
    for k, level in LONG.items():
        R["X_" + k] = level.reindex(B.days).pct_change().fillna(0).values
    cols, RA = list(R.columns), R.values
    ci = {c: i for i, c in enumerate(cols)}
    n, w0 = int((B.YEARS + B.WARM) * B.TD), int(B.WARM * B.TD)
    rng = np.random.default_rng(7)
    res, sres = {k: [] for k in LONG}, {k: [] for k in LONG}
    for _ in range(B.NPATH):
        ii = []
        while len(ii) < n:
            a = rng.integers(0, len(RA) - B.BLOCK)
            ii.extend(range(a, a + B.BLOCK))
        P = RA[np.array(ii[:n])]
        lvl = lambda c: np.cumprod(1 + P[:, ci[c]])
        t, fx, cu = lvl("TQQQ"), lvl("FX"), P[:, ci["CASHU"]]
        gold, sgov, mt = lvl("GOLD") * fx, np.cumprod(1 + cu) * fx, sma(t, 200)
        yrs = pd.bdate_range("2000-01-03", periods=n).year.values
        reb = np.r_[yrs[1:] != yrs[:-1], False]
        for k in LONG:
            e = E.engine(t, lvl("X_" + k), cu, mt)
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
        b = r["SCHD (current)"]
        print(f"\n{lab}: block {B.BLOCK} trading days, {B.NPATH} paths x {B.YEARS}y")
        print(f"{'':16s} CAGR med    5%    MDD med    5%  P(MDD<-40) Sharpe | vs SCHD: CAGR> MDD> Sharpe>  dCAGR med   5%~95%")
        for k, v in r.items():
            d = v[:, 0] - b[:, 0]
            print(f"{k:16s} {np.median(v[:,0]):7.1%} {np.percentile(v[:,0],5):6.1%} {np.median(v[:,1]):8.1%} {np.percentile(v[:,1],5):6.1%}"
                  f" {(v[:,1]<-.4).mean():9.0%} {np.median(v[:,2]):8.2f} |      {(d>0).mean():5.0%} {(v[:,1]>b[:,1]).mean():5.0%}"
                  f" {(v[:,2]>b[:,2]).mean():5.0%}   {np.median(d):+.2%}  [{np.percentile(d,5):+.2%}, {np.percentile(d,95):+.2%}]")


if __name__ == "__main__":
    pd.set_option("display.width", 300)
    {"actual": actual, "boot": boot}[MODE]()
