"""Candidates for the SGOV 10% slot of P1 (KRW): IEF, IEI, BTAL, vs SGOV and DBC with the 250-day filter.

Base = TQQQ switching 40 / gold held 30 / BTC KRW SMA30 2% 20 / SGOV 10 (BTC weight to TQQQ before 2018).
IEF real from 2002-07, IEI from 2007-01; before that a regression on Vanguard Intermediate Treasury (VFITX).
BTAL exists only from 2011-09 (no proxy) -> compared from 2011-10 only.
  python3 sgov_alt_test.py          actual history
  python3 sgov_alt_test.py long     block bootstrap 1999~ (no BTC, no BTAL), 126 trading-day blocks
  python3 sgov_alt_test.py btc      block bootstrap 2014-09~ (BTC and BTAL included), 182-day blocks
"""
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "actual"
sys.argv = sys.argv[:1]                      # imported modules read sys.argv
import numpy as np
import pandas as pd
import yfinance as yf
import dbc_test as D

pb, K, rs, C = D.pb, D.K, D.rs, D.C
stock = pb.stock
raw = yf.download(["IEF", "IEI", "BTAL", "VFITX"], start="1995-01-01", end=pb.END, auto_adjust=True, progress=False)["Close"]
rr = raw.pct_change(fill_method=None)
FIT = {}


def spliced(t):
    f = rr[[t, "VFITX"]].dropna()
    X = np.column_stack([np.ones(len(f)), f["VFITX"]])
    c, *_ = np.linalg.lstsq(X, f[t].values, rcond=None)
    FIT[t] = (c[0] * 252, c[1], np.corrcoef(f[t], f["VFITX"])[0, 1] ** 2)
    r = rr[t].fillna(c[0] + c[1] * rr["VFITX"])
    return r.reindex(stock).fillna(0.0)


RET = {"IEF": spliced("IEF"), "IEI": spliced("IEI"),
       "BTAL": rr["BTAL"].reindex(stock).where(lambda x: x.notna(), pb.cash_r),       # SGOV before inception
       "DBCF": D.sleeve_usd(D.held_daily(250, 0.02, 1)).pct_change().fillna(0.0)}
MIX = {"IEF": {"IEF": .1}, "IEI": {"IEI": .1}, "BTAL": {"BTAL": .1}, "DBC 250d": {"DBCF": .1},
       "IEF 5 + SGOV 5": {"IEF": .05}, "BTAL 5 + SGOV 5": {"BTAL": .05}, "IEF 5 + BTAL 5": {"IEF": .05, "BTAL": .05},
       "IEF 5 + DBC 5": {"IEF": .05, "DBCF": .05}}
NO_BTAL = [k for k in MIX if "BTAL" not in k]


def wts(mix):
    out = []
    for base in K.P1:
        x = dict(base)
        x["SGOV"] = x["SGOV"] - sum(mix.values())
        x.update(mix)
        out.append({k: v for k, v in x.items() if v > 1e-9})
    return out


def actual():
    sl = D.base_sleeves()
    for k, r in RET.items():
        D.add(sl, k, (1 + r).cumprod())
    rs.SL = sl
    print("proxy fit vs VFITX (alpha/yr, beta, R2):", {k: tuple(round(float(x), 3) for x in v) for k, v in FIT.items()})
    print("\nAlone, USD (CAGR / MDD):")
    for k, r in {"SGOV": pb.cash_r, **RET}.items():
        s = (1 + r).cumprod()
        f = lambda a, b=None: "%6.1f%% / %6.1f%%" % tuple(100 * pb.stats(s.loc[a:b])[x] for x in ("CAGR", "MDD"))
        print(f"  {k:5s} 1999~ {f('1999-01-04')} | 2011-10~ {f('2011-10-03')} | 2018~ {f('2018-01-01')} | 2022 {s.loc['2022'].iloc[-1] / s.loc['2021'].iloc[-1] - 1:+.1%}"
              f" | 2008 {s.loc['2008'].iloc[-1] / s.loc['2007'].iloc[-1] - 1:+.1%}")
    m = sl.loc[sl.index.isin(stock)].resample("ME").last().pct_change().loc["2011-10":]
    print("\nMonthly return correlation in KRW, 2011-10~ (rows: candidates):")
    print(m.corr().loc[["SGOV", "IEF", "IEI", "BTAL", "DBCF"], ["TQQQ_SW", "GLD", "BTC", "SGOV"]].round(2).to_string())
    rows = {}
    for name, mix in {"SGOV (base P1)": {}, **MIX}.items():
        w = wts(mix)
        lg, c11, ty = rs.fixed(*w), rs.fixed(*w, "2011-10-03"), rs.fixed(*w, "2016-11-07", "2016-11-07")
        m, a, b, t = C.metrics(lg), pb.stats(c11), pb.stats(lg.loc["2018-01-01":]), pb.stats(ty)
        x = lg.loc[lg.index.isin(stock)]
        yr = pd.concat([x.iloc[:1], x.resample("YE").last()]).pct_change().dropna()
        yr.index = yr.index.year
        nb = "BTAL" not in name
        rows[name] = {"99~ CAGR": m["CAGR"] if nb else np.nan, "99~ MDD": m["MDD"] if nb else np.nan,
                      "99~ Sh": m["Sharpe"] if nb else np.nan, "2008": yr[2008] if nb else np.nan,
                      "11~ CAGR": a["CAGR"], "11~ MDD": a["MDD"], "11~ Sh": a["Sharpe"],
                      "18~ CAGR": b["CAGR"], "18~ MDD": b["MDD"], "18~ Sh": b["Sharpe"],
                      "10y CAGR": t["CAGR"], "10y MDD": t["MDD"], "10y Sh": t["Sharpe"],
                      "2018": yr[2018], "2022": yr[2022], "2025": yr[2025], "2026": yr[2026]}
    print()
    D.show(rows).to_csv("sgov_alt_grid.csv", float_format="%.4f")


def report(res, base, title):
    b = res[base]
    print(title)
    print(f"{'':18s} CAGR med    5%    MDD med    5%  P(MDD<-30) P(MDD<-40) Sharpe | vs base: CAGR> MDD> Sharpe>  dCAGR med")
    for k, v in res.items():
        print(f"{k:18s} {np.median(v[:,0]):7.1%} {np.percentile(v[:,0],5):6.1%} {np.median(v[:,1]):8.1%} {np.percentile(v[:,1],5):6.1%}"
              f" {(v[:,1]<-.3).mean():9.0%} {(v[:,1]<-.4).mean():9.0%} {np.median(v[:,2]):8.2f} |      {(v[:,0]>b[:,0]).mean():5.0%}"
              f" {(v[:,1]>b[:,1]).mean():5.0%} {(v[:,2]>b[:,2]).mean():5.0%}   {np.median(v[:,0]-b[:,0]):+.2%}")


def combine(cols, wmap, reb, w0):
    out = {}
    for name, w in wmap.items():
        ks = list(w)
        sl = np.column_stack([cols[k] for k in ks])[w0:]
        sl = sl / sl[0]
        wv = np.array([w[k] for k in ks])
        units, v = wv / sl[0], np.empty(len(sl))
        for i in range(len(sl)):
            v[i] = units @ sl[i]
            if reb[w0 + i]:
                units = v[i] * wv / sl[i]
        out[name] = v
    return out


def boot_long():
    import bootstrap_long as B
    from bootstrap import sma, sleeve_tqqq, sleeve_trend
    R = B.R.copy()
    for k in ("IEF", "IEI"):
        R[k] = RET[k].values
    R["DBC"] = D.dbc_r.values
    cols_, RA = list(R.columns), R.values
    names = {"SGOV (base)": {}, **{k: MIX[k] for k in NO_BTAL}}
    wmap = {n: {"TQ": .6, "GOLD": .3, **({"SGOV": .1 - sum(m.values())} if sum(m.values()) < .1 else {}), **m} for n, m in names.items()}
    n = int((B.YEARS + B.WARM) * B.TD)
    rng = np.random.default_rng(7)
    res = {k: [] for k in wmap}
    for _ in range(B.NPATH):
        idx = []
        while len(idx) < n:
            a = rng.integers(0, len(RA) - B.BLOCK)
            idx.extend(range(a, a + B.BLOCK))
        A = RA[np.array(idx[:n])]
        g = {c: A[:, i] for i, c in enumerate(cols_)}
        lvl = lambda r: np.cumprod(1 + r)
        fx, ones, d = lvl(g["FX"]), np.ones(n), lvl(g["DBC"])
        cols = {"TQ": sleeve_tqqq(lvl(g["TQQQ"]), lvl(g["SCHD"]), g["CASHU"], ones) * fx, "GOLD": lvl(g["GOLD"]) * fx,
                "SGOV": lvl(g["CASHU"]) * fx, "IEF": lvl(g["IEF"]) * fx, "IEI": lvl(g["IEI"]) * fx,
                "DBCF": sleeve_trend(g["DBC"], d, sma(d, 250), 0.02, 1, g["CASHU"], 0.001) * fx}
        yrs = pd.bdate_range("2000-01-03", periods=n).year.values
        out = combine(cols, wmap, np.r_[yrs[1:] != yrs[:-1], False], int(B.WARM * B.TD))
        for k in res:
            res[k].append(B.metrics(out[k], B.YEARS))
    report({k: np.array(v) for k, v in res.items()}, "SGOV (base)",
           f"Long bootstrap 1999~: block {B.BLOCK} trading days, {B.NPATH} paths x {B.YEARS}y, KRW, TQQQ 60 / gold 30 / slot 10")


def boot_btc():
    import bootstrap as S
    R = S.R.copy()
    ff = lambda r: (1 + r).cumprod().reindex(S.cal.union(stock)).ffill().reindex(S.cal).pct_change().fillna(0.0).values
    for k in ("IEF", "IEI", "BTAL"):
        R[k] = ff(RET[k])
    R["DBC"] = ff(D.dbc_r)
    cols_ = list(R.columns)
    S.RA = R.values
    names = {"SGOV (base)": {}, **MIX}
    wmap = {n: {"TQ": .4, "GOLD": .3, "BTC": .2, **({"SGOV": .1 - sum(m.values())} if sum(m.values()) < .1 else {}), **m}
            for n, m in names.items()}
    n = int((S.YEARS + S.WARM) * 365)
    rng = np.random.default_rng(42)
    res = {k: [] for k in wmap}
    for _ in range(S.NPATH):
        A = S.sample_path(rng, n)
        g = {c: A[:, i] for i, c in enumerate(cols_)}
        lvl = lambda r: np.cumprod(1 + r)
        fx, bk, d = lvl(g["FX"]), lvl(g["BTCK"]), lvl(g["DBC"])
        st = g["STOCK"] > 0
        ii = np.where(st)[0]
        h = np.ones(n)
        hs = S.trend_pos(d[ii], S.sma(d[ii], 250), 0.02, 1, start_pos=1)
        for a, v in zip(ii, hs):
            h[a:] = v
        prev = np.r_[1.0, h[:-1]]
        dbcf = np.cumprod(1 + g["DBC"] * prev + (1 - prev) * g["CASHU"] - (h != prev) * 0.001)
        cols = {"TQ": S.sleeve_tqqq(lvl(g["TQQQ"]), lvl(g["SCHD"]), g["CASHU"], g["STOCK"]) * fx, "GOLD": lvl(g["GOLD"]) * fx,
                "BTC": S.sleeve_trend(g["BTCK"], bk, S.sma(bk, 30), 0.02, 1, g["CASHK"], 0.0005),
                "SGOV": lvl(g["CASHU"]) * fx, "IEF": lvl(g["IEF"]) * fx, "IEI": lvl(g["IEI"]) * fx, "BTAL": lvl(g["BTAL"]) * fx,
                "DBCF": dbcf * fx}
        days = pd.date_range("2000-01-01", periods=n, freq="D")
        yrs = days.year.values
        reb = np.zeros(n, bool)
        for y in np.unique(yrs):
            j = np.where((yrs == y) & st)[0]
            if len(j):
                reb[j[-1]] = True
        w0 = int(S.WARM * 365)
        out = combine(cols, wmap, reb, w0)
        for k in res:
            res[k].append(S.metrics(out[k], st[w0:]))
    report({k: np.array(v) for k, v in res.items()}, "SGOV (base)",
           f"BTC-era bootstrap 2014-09~: block {S.BLOCK} days, {S.NPATH} paths x {S.YEARS}y, KRW, TQQQ 40 / gold 30 / BTC 20 / slot 10")


if __name__ == "__main__":
    pd.set_option("display.width", 280)
    {"actual": actual, "long": boot_long, "btc": boot_btc}[MODE]()
