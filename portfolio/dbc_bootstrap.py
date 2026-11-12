"""Long block bootstrap (1999~, KRW, no BTC) of P1 with a DBC sleeve. See dbc_test.py / bootstrap_long.py.

Base = TQQQ switching 60 / gold held 30 / SGOV 10 (the pre-2018 form of P1).
DBC sleeve: held, or daily SMA trend filter on the USD price (SGOV when out, fee 0.1%).
"""
import numpy as np
import pandas as pd
import bootstrap_long as B
import dbc_test as D
from bootstrap import sma, sleeve_tqqq, sleeve_trend

R = B.R.copy()
R["DBC"] = D.dbc_r.reindex(B.days).fillna(0).values
COLS, RA = list(R.columns), R.values
FILTERS = {"F150": (150, 0.02, 3), "F200": (200, 0.02, 1), "F250": (250, 0.02, 1)}
# name: (TQQQ, gold, SGOV, DBC weight, DBC sleeve)
PORTS = {"base": (.6, .3, .1, 0, "H")}
for f in ("H", "F150", "F200", "F250"):
    PORTS.update({f"{f} 5 S": (.6, .3, .05, .05, f), f"{f} 10 S": (.6, .3, 0, .1, f), f"{f} 10 G": (.6, .2, .1, .1, f),
                  f"{f} 10 G+S": (.6, .25, .05, .1, f), f"{f} 20 G+S": (.6, .2, 0, .2, f), f"{f} 10 T": (.5, .3, .1, .1, f)})


def portfolios(A):
    g = {c: A[:, i] for i, c in enumerate(COLS)}
    lvl = lambda r: np.cumprod(1 + r)
    fx = lvl(g["FX"])
    ones = np.ones(len(fx))
    tq = sleeve_tqqq(lvl(g["TQQQ"]), lvl(g["SCHD"]), g["CASHU"], ones) * fx
    gold, sgov, d = lvl(g["GOLD"]) * fx, lvl(g["CASHU"]) * fx, lvl(g["DBC"])
    dbc = {"H": d * fx}
    for k, (n, b, c) in FILTERS.items():
        dbc[k] = sleeve_trend(g["DBC"], d, sma(d, n), b, c, g["CASHU"], 0.001) * fx
    yrs = pd.bdate_range("2000-01-03", periods=len(fx)).year.values
    reb = np.r_[yrs[1:] != yrs[:-1], False]
    w0 = int(B.WARM * B.TD)
    out = {}
    for name, (a, b, c, e, f) in PORTS.items():
        sl = np.column_stack([tq, gold, sgov, dbc[f]])[w0:]
        sl = sl / sl[0]
        w = np.array([a, b, c, e])
        units, v = w / sl[0], np.empty(len(sl))
        for i in range(len(sl)):
            v[i] = units @ sl[i]
            if reb[w0 + i]:
                units = v[i] * w / sl[i]
        out[name] = v
    return out


if __name__ == "__main__":
    n = int((B.YEARS + B.WARM) * B.TD)
    rng = np.random.default_rng(7)
    res = {k: [] for k in PORTS}
    for _ in range(B.NPATH):
        idx = []
        while len(idx) < n:
            a = rng.integers(0, len(RA) - B.BLOCK)
            idx.extend(range(a, a + B.BLOCK))
        out = portfolios(RA[np.array(idx[:n])])
        for k in res:
            res[k].append(B.metrics(out[k], B.YEARS))
    res = {k: np.array(v) for k, v in res.items()}
    b = res["base"]
    print(f"Block {B.BLOCK} trading days, {B.NPATH} paths of {B.YEARS} years (KRW, no BTC)")
    print(f"{'':14s} CAGR med    5%    MDD med   5%   P(MDD<-40) Sharpe | vs base: CAGR> MDD> Sharpe>  dCAGR med")
    for k, v in res.items():
        print(f"{k:14s} {np.median(v[:,0]):7.1%} {np.percentile(v[:,0],5):6.1%} {np.median(v[:,1]):8.1%} {np.percentile(v[:,1],5):6.1%}"
              f" {(v[:,1]<-.4).mean():8.0%} {np.median(v[:,2]):8.2f} |      {(v[:,0]>b[:,0]).mean():5.0%} {(v[:,1]>b[:,1]).mean():5.0%}"
              f" {(v[:,2]>b[:,2]).mean():5.0%}   {np.median(v[:,0]-b[:,0]):+.2%}")
