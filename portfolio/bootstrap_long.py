"""Long-history block bootstrap (1999-01 ~ 2026-11, KRW), without BTC (no BTC data before 2014).

As in the long backtest before 2018, the BTC weight goes to TQQQ switching:
  A  = TQQQ 60 / gold held 30 / SGOV 10
  E  = TQQQ 60 / gold KRW monthly SMA12 5% band 2-month filter 30 / SGOV 10
  P2 = TQQQ 80 / SGOV 20
US trading-day calendar; blocks of BLOCK trading days drawn jointly (TQQQ, SCHD, gold, FX, US/KRW cash).
TQQQ before 2010 is synthetic (3x QQQ/NDX with costs).
"""
import sys
import numpy as np
import pandas as pd
import portfolio_long as pb
import gold_krw as K
from bootstrap import sma, sleeve_tqqq, gold_filter

BLOCK = int(sys.argv[1]) if len(sys.argv) > 1 else 126
NPATH = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
YEARS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
WARM = 1.5
TD = 252

days = pb.stock
lv = pd.DataFrame({"TQQQ": pb.bl.px["TQQQ"], "SCHD": pb.bl.px["SCHD"],
                   "GOLD": (1 + pb.gold_r).cumprod(), "FX": K.fx_s}).reindex(days).ffill()
R = lv.pct_change().fillna(0.0)
R["CASHU"] = pb.cash_r.reindex(days).fillna(0).values
R["CASHK"] = K.cash_krw_d.reindex(days).fillna(0).values
COLS, RA = list(R.columns), R.values


def portfolios(A, start="2000-01-03"):
    g = {c: A[:, i] for i, c in enumerate(COLS)}
    lvl = lambda r: np.cumprod(1 + r)
    t, s, gold, fx = lvl(g["TQQQ"]), lvl(g["SCHD"]), lvl(g["GOLD"]), lvl(g["FX"])
    ones = np.ones(len(t))
    goldk = gold * fx
    tq = sleeve_tqqq(t, s, g["CASHU"], ones) * fx
    sgov = lvl(g["CASHU"]) * fx
    dts = pd.bdate_range(start, periods=len(t))
    months = (dts.year * 12 + dts.month).values
    gret = np.concatenate([[0.0], goldk[1:] / goldk[:-1] - 1])
    goldE = gold_filter(goldk, gret, g["CASHK"], ones, months)
    years = dts.year.values
    reb = np.r_[years[1:] != years[:-1], False]
    w0 = int(WARM * TD)

    def port(sleeves, w):
        sl = np.column_stack(sleeves)[w0:]
        sl = sl / sl[0]
        w = np.array(w)
        units = w / sl[0]
        out = np.empty(len(sl))
        rb = reb[w0:]
        for i in range(len(sl)):
            v = units @ sl[i]
            out[i] = v
            if rb[i]:
                units = v * w / sl[i]
        return out

    return {"A": port([tq, goldk, sgov], [.6, .3, .1]), "E": port([tq, goldE, sgov], [.6, .3, .1]),
            "P2": port([tq, sgov], [.8, .2])}


def metrics(x, yrs):
    cagr = x[-1] ** (1 / yrs) - 1
    mdd = (x / np.maximum.accumulate(x) - 1).min()
    r = x[1:] / x[:-1] - 1
    return cagr, mdd, r.mean() / r.std() * np.sqrt(TD)


def sample_path(rng, n):
    idx = []
    while len(idx) < n:
        a = rng.integers(0, len(RA) - BLOCK)
        idx.extend(range(a, a + BLOCK))
    return RA[np.array(idx[:n])]


if __name__ == "__main__":
    yrs_act = (len(RA) - int(WARM * TD)) / TD
    act = portfolios(RA, start=days[0])
    print(f"Actual {days[int(WARM*TD)].date()}~ ({yrs_act:.1f}y):",
          {k: tuple(round(float(v), 3) for v in metrics(x, yrs_act)) for k, x in act.items()})
    n = int((YEARS + WARM) * TD)
    rng = np.random.default_rng(7)
    res = {k: [] for k in ("A", "E", "P2")}
    for _ in range(NPATH):
        out = portfolios(sample_path(rng, n))
        for k in res:
            res[k].append(metrics(out[k], YEARS))
    res = {k: np.array(v) for k, v in res.items()}
    q = [5, 25, 50, 75, 95]
    print(f"\nBlock {BLOCK} trading days, {NPATH} paths of {YEARS} years (KRW, no BTC)")
    for name, j in [("CAGR", 0), ("MDD", 1), ("Sharpe", 2)]:
        print(f"{name} {q}")
        for k in res:
            v = np.percentile(res[k][:, j], q)
            print(f"  {k:3s} " + "  ".join((f"{x:7.1%}" if j < 2 else f"{x:6.2f}") for x in v))
    A, E, P = res["A"], res["E"], res["P2"]
    print("Head-to-head:")
    print(f"  E vs A : CAGR {(E[:,0]>A[:,0]).mean():.0%}  MDD {(E[:,1]>A[:,1]).mean():.0%}  Sharpe {(E[:,2]>A[:,2]).mean():.0%}")
    print(f"  P2 vs A: CAGR {(P[:,0]>A[:,0]).mean():.0%}  MDD {(P[:,1]>A[:,1]).mean():.0%}  Sharpe {(P[:,2]>A[:,2]).mean():.0%}")
    print(f"  P2 vs E: CAGR {(P[:,0]>E[:,0]).mean():.0%}  MDD {(P[:,1]>E[:,1]).mean():.0%}  Sharpe {(P[:,2]>E[:,2]).mean():.0%}")
    print("Tail risk:")
    for k in res:
        v = res[k]
        print(f"  {k:3s} P(CAGR<0) {(v[:,0]<0).mean():.1%}  P(CAGR<10%) {(v[:,0]<.10).mean():.1%}  "
              f"P(MDD<-30%) {(v[:,1]<-.30).mean():.1%}  P(MDD<-40%) {(v[:,1]<-.40).mean():.1%}  P(MDD<-50%) {(v[:,1]<-.50).mean():.1%}")
    d = E[:, 0] - A[:, 0]
    print(f"E-A CAGR diff: median {np.median(d):+.2%}, 5~95% [{np.percentile(d,5):+.2%}, {np.percentile(d,95):+.2%}]")
