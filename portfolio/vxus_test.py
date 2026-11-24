"""VXUS (total international ex-US stocks) instead of SCHD as the profit-taking destination of the TQQQ switching sleeve.

Rules unchanged: +20/50/100/200% -> 25/25/50/50% of TQQQ into X, X sold with TQQQ on the sell signal, SGOV when out.
VXUS before 2011-01-28: Vanguard Total International Stock Index fund (VGTSX, weekly corr 0.996), spliced.
SCHD before 2011-10-20: VEIPX (as in all other tests).
  python3 vxus_test.py         actual history (sleeve USD, P1 KRW)
  python3 vxus_test.py boot    long block bootstrap 1999~ (KRW, TQQQ 60 / gold 30 / SGOV 10)
"""
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "actual"
sys.argv = sys.argv[:1]
import numpy as np
import pandas as pd
import yfinance as yf
import schd_alt_test as S

E, D, pb, K, rs, C = S.E, S.D, S.pb, S.K, S.rs, S.C
idx = E.PX.index
raw = yf.download(["VXUS", "VGTSX"], start="1996-01-01", end=pb.END, auto_adjust=True, progress=False)["Close"]
vxus = pb.bl.splice(raw["VGTSX"].ffill(), raw["VXUS"]).reindex(idx).ffill()
mix = lambda a, b: (1 + (a.pct_change().fillna(0) + b.pct_change().fillna(0)) / 2).cumprod()
X = {"SCHD (current)": E.PX["SCHD"], "VXUS": vxus, "VXUS 50 + SCHD 50": mix(vxus, E.PX["SCHD"]),
     "SPY": S.LONG["SPY"], "SGOV": S.LONG["SGOV"]}


def actual():
    tq = E.PX["TQQQ"]
    print("Alone (USD, CAGR / MDD), and monthly correlation with TQQQ:")
    for k in ("SCHD (current)", "VXUS", "SPY"):
        l = X[k]
        f = lambda a, b=None: "%5.1f%% / %6.1f%%" % tuple(100 * pb.stats(l.loc[a:b])[m] for m in ("CAGR", "MDD"))
        c = l.resample("ME").last().pct_change().corr(tq.resample("ME").last().pct_change())
        print(f"  {k:15s} 1999~ {f('1999-01-04')} | 2011~ {f('2011-01-03')} | 2018~ {f('2018-01-01')} | corr TQQQ {c:.2f}")
    y = pd.DataFrame({k: X[k].resample("YE").last().pct_change() for k in ("SCHD (current)", "VXUS", "SPY")}).loc["2000":]
    y.index = y.index.year
    print((y * 100).round(0).astype(int).T.to_string())
    print(f"  VXUS beat SCHD in {(y['VXUS'] > y['SCHD (current)']).sum()} of {len(y)} years")
    rows = {}
    for k, l in X.items():
        e = S.sleeve(l)
        e11 = S.sleeve(l, "2011-01-28")
        a, b, c = pb.stats(e), pb.stats(e.loc["2018-01-01":]), pb.stats(e11)
        lg, ty = S.p1(e, "1999-01-04", pb.SWITCH), S.p1(e, "2016-11-07", "2016-11-07")
        m, t = C.metrics(lg), pb.stats(ty)
        x = lg.loc[lg.index.isin(pb.stock)]
        yr = pd.concat([x.iloc[:1], x.resample("YE").last()]).pct_change().dropna()
        yr.index = yr.index.year
        rows[k] = {"99~ CAGR": a["CAGR"], "99~ MDD": a["MDD"], "99~ Sh": a["Sharpe"], "11~ CAGR": c["CAGR"], "11~ MDD": c["MDD"],
                   "18~ CAGR": b["CAGR"], "18~ MDD": b["MDD"], "P1 CAGR": m["CAGR"], "P1 MDD": m["MDD"], "P1 Sh": m["Sharpe"],
                   "P1 10y CAGR": t["CAGR"], "P1 10y MDD": t["MDD"], "P1 10y Sh": t["Sharpe"],
                   "P1 2003": yr[2003], "P1 2007": yr[2007], "P1 2017": yr[2017], "P1 2021": yr[2021], "P1 2025": yr[2025]}
    df = pd.DataFrame(rows).T
    df.to_csv("vxus_grid.csv", float_format="%.4f")
    fm = {c: ("{:.2f}".format if "Sh" in c else "{:.1%}".format) for c in df.columns}
    print("\nTQQQ switching sleeve (USD) and P1 (KRW)")
    print(df.to_string(formatters=fm))
    # per-trade: return of X while held inside the sleeve (from first ladder step to the sell)
    held = []
    inv, entry, hit, t0 = False, 0.0, 0, None
    up = dn = 0
    T, MA = E.T, E.MA
    for i in range(len(T)):
        if MA[i] == MA[i]:
            up = up + 1 if T[i] > MA[i] * 1.01 else 0
            dn = dn + 1 if T[i] < MA[i] * 0.99 else 0
        if not inv and up >= 3:
            inv, entry, hit, t0 = True, T[i], 0, None
        elif inv and dn >= 2:
            if t0 is not None:
                held.append((idx[t0].date(), idx[i].date(), *[X[k].iloc[i] / X[k].iloc[t0] - 1 for k in ("SCHD (current)", "VXUS", "SPY")]))
            inv = False
        elif inv and hit == 0 and T[i] / entry - 1 >= 0.20:
            hit, t0 = 1, i
    if inv and t0 is not None:
        held.append((idx[t0].date(), idx[-1].date(), *[X[k].iloc[-1] / X[k].iloc[t0] - 1 for k in ("SCHD (current)", "VXUS", "SPY")]))
    h = pd.DataFrame(held, columns=["first +20%", "TQQQ sell", "SCHD", "VXUS", "SPY"])
    print("\nReturn of the destination ETF from the first +20% step to the TQQQ sell (per trade):")
    print(h.to_string(index=False, formatters={c: "{:+.1%}".format for c in ("SCHD", "VXUS", "SPY")}))
    print(f"  VXUS beat SCHD in {(h.VXUS > h.SCHD).sum()} of {len(h)} trades; mean SCHD {h.SCHD.mean():+.1%}, VXUS {h.VXUS.mean():+.1%}")


def boot():
    S.LONG = {k: v for k, v in X.items() if k != "SPY"}
    S.boot()


if __name__ == "__main__":
    pd.set_option("display.width", 300)
    {"actual": actual, "boot": boot}[MODE]()
