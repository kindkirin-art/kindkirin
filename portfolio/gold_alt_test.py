"""Replace the 30% gold of P1 with a leveraged gold ETF (UGL, 2x) and/or a gold miners ETF (GDX). KRW, gold simply held.

UGL real from 2008-12; before: 2 x daily gold return - T-bill - 0.95%/yr. GDX real from 2006-05; before: regression on
Fidelity Select Gold (FSAGX, daily corr 0.97). GDXJ (2009-11~) only in the 2011~ tests.
  python3 gold_alt_test.py          actual history
  python3 gold_alt_test.py long     block bootstrap 1999~ (no BTC)
  python3 gold_alt_test.py btc      block bootstrap 2014-09~ (BTC included)
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
raw = yf.download(["UGL", "GDX", "GDXJ", "FSAGX"], start="1995-01-01", end=pb.END, auto_adjust=True, progress=False)["Close"].reindex(stock)
rr = raw.pct_change(fill_method=None)
f = rr[["GDX", "FSAGX"]].dropna()
BETA = np.polyfit(f["FSAGX"], f["GDX"], 1)
RET = {"UGL": rr["UGL"].fillna(2 * pb.gold_r - pb.irx - 0.0095 / 252).fillna(0.0),
       "GDX": rr["GDX"].fillna(BETA[1] + BETA[0] * rr["FSAGX"]).fillna(0.0),
       "GDXJ": rr["GDXJ"].fillna(rr["GDX"]).fillna(0.0)}
# what replaces the 30% gold slot (remaining weight stays in gold unless SGOV is named)
MIX = {"UGL 30": {"UGL": .30}, "UGL 15 + GLD 15": {"UGL": .15, "GLD": .15}, "UGL 15 + SGOV 15": {"UGL": .15, "SGOV": .15},
       "GDX 30": {"GDX": .30}, "GDX 15 + GLD 15": {"GDX": .15, "GLD": .15}, "GDX 10 + GLD 20": {"GDX": .10, "GLD": .20},
       "UGL 15 + GDX 15": {"UGL": .15, "GDX": .15}, "UGL 10 + GDX 10 + GLD 10": {"UGL": .10, "GDX": .10, "GLD": .10},
       "GDXJ 15 + GLD 15": {"GDXJ": .15, "GLD": .15}}
LONG = [k for k in MIX if "GDXJ" not in k]


def wts(mix):
    out = []
    for base in K.P1:
        x = {k: v for k, v in base.items() if k != "GLD"}
        for k, v in mix.items():
            x[k] = x.get(k, 0) + v
        out.append(x)
    return out


def actual():
    print(f"GDX proxy: GDX = {BETA[0]:.3f} x FSAGX {BETA[1] * 252:+.2%}/yr")
    sl = D.base_sleeves()
    for k, r in RET.items():
        D.add(sl, k, (1 + r).cumprod())
    rs.SL = sl
    print("\nAlone in KRW (CAGR / MDD / vol):")
    for k in ("GLD", "UGL", "GDX", "GDXJ"):
        g = lambda a, b=None: "%5.1f%% / %6.1f%% / %4.1f%%" % tuple(100 * pb.stats(sl[k].loc[a:b])[x] for x in ("CAGR", "MDD", "Vol"))
        print(f"  {k:4s} 1999~ {g('1999-01-04') if k != 'GDXJ' else '-':28s} | 2011-09~2015-12 {g('2011-09-01', '2015-12-31')} | 2018~ {g('2018-01-01')}")
    m = sl.loc[sl.index.isin(stock)].resample("ME").last().pct_change()
    print("\nMonthly return correlation in KRW, 2006-06~:")
    print(m.loc["2006-06":].corr().loc[["GLD", "UGL", "GDX"], ["TQQQ_SW", "GLD", "BTC", "SGOV", "UGL", "GDX"]].round(2).to_string())
    y0 = sl.loc[sl.index.isin(stock), ["GLD", "UGL", "GDX"]]
    yy = pd.concat([y0.iloc[:1], y0.resample("YE").last()]).pct_change().dropna()
    yy.index = yy.index.year
    print("\nYearly % (KRW):")
    print((yy * 100).round(0).astype(int).T.to_string())
    rows = {}
    for name, mix in {"GLD 30 (base P1)": {"GLD": .30}, **MIX}.items():
        w = wts(mix)
        lg, c18, ty = rs.fixed(*w), rs.fixed(*w, "2018-01-02", "2018-01-02"), rs.fixed(*w, "2016-11-07", "2016-11-07")
        m_, b, t = C.metrics(lg), pb.stats(c18), pb.stats(ty)
        x = lg.loc[lg.index.isin(stock)]
        yr = pd.concat([x.iloc[:1], x.resample("YE").last()]).pct_change().dropna()
        yr.index = yr.index.year
        ok = "GDXJ" not in name
        nan = float("nan")
        rows[name] = {"99~ CAGR": m_["CAGR"] if ok else nan, "99~ MDD": m_["MDD"] if ok else nan, "99~ Sh": m_["Sharpe"] if ok else nan,
                      "Worst 1y": m_["Worst 1y roll"] if ok else nan, "11-15 CAGR": pb.stats(lg.loc["2011-09-01":"2015-12-31"])["CAGR"],
                      "18~ CAGR": b["CAGR"], "18~ MDD": b["MDD"], "18~ Sh": b["Sharpe"], "10y CAGR": t["CAGR"], "10y MDD": t["MDD"], "10y Sh": t["Sharpe"],
                      "2008": yr[2008] if ok else nan, "2013": yr[2013], "2022": yr[2022], "2025": yr[2025], "2026": yr[2026]}
    print()
    D.show(rows).to_csv("gold_alt_grid.csv", float_format="%.4f")


def boot_long():
    import bootstrap_long as B
    from bootstrap import sleeve_tqqq
    R = B.R.copy()
    for k in ("UGL", "GDX"):
        R[k] = RET[k].values
    cols_, RA = list(R.columns), R.values
    wmap = {n: {"TQ": .6, "SGOV": .1 + m.get("SGOV", 0), **{k: v for k, v in m.items() if k != "SGOV"}}
            for n, m in {"GLD 30 (base)": {"GLD": .30}, **{k: MIX[k] for k in LONG}}.items()}
    n = int((B.YEARS + B.WARM) * B.TD)
    rng = np.random.default_rng(7)
    res = {k: [] for k in wmap}
    for _ in range(B.NPATH):
        idx = []
        while len(idx) < n:
            a = rng.integers(0, len(RA) - B.BLOCK)
            idx.extend(range(a, a + B.BLOCK))
        P = RA[np.array(idx[:n])]
        g = {c: P[:, i] for i, c in enumerate(cols_)}
        lvl = lambda r: np.cumprod(1 + r)
        fx = lvl(g["FX"])
        cols = {"TQ": sleeve_tqqq(lvl(g["TQQQ"]), lvl(g["SCHD"]), g["CASHU"], np.ones(n)) * fx, "GLD": lvl(g["GOLD"]) * fx,
                "SGOV": lvl(g["CASHU"]) * fx, "UGL": lvl(g["UGL"]) * fx, "GDX": lvl(g["GDX"]) * fx}
        yrs = pd.bdate_range("2000-01-03", periods=n).year.values
        out = A.combine(cols, wmap, np.r_[yrs[1:] != yrs[:-1], False], int(B.WARM * B.TD))
        for k in res:
            res[k].append(B.metrics(out[k], B.YEARS))
    A.report({k: np.array(v) for k, v in res.items()}, "GLD 30 (base)",
             f"Long bootstrap 1999~: block {B.BLOCK} trading days, {B.NPATH} paths x {B.YEARS}y, KRW, TQQQ 60 / gold slot 30 / SGOV 10")


def boot_btc():
    import bootstrap as S
    R = S.R.copy()
    ff = lambda r: (1 + r).cumprod().reindex(S.cal.union(stock)).ffill().reindex(S.cal).pct_change().fillna(0.0).values
    for k in ("UGL", "GDX", "GDXJ"):
        R[k] = ff(RET[k])
    cols_ = list(R.columns)
    S.RA = R.values
    wmap = {n: {"TQ": .4, "BTC": .2, "SGOV": .1 + m.get("SGOV", 0), **{k: v for k, v in m.items() if k != "SGOV"}}
            for n, m in {"GLD 30 (base)": {"GLD": .30}, **MIX}.items()}
    n, w0 = int((S.YEARS + S.WARM) * 365), int(S.WARM * 365)
    rng = np.random.default_rng(42)
    res = {k: [] for k in wmap}
    for _ in range(S.NPATH):
        P = S.sample_path(rng, n)
        g = {c: P[:, i] for i, c in enumerate(cols_)}
        lvl = lambda r: np.cumprod(1 + r)
        fx, bk = lvl(g["FX"]), lvl(g["BTCK"])
        st = g["STOCK"] > 0
        cols = {"TQ": S.sleeve_tqqq(lvl(g["TQQQ"]), lvl(g["SCHD"]), g["CASHU"], g["STOCK"]) * fx, "GLD": lvl(g["GOLD"]) * fx,
                "BTC": S.sleeve_trend(g["BTCK"], bk, S.sma(bk, 30), 0.02, 1, g["CASHK"], 0.0005), "SGOV": lvl(g["CASHU"]) * fx,
                "UGL": lvl(g["UGL"]) * fx, "GDX": lvl(g["GDX"]) * fx, "GDXJ": lvl(g["GDXJ"]) * fx}
        yrs = pd.date_range("2000-01-01", periods=n, freq="D").year.values
        reb = np.zeros(n, bool)
        for y in np.unique(yrs):
            j = np.where((yrs == y) & st)[0]
            if len(j):
                reb[j[-1]] = True
        out = A.combine(cols, wmap, reb, w0)
        for k in res:
            res[k].append(S.metrics(out[k], st[w0:]))
    A.report({k: np.array(v) for k, v in res.items()}, "GLD 30 (base)",
             f"BTC-era bootstrap 2014-09~: block {S.BLOCK} days, {S.NPATH} paths x {S.YEARS}y, KRW, TQQQ 40 / gold slot 30 / BTC 20 / SGOV 10")


if __name__ == "__main__":
    pd.set_option("display.width", 300)
    {"actual": actual, "long": boot_long, "btc": boot_btc}[MODE]()
