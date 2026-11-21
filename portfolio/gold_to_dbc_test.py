"""Replace the 30% gold of P1 with PDBC (DBC index; DBC/GSCI proxy before 2006, see dbc_test.py) under a trend filter.

Filter on the DBC adjusted close (USD); SGOV when out; fee 0.1%. Everything else is the current P1 (KRW, BTC SMA30 2%).
  python3 gold_to_dbc_test.py          filter grid on actual history (1999~, proxy 1999-2005 / real 2006~)
  python3 gold_to_dbc_test.py long     block bootstrap 1999~ (no BTC): TQQQ 60 / slot 30 / SGOV 10
  python3 gold_to_dbc_test.py btc      block bootstrap 2014-09~ (BTC included)
"""
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "actual"
sys.argv = sys.argv[:1]
import numpy as np
import pandas as pd
import sgov_alt_test as A

D, pb, K, rs, C = A.D, A.pb, A.K, A.rs, A.C
stock = pb.stock
RULES = [("hold",)]
RULES += [("d", n, b, k) for n in (50, 100, 150, 200, 250) for b in (0.0, 0.01, 0.02, 0.03) for k in (1, 3)]
RULES += [("sma", n, b, k) for n in (6, 8, 10, 12) for b in (0.0, 0.02, 0.05) for k in (1, 2)]
RULES += [("mom", n, 0.0, 1) for n in (3, 6, 9, 12)]
name = lambda r: "hold" if r[0] == "hold" else (f"{r[1]}d {r[2]:.0%} k{r[3]}" if r[0] == "d" else
                                               f"{r[0]}{r[1]}m {r[2]:.0%} k{r[3]}" if r[0] == "sma" else f"mom{r[1]}m")


def held(r):
    return None if r[0] == "hold" else D.held_daily(*r[1:]) if r[0] == "d" else D.held_monthly(*r)


def wts(slot):
    out = []
    for base in K.P1:
        x = {k: v for k, v in base.items() if k != "GLD"}
        x.update(slot)
        out.append(x)
    return out


def evaluate(sl, w):
    rs.SL = sl
    lg = rs.fixed(*w)
    m = C.metrics(lg)
    x = lg.loc[lg.index.isin(stock)]
    y = pd.concat([x.iloc[:1], x.resample("YE").last()]).pct_change().dropna()
    y.index = y.index.year
    s = {p: pb.stats(lg.loc[a:b]) for p, (a, b) in {"99-05": ("1999-01-04", "2005-12-31"), "06~": ("2006-02-06", None),
                                                     "06-17": ("2006-02-06", "2017-12-31")}.items()}
    s18, ty = pb.stats(rs.fixed(*w, "2018-01-02", "2018-01-02")), pb.stats(rs.fixed(*w, "2016-11-07", "2016-11-07"))
    return {"CAGR": m["CAGR"], "MDD": m["MDD"], "Sh": m["Sharpe"], "Worst 1y": m["Worst 1y roll"],
            "99-05": s["99-05"]["CAGR"], "06~": s["06~"]["CAGR"], "06~ MDD": s["06~"]["MDD"], "06~ Sh": s["06~"]["Sharpe"],
            "06-17": s["06-17"]["CAGR"], "18~": s18["CAGR"], "18~ MDD": s18["MDD"], "18~ Sh": s18["Sharpe"],
            "10y": ty["CAGR"], "10y MDD": ty["MDD"], "10y Sh": ty["Sharpe"],
            "2008": y[2008], "2013": y[2013], "2015": y[2015], "2022": y[2022], "2025": y[2025]}


def actual():
    sl = D.base_sleeves()
    base = evaluate(sl, K.P1)
    rows, alone = {"gold 30 (current P1)": base}, {}
    for r in RULES:
        h = held(r)
        s = D.sleeve_usd(h)
        D.add(sl, "DF", s)
        rows[f"PDBC 30 {name(r)}"] = evaluate(sl, wts({"DF": .30}))
        st, s06 = pb.stats(s), pb.stats(s.loc["2006-02-06":])
        alone[name(r)] = {"in%": 1.0 if h is None else h.mean(), "trades/yr": 0 if h is None else h.diff().abs().sum() / 27.8,
                          "CAGR": st["CAGR"], "MDD": st["MDD"], "06~": s06["CAGR"], "06~ MDD": s06["MDD"]}
    df = pd.DataFrame(rows).T
    df.to_csv("gold_to_dbc_grid.csv", float_format="%.4f")
    al = pd.DataFrame(alone).T
    fm = lambda d: {c: ("{:.2f}".format if "Sh" in c else "{:.1f}".format if c == "trades/yr" else "{:.1%}".format) for c in d.columns}
    print("DBC sleeve alone (USD, filter -> SGOV):")
    print(al.sort_values("06~", ascending=False).head(15).to_string(formatters=fm(al)))
    print(f"\nP1 (KRW) with gold 30 replaced, top 25 by 06~ Sharpe (real DBC period) of {len(df) - 1} rules; base first")
    top = pd.concat([df.iloc[:1], df.iloc[1:].sort_values("06~ Sh", ascending=False).head(25)])
    print(top.to_string(formatters=fm(df)))
    print("\nWorst 5 by 06~ Sharpe:")
    print(df.iloc[1:].sort_values("06~ Sh").head(5).to_string(formatters=fm(df)))
    d = df.iloc[1:]
    print(f"\nRules beating gold-held P1 on: 99~ CAGR {(d.CAGR > base['CAGR']).sum()}, 99~ Sharpe {(d.Sh > base['Sh']).sum()}, "
          f"06~ CAGR {(d['06~'] > base['06~']).sum()}, 18~ CAGR {(d['18~'] > base['18~']).sum()}, 10y CAGR {(d['10y'] > base['10y']).sum()}, "
          f"MDD {(d.MDD > base['MDD']).sum()} (of {len(d)})")
    for lab, w in [("PDBC 15 + gold 15", {"DF": .15, "GLD": .15})]:
        for r in [("d", 250, 0.02, 1), ("d", 150, 0.02, 3), ("sma", 10, 0.05, 1)]:
            D.add(sl, "DF", D.sleeve_usd(held(r)))
            e = evaluate(sl, wts(w))
            print(f"  {lab} {name(r)}: 99~ {e['CAGR']:.1%}/{e['MDD']:.1%} Sh {e['Sh']:.2f} | 06~ {e['06~']:.1%} Sh {e['06~ Sh']:.2f} | "
                  f"18~ {e['18~']:.1%}/{e['18~ MDD']:.1%} | 10y {e['10y']:.1%}/{e['10y MDD']:.1%}")


SEL = [("hold",), ("d", 100, 0.02, 3), ("d", 150, 0.02, 3), ("d", 200, 0.02, 1), ("d", 250, 0.02, 1), ("sma", 10, 0.05, 1), ("sma", 12, 0.05, 1)]


def dbc_sleeves(dret, cashu, stock_mask, months):
    """DBC filtered sleeves on one synthetic path (USD). Daily rules on stock days; monthly on stock-day month ends."""
    lvl = np.cumprod(1 + dret)
    ii = np.where(stock_mask)[0]
    out = {}
    for r in SEL:
        h = np.ones(len(lvl))
        if r[0] == "d":
            p = lvl[ii]
            for a, v in zip(ii, A_trend(p, D_sma(p, r[1]), r[2], r[3])):
                h[a:] = v
        elif r[0] == "sma":
            last = {}
            for i in ii:
                last[months[i]] = i
            me = np.array(sorted(last.values()))
            p = lvl[me]
            for a, v in zip(me, A_trend(p, D_sma(p, r[1]), r[2], r[3])):
                h[a:] = v
        prev = np.r_[1.0, h[:-1]]
        out[name(r)] = np.cumprod(1 + dret * prev + (1 - prev) * cashu - (h != prev) * 0.001)
    return out


from bootstrap import sma as D_sma, trend_pos


def A_trend(p, m, b, k):
    return trend_pos(p, m, b, k, start_pos=1)


def boot_long():
    import bootstrap_long as B
    from bootstrap import sleeve_tqqq
    R = B.R.copy()
    R["DBC"] = D.dbc_r.reindex(B.days).fillna(0).values
    cols_, RA = list(R.columns), R.values
    wmap = {"gold 30 (base)": {"TQ": .6, "GLD": .3, "SGOV": .1}, **{f"PDBC 30 {name(r)}": {"TQ": .6, name(r): .3, "SGOV": .1} for r in SEL},
            "PDBC 15 sma10m + gold 15": {"TQ": .6, "sma10m 5% k1": .15, "GLD": .15, "SGOV": .1}}
    n = int((B.YEARS + B.WARM) * B.TD)
    rng = np.random.default_rng(7)
    res = {k: [] for k in wmap}
    dts = pd.bdate_range("2000-01-03", periods=n)
    months = (dts.year * 12 + dts.month).values
    yrs = dts.year.values
    reb = np.r_[yrs[1:] != yrs[:-1], False]
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
                "SGOV": lvl(g["CASHU"]) * fx}
        cols.update({k: v * fx for k, v in dbc_sleeves(g["DBC"], g["CASHU"], np.ones(n, bool), months).items()})
        out = A.combine(cols, wmap, reb, int(B.WARM * B.TD))
        for k in res:
            res[k].append(B.metrics(out[k], B.YEARS))
    A.report({k: np.array(v) for k, v in res.items()}, "gold 30 (base)",
             f"Long bootstrap 1999~: block {B.BLOCK} trading days, {B.NPATH} paths x {B.YEARS}y, KRW, TQQQ 60 / slot 30 / SGOV 10")


def boot_btc():
    import bootstrap as S
    R = S.R.copy()
    R["DBC"] = (1 + D.dbc_r).cumprod().reindex(S.cal.union(stock)).ffill().reindex(S.cal).pct_change().fillna(0.0).values
    cols_ = list(R.columns)
    S.RA = R.values
    wmap = {"gold 30 (base)": {"TQ": .4, "GLD": .3, "BTC": .2, "SGOV": .1},
            **{f"PDBC 30 {name(r)}": {"TQ": .4, name(r): .3, "BTC": .2, "SGOV": .1} for r in SEL},
            "PDBC 15 sma10m + gold 15": {"TQ": .4, "sma10m 5% k1": .15, "GLD": .15, "BTC": .2, "SGOV": .1}}
    n, w0 = int((S.YEARS + S.WARM) * 365), int(S.WARM * 365)
    days = pd.date_range("2000-01-01", periods=n, freq="D")
    months, yrs = (days.year * 12 + days.month).values, days.year.values
    rng = np.random.default_rng(42)
    res = {k: [] for k in wmap}
    for _ in range(S.NPATH):
        P = S.sample_path(rng, n)
        g = {c: P[:, i] for i, c in enumerate(cols_)}
        lvl = lambda r: np.cumprod(1 + r)
        fx, bk = lvl(g["FX"]), lvl(g["BTCK"])
        st = g["STOCK"] > 0
        cols = {"TQ": S.sleeve_tqqq(lvl(g["TQQQ"]), lvl(g["SCHD"]), g["CASHU"], g["STOCK"]) * fx, "GLD": lvl(g["GOLD"]) * fx,
                "BTC": S.sleeve_trend(g["BTCK"], bk, S.sma(bk, 30), 0.02, 1, g["CASHK"], 0.0005), "SGOV": lvl(g["CASHU"]) * fx}
        cols.update({k: v * fx for k, v in dbc_sleeves(g["DBC"], g["CASHU"], st, months).items()})
        reb = np.zeros(n, bool)
        for y in np.unique(yrs):
            j = np.where((yrs == y) & st)[0]
            if len(j):
                reb[j[-1]] = True
        out = A.combine(cols, wmap, reb, w0)
        for k in res:
            res[k].append(S.metrics(out[k], st[w0:]))
    A.report({k: np.array(v) for k, v in res.items()}, "gold 30 (base)",
             f"BTC-era bootstrap 2014-09~: block {S.BLOCK} days, {S.NPATH} paths x {S.YEARS}y, KRW, TQQQ 40 / slot 30 / BTC 20 / SGOV 10")


if __name__ == "__main__":
    pd.set_option("display.width", 320)
    {"actual": actual, "long": boot_long, "btc": boot_btc}[MODE]()
