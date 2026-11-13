"""When to sell the SCHD that the TQQQ switching sleeve accumulated by profit taking.

Current rule: SCHD is sold together with TQQQ on the TQQQ sell signal. Here TQQQ is still sold on its signal
(-> SGOV) and the TQQQ -> SCHD ladder is unchanged (+20/50/100/200% -> 25/25/50/50%), but SCHD gets its own exit:
  with            sell with TQQQ (current rule)
  rebuy           hold until the next TQQQ buy signal, then switch SCHD into TQQQ
  days N          hold N more trading days
  stop X          sell when SCHD closes X below its close on the TQQQ sell day
  trail X         sell when SCHD closes X below its highest close since the TQQQ sell day
  ds N X          hold N more trading days, but sell earlier if SCHD closes X below its TQQQ-sell-day close
  sma N B K       sell after K closes below SCHD's N-day SMA * (1 - B)   (checked from the TQQQ sell day on)
At the next TQQQ buy any SCHD still held is switched into TQQQ ("merge"), so each new trade starts 100% TQQQ
as now. "carry" variants instead keep SCHD through the next trade (only cash buys TQQQ); "never" never sells.
SCHD before 2011-10 is Vanguard Equity Income (VEIPX). Fee 0.1%, idle cash earns T-bills.
  python3 schd_exit_test.py         actual history (sleeve in USD, P1 in KRW)
  python3 schd_exit_test.py boot    long block bootstrap (1999~, KRW, TQQQ 60 / gold 30 / SGOV 10)
"""
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "actual"
sys.argv = sys.argv[:1]
import numpy as np
import pandas as pd
import dbc_test as D
from bootstrap import sma

pb, K, rs, C = D.pb, D.K, D.rs, D.C
LEVELS = [(0.20, 0.25), (0.50, 0.25), (1.00, 0.50), (2.00, 0.50)]
FEE = 0.001


def engine(t, s, cashr, ma_t, rule=("with",), carry=False, sm=None, info=None):
    """t, s: TQQQ / SCHD levels; ma_t: TQQQ 200d SMA; sm: SCHD SMA for 'sma' rules. Returns equity array."""
    cash, st, ss = 1.0, 0.0, 0.0
    inv, up, dn, entry, hit = False, 0, 0, 0.0, 0
    ref = peak = 0.0
    age = below = 0
    eq = np.empty(len(t))
    kind = rule[0]
    for i in range(len(t)):
        if cash > 0:
            cash *= 1 + cashr[i]
        p, m, q = t[i], ma_t[i], s[i]
        if m == m:
            up = up + 1 if p > m * 1.01 else 0
            dn = dn + 1 if p < m * 0.99 else 0
        sold_now = False
        if not inv and m == m and up >= 3:
            if ss and not carry:
                cash += ss * q * (1 - FEE)
                if info is not None:
                    info.append((i, "rebuy", age))
                ss = 0.0
            st, cash = cash / (1 + FEE) / p, 0.0
            inv, entry, hit = True, p, 0
        elif inv and dn >= 2:
            cash += st * p * (1 - FEE)
            st, inv, sold_now = 0.0, False, True
            if ss:
                if info is not None:
                    info.append((i, "tsell", ss * q / (cash + ss * q)))
                ref = peak = q
                age = below = 0
        elif inv:
            while hit < 4 and p / entry - 1 >= LEVELS[hit][0]:
                v = st * p * LEVELS[hit][1]
                st -= v / p
                ss += v * (1 - FEE) / (1 + FEE) / q
                hit += 1
        if ss and (not inv or carry) and kind != "never" and not (carry and inv and kind != "sma"):
            if not sold_now:
                age += 1
            peak = max(peak, q)
            if kind == "sma":
                below = below + 1 if q < sm[i] * (1 - rule[2]) else 0
            out = (kind == "with" or (kind == "days" and age >= rule[1]) or (kind == "stop" and q < ref * (1 - rule[1]))
                   or (kind == "trail" and q < peak * (1 - rule[1]))
                   or (kind == "ds" and (age >= rule[1] or q < ref * (1 - rule[2]))) or (kind == "sma" and below >= rule[3]))
            if out:
                cash += ss * q * (1 - FEE)
                if info is not None and not sold_now:
                    info.append((i, "ssell", age))
                ss = 0.0
        eq[i] = cash + st * p + ss * q
    return eq


RULES = [("with",), ("rebuy",)]
RULES += [("days", n) for n in (5, 10, 20, 40, 60, 120)]
RULES += [("stop", x) for x in (0.02, 0.03, 0.05, 0.07, 0.10)]
RULES += [("trail", x) for x in (0.02, 0.03, 0.05, 0.07, 0.10)]
RULES += [("ds", n, x) for n in (60, 120) for x in (0.05, 0.10, 0.15)]
RULES += [("sma", n, b, k) for n in (20, 50, 100, 200) for b in (0.0, 0.01, 0.02) for k in (1, 2)]
CARRY = [("never",), ("sma", 100, 0.01, 1), ("sma", 200, 0.01, 1), ("sma", 200, 0.02, 2), ("trail", 0.05), ("trail", 0.10)]
name = lambda r, c=False: " ".join(str(x) for x in r) + (" carry" if c else "")

PX = pb.bl.px
schd_full = pb.bl.splice(pb.bl.raw["VEIPX"].ffill(), pb.bl.raw["SCHD"])
T, S = PX["TQQQ"].values, PX["SCHD"].values
CASH = pb.cash_r.reindex(PX.index).fillna(0).values
MA = PX["MA200"].values
SM = {n: schd_full.rolling(n).mean().reindex(PX.index).values for n in (20, 50, 100, 200)}


def run(rule, carry=False, info=None):
    return pd.Series(engine(T, S, CASH, MA, rule, carry, SM.get(rule[1]) if rule[0] == "sma" else None, info), index=PX.index)


def actual():
    base = run(("with",))
    ref = pb.sleeves["TQQQ_SW"].reindex(PX.index)
    print("engine check vs existing sleeve: max rel diff %.2e" % abs(base / base.iloc[0] / (ref / ref.iloc[0]) - 1).max())
    # event study
    info = []
    run(("rebuy",), info=info)
    ts = [(i, x) for i, k, x in info if k == "tsell"]
    idx = PX.index
    print(f"\nTQQQ sell signals with SCHD on hand: {len(ts)} (of {sum(1 for x in pb.bt.run(PX, FEE=FEE, band=.01, cash_r=pb.cash_r)[1] if x[1] == 'SELL')} sells), "
          f"SCHD share of sleeve at the sell: median {np.median([x for _, x in ts]):.0%}, max {max(x for _, x in ts):.0%}")
    rb = [i for i, k, _ in info if k == "rebuy"]
    rows = []
    cl = np.cumprod(1 + CASH)
    for i, shr in ts:
        j = next((x for x in rb if x > i), len(idx) - 1)
        row = {"TQQQ sell": idx[i].date(), "next buy": idx[j].date(), "days": j - i, "SCHD share": shr}
        for n in (5, 20, 60, 120):
            e = min(i + n, len(idx) - 1)
            row[f"SCHD +{n}d"] = S[e] / S[i] - 1
        row["SCHD to rebuy"] = S[j] / S[i] - 1
        row["SCHD low"] = S[i:j + 1].min() / S[i] - 1
        row["cash to rebuy"] = cl[j] / cl[i] - 1
        row["QQQ to rebuy"] = PX["QQQ"].values[j] / PX["QQQ"].values[i] - 1
        rows.append(row)
    ev = pd.DataFrame(rows)
    f = {c: "{:+.1%}".format for c in ev.columns if c.startswith(("SCHD +", "SCHD to", "SCHD low", "cash", "QQQ"))}
    f["SCHD share"] = "{:.0%}".format
    print(ev.to_string(index=False, formatters=f))
    num = ev.drop(columns=["TQQQ sell", "next buy"])
    print("mean  :", {c: f"{num[c].mean():+.1%}" for c in num.columns if c not in ("days", "SCHD share")})
    print("median:", {c: f"{num[c].median():+.1%}" for c in num.columns if c not in ("days", "SCHD share")})
    print("up share:", {c: f"{(num[c] > 0).mean():.0%}" for c in num.columns if c.startswith(("SCHD +", "SCHD to"))})
    real = ev[ev["TQQQ sell"] >= pd.Timestamp("2011-10-20").date()]
    print(f"real SCHD only (2011-10~, n={len(real)}): SCHD to rebuy mean {real['SCHD to rebuy'].mean():+.1%}, up {(real['SCHD to rebuy'] > 0).mean():.0%}, "
          f"worst {real['SCHD to rebuy'].min():+.1%}, mean low {real['SCHD low'].mean():+.1%}")

    sl = D.base_sleeves()
    rows = {}
    for rule, carry in [(r, False) for r in RULES] + [(r, True) for r in CARRY]:
        inf = []
        e = run(rule, carry, inf)
        held = [x for _, k, x in inf if k in ("ssell", "rebuy")]
        a, b, c, d = (pb.stats(e.loc[x:]) for x in ("1999-01-04", "2011-10-20", "2018-01-01", "2016-11-07"))
        s = e.reindex(sl.index).ffill() * K.fx_cal
        sl["TQQQ_SW"] = s / s.iloc[0]
        rs.SL = sl
        lg, ty = rs.fixed(*K.P1), rs.fixed(*K.P1, "2016-11-07", "2016-11-07")
        m, t = C.metrics(lg), pb.stats(ty)
        rows[name(rule, carry)] = {"extra days": np.mean(held) if held else 0,
                                   "99~ CAGR": a["CAGR"], "99~ MDD": a["MDD"], "99~ Sh": a["Sharpe"],
                                   "11~ CAGR": b["CAGR"], "11~ MDD": b["MDD"], "18~ CAGR": c["CAGR"], "18~ MDD": c["MDD"],
                                   "10y CAGR": d["CAGR"], "10y MDD": d["MDD"],
                                   "P1 CAGR": m["CAGR"], "P1 MDD": m["MDD"], "P1 Sh": m["Sharpe"],
                                   "P1 10y CAGR": t["CAGR"], "P1 10y MDD": t["MDD"], "P1 10y Sh": t["Sharpe"]}
    df = pd.DataFrame(rows).T
    df.to_csv("schd_exit_grid.csv", float_format="%.4f")
    fm = {c: "{:.1%}".format for c in df.columns if "Sh" not in c}
    fm.update({c: "{:.2f}".format for c in df.columns if "Sh" in c})
    fm["extra days"] = "{:.0f}".format
    print("\nTQQQ switching sleeve (USD) and P1 (KRW, current rules)")
    print(df.to_string(formatters=fm))


def boot():
    import bootstrap_long as B
    sel = [(r, False) for r in [("with",), ("rebuy",), ("days", 20), ("days", 60), ("days", 120), ("ds", 60, 0.05), ("ds", 60, 0.10), ("ds", 60, 0.15), ("stop", 0.03), ("stop", 0.05), ("trail", 0.03),
                                ("trail", 0.05), ("trail", 0.10), ("sma", 50, 0.01, 1), ("sma", 100, 0.01, 1), ("sma", 200, 0.0, 1),
                                ("sma", 200, 0.02, 2)]] + [(r, True) for r in [("never",), ("sma", 200, 0.01, 1), ("trail", 0.10)]]
    n = int((B.YEARS + B.WARM) * B.TD)
    w0 = int(B.WARM * B.TD)
    rng = np.random.default_rng(7)
    res = {name(*x): [] for x in sel}
    sres = {name(*x): [] for x in sel}
    ci = {c: i for i, c in enumerate(B.COLS)}
    for _ in range(B.NPATH):
        idx = []
        while len(idx) < n:
            a = rng.integers(0, len(B.RA) - B.BLOCK)
            idx.extend(range(a, a + B.BLOCK))
        A = B.RA[np.array(idx[:n])]
        lvl = lambda c: np.cumprod(1 + A[:, ci[c]])
        t, s, fx, cu = lvl("TQQQ"), lvl("SCHD"), lvl("FX"), A[:, ci["CASHU"]]
        gold, sgov = lvl("GOLD") * fx, np.cumprod(1 + cu) * fx
        mt = sma(t, 200)
        sms = {k: sma(s, k) for k in (50, 100, 200)}
        yrs = pd.bdate_range("2000-01-03", periods=n).year.values
        reb = np.r_[yrs[1:] != yrs[:-1], False]
        for rule, carry in sel:
            e = engine(t, s, cu, mt, rule, carry, sms.get(rule[1]) if rule[0] == "sma" else None)
            sres[name(rule, carry)].append(B.metrics(e[w0:] / e[w0], B.YEARS))
            sl = np.column_stack([e * fx, gold, sgov])[w0:]
            sl = sl / sl[0]
            w = np.array([.6, .3, .1])
            units, v = w / sl[0], np.empty(len(sl))
            for i in range(len(sl)):
                v[i] = units @ sl[i]
                if reb[w0 + i]:
                    units = v[i] * w / sl[i]
            res[name(rule, carry)].append(B.metrics(v, B.YEARS))
    for lab, r in [("TQQQ switching sleeve alone (USD)", sres), ("P1 form TQQQ 60 / gold 30 / SGOV 10 (KRW)", res)]:
        r = {k: np.array(v) for k, v in r.items()}
        b = r["with"]
        print(f"\n{lab}: block {B.BLOCK} trading days, {B.NPATH} paths x {B.YEARS}y")
        print(f"{'':20s} CAGR med    5%    MDD med    5%  P(MDD<-40) Sharpe | vs current: CAGR> MDD> Sharpe>  dCAGR med   5%~95%")
        for k, v in r.items():
            d = v[:, 0] - b[:, 0]
            print(f"{k:20s} {np.median(v[:,0]):7.1%} {np.percentile(v[:,0],5):6.1%} {np.median(v[:,1]):8.1%} {np.percentile(v[:,1],5):6.1%}"
                  f" {(v[:,1]<-.4).mean():9.0%} {np.median(v[:,2]):8.2f} |         {(d>0).mean():5.0%} {(v[:,1]>b[:,1]).mean():5.0%}"
                  f" {(v[:,2]>b[:,2]).mean():5.0%}   {np.median(d):+.2%}  [{np.percentile(d,5):+.2%}, {np.percentile(d,95):+.2%}]")


if __name__ == "__main__":
    pd.set_option("display.width", 300)
    {"actual": actual, "boot": boot}[MODE]()
