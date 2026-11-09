"""BTC/KRW trend following with a profit-taking ladder into a KRW ultra-short bond.

Rule: SMA N days with band b on Upbit BTC/KRW (fee 0.05%). On each entry, when BTC reaches +25/+50/+100/+200%
over the entry price, sell 25/25/50/50% of the BTC still held into KRW ultra-short bonds (KRW short rate).
Sell signal: sell remaining BTC (bond part stays as KRW cash). Next buy signal: the whole sleeve goes back to BTC.
Compared with no ladder, standalone and inside P1 (KRW, gold held, TQQQ unchanged).
"""
import numpy as np
import pandas as pd
import btc_krw_grid as BG
import gold_krw as K
import gold_regime_switch as rs
import portfolio_long as pb
import compare_three as C

LADDER = [(0.25, 0.25), (0.50, 0.25), (1.00, 0.50), (2.00, 0.50)]
FEE = BG.FEE
px, cash = BG.px, BG.cash


def run(n, band, ladder=None):
    ma = px.rolling(n).mean().values
    p_, c_ = px.values, cash.values
    money, coin = 1.0, 0.0
    pos, entry, hit, log = 0, 0.0, 0, []
    eq = np.empty(len(p_))
    for i, (p, m) in enumerate(zip(p_, ma)):
        money *= 1 + c_[i]
        if m == m:
            if pos == 0 and p > m * (1 + band):
                coin, money = money * (1 - FEE) / p, 0.0
                pos, entry, hit = 1, p, 0
                log.append((px.index[i], "BUY"))
            elif pos == 1 and p < m * (1 - band):
                money += coin * p * (1 - FEE)
                coin, pos = 0.0, 0
                log.append((px.index[i], "SELL"))
            elif pos == 1 and ladder:
                while hit < len(ladder) and p / entry - 1 >= ladder[hit][0]:
                    q = coin * ladder[hit][1]
                    coin -= q
                    money += q * p * (1 - FEE)
                    log.append((px.index[i], f"TP+{ladder[hit][0]:.0%}"))
                    hit += 1
        eq[i] = money + coin * p
    return pd.Series(eq, index=px.index), log


def cm(s, a, b=None):
    s = s.loc[a:b] / s.loc[a:b].iloc[0]
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    r = s.pct_change().dropna()
    return s.iloc[-1] ** (1 / yrs) - 1, (s / s.cummax() - 1).min(), r.mean() / r.std() * 365 ** 0.5


def p1(eq, start="2016-11-07"):
    sl = K.SLK.copy()
    e = eq.reindex(sl.index).ffill().fillna(1.0)
    sl["BTC"] = e / e.iloc[0]
    rs.SL = sl
    return rs.fixed(*K.P1, start, start)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    V = {"120d b1%": (120, 0.01, None), "120d b1% + ladder": (120, 0.01, LADDER),
         "30d b2%": (30, 0.02, None), "30d b2% + ladder": (30, 0.02, LADDER)}
    curves, logs = {}, {}
    for k, (n, b, lad) in V.items():
        curves[k], logs[k] = run(n, b, lad)
    curves["BTC B&H"] = px / px.iloc[0]
    print("BTC/KRW standalone (CAGR / MDD / Sharpe)")
    for k, s in curves.items():
        row = []
        for a, b in [("2015-01-15", None), ("2015-01-15", "2020-12-31"), ("2021-01-01", None), ("2018-01-01", None)]:
            c, d, sh = cm(s, a, b)
            row.append(f"{c:6.1%}/{d:6.1%}/{sh:4.2f}")
        tp = sum(1 for x in logs.get(k, []) if x[1].startswith("TP"))
        print(f"  {k:18s} 2015~ {row[0]}  15~20 {row[1]}  21~ {row[2]}  18~ {row[3]}  TP sells {tp}")
    per = [("2017-01-01", "2017-12-17"), ("2017-12-17", "2018-12-15"), ("2020-03-13", "2021-04-14"),
           ("2021-04-14", "2021-07-20"), ("2021-07-20", "2021-11-10"), ("2021-11-10", "2022-11-21"),
           ("2023-01-01", "2024-03-14"), ("2024-03-14", "2024-09-06"), ("2024-09-06", "2025-01-20"), ("2025-01-20", "2026-11-06")]
    print("\nCycle legs:")
    for k, s in curves.items():
        print(f"  {k:18s} " + " ".join(f"{a[2:7]}~{b[2:7]} {s.asof(b) / s.asof(a) - 1:+5.0%}" for a, b in per))
    yr = pd.DataFrame(curves).resample("YE").last()
    yr = pd.concat([pd.DataFrame(curves).iloc[:1], yr]).pct_change().dropna()
    yr.index = yr.index.year
    print("\nYearly %\n", (yr * 100).round(0).T.to_string())
    print("\nP1 KRW (TQQQ unchanged, gold held, BTC sleeve replaced)")
    fmt = {k: "{:.1%}".format for k in ("CAGR", "MDD", "Vol", "Worst yr", "Worst 1y roll", "Worst day")}
    fmt.update({"Sharpe": "{:.2f}".format, "Calmar": "{:.2f}".format})
    pc = pd.DataFrame({k: p1(curves[k]) for k in V})
    print(pd.DataFrame({c: C.metrics(pc[c]) for c in pc}).T.to_string(formatters=fmt))
    for a in ("2018-01-01", "2021-01-01"):
        print(f"  from {a}: " + " | ".join(f"{c} {pb.stats(pc[c].loc[a:])['CAGR']:.1%}/{pb.stats(pc[c].loc[a:])['MDD']:.1%}" for c in pc))
    print("\n120d + ladder log since 2023:", [(d.date(), e) for d, e in logs["120d b1% + ladder"] if d >= pd.Timestamp("2023-01-01")])
