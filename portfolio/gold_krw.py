"""Gold filter test in KRW (for KRX gold spot investors).

KRW gold = USD gold x USD/KRW (KRX gold tracks this closely; premium/fees ignored).
- FX: FRED EXKOUS (monthly avg, 1981-04~) / DEXKOUS (daily, 1981-04~). Before 1981-04 the KRW was
  pegged: 400 (~1974-11), 484 (1974-12~1979-12), 580 (1980-01) rising linearly to 677 (1981-04).
- KRW cash: Korea 3-month interbank rate (1991~), discount rate before (understates 1980s market rates).
A) Gold alone in KRW, monthly 1972~2026, filter rules from gold_filter_grid.py, idle cash = KRW cash.
B) P1 in KRW (1999~ long test and 10y): every USD sleeve x FX; gold sleeve = KRW gold, filtered on the
   KRW gold month-end close, KRW cash when out. Year-end rebalancing in KRW.
"""
import os
import pandas as pd
import gold_filter_grid as G
import portfolio_long as pb
import gold_regime_switch as rs

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data_krw")
fred = lambda f: pd.to_numeric(pd.read_csv(os.path.join(D, f), index_col=0, parse_dates=True).iloc[:, 0],
                               errors="coerce").dropna()

# FX monthly (month-end labels, monthly averages like the gold series)
fxm = fred("EXKOUS.csv")
fxm.index = fxm.index + pd.offsets.MonthEnd(0)
peg = pd.Series(index=pd.date_range("1960-01-31", "1981-03-31", freq="ME"), dtype=float)
peg[:"1974-11-30"] = 400.0
peg["1974-12-31":"1979-12-31"] = 484.0
peg["1980-01-31"] = 580.0
peg["1981-04-30"] = fxm.iloc[0]
peg = peg.reindex(peg.index.union([pd.Timestamp("1981-04-30")])).interpolate()
fx_m = pd.concat([peg.loc[:"1981-03-31"], fxm]).reindex(G.gm.index).ffill()

# KRW cash
kr = fred("IR3TIB01KRM156N.csv")
disc = fred("INTDSRKRM193N.csv")
rate_m = pd.concat([disc.loc[:kr.index[0] - pd.Timedelta(days=1)], kr])
rate_m.index = rate_m.index + pd.offsets.MonthEnd(0)
krw_cash_m = (rate_m / 100 / 12).reindex(G.gm.index).ffill().fillna(0.0)

gk_m = G.gm * fx_m                                  # KRW gold, monthly average


def alone(held, cash=krw_cash_m, px=gk_m):
    prev = held.shift().fillna(1)
    r = px.pct_change().fillna(0) * prev + (1 - prev) * cash - (held != prev) * G.FEE
    return (1 + r).cumprod()


# daily
fx_d = fred("DEXKOUS.csv")
stock = pb.stock
fx_s = fx_d.reindex(stock.union(fx_d.index)).ffill().reindex(stock)
gold_usd_d = (1 + pb.gold_r).cumprod()
gold_krw_d = gold_usd_d * fx_s
gold_krw_r = gold_krw_d.pct_change().fillna(0)
cash_krw_d = (rate_m / 100 / 252).reindex(stock.union(rate_m.index)).ffill().reindex(stock).fillna(0)
me_k = gold_krw_d.groupby([stock.year, stock.month]).tail(1)
pre = gk_m.loc[:"1998-12-31"] / gk_m.loc["1999-01-31"] * me_k.iloc[0]
m_daily_k = pd.concat([pre, me_k])

cal = pb.sleeves.index
fx_cal = fx_d.reindex(cal.union(fx_d.index)).ffill().reindex(cal)
SLK = pb.sleeves.copy()
for c in ("TQQQ_SW", "BTC", "SGOV", "GLD"):
    SLK[c] = pb.sleeves[c] * fx_cal / fx_cal.iloc[0]
P1 = pb.PORTS["P1 TQQQ40/Gold30/BTC20/SGOV10"]


def p1_krw(held_m=None, start="1999-01-04", switch=pb.SWITCH):
    sl = SLK.copy()
    if held_m is not None:
        held = held_m.loc[me_k.index].reindex(stock).ffill().fillna(1)
        prev = held.shift().fillna(1)
        r = gold_krw_r * prev + (1 - prev) * cash_krw_d - (held != prev) * G.FEE
        s = (1 + r).cumprod().reindex(cal).ffill()
        sl["GF"] = s / s.iloc[0]
    rs.SL = sl
    if held_m is None:
        return rs.fixed(*P1, start, switch)
    sw = lambda w: {("GF" if k == "GLD" else k): v for k, v in w.items()}
    return rs.fixed(sw(P1[0]), sw(P1[1]), start, switch)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    cm = G.cagr_mdd
    print("USD/KRW: 1972 %.0f, 1980 %.0f, 1999 %.0f, 2026 %.0f" % (fx_m["1972-01"].iloc[0], fx_m["1980-01"].iloc[0],
                                                                 fx_m["1999-01"].iloc[0], fx_m.iloc[-1]))
    for lab, a, b in [("1972~", "1972", None), ("1980~99", "1980-01", "1999-12"), ("2000~", "2000-01", None)]:
        u, k = cm(G.gm.loc[a:b]), cm(gk_m.loc[a:b])
        print(f"Gold B&H {lab}: USD {u[0]:.1%}/{u[1]:.1%}   KRW {k[0]:.1%}/{k[1]:.1%}")
    b_l, b_10 = pb.stats(p1_krw()), pb.stats(p1_krw(None, "2016-11-07", "2016-11-07"))
    print(f"P1 KRW gold B&H: long {b_l['CAGR']:.2%}/{b_l['MDD']:.1%} Sharpe {b_l['Sharpe']:.2f} | "
          f"10y {b_10['CAGR']:.2%}/{b_10['MDD']:.1%}")
    grid = [("sma", n, b, k) for n in (6, 8, 10, 12, 15, 18, 24) for b in (0.0, 0.02, 0.05) for k in (1, 2)]
    grid += [("ema", n, b, k) for n in (10, 12, 18, 24) for b in (0.0, 0.02) for k in (1, 2)]
    grid += [("mom", n, b, k) for n in (6, 12, 18, 24) for b in (0.0, 0.05) for k in (1, 2)]
    rows = []
    for rule in grid:
        h = G.signal(gk_m, *rule)
        s = alone(h)
        a1, a2, a3 = cm(s.loc["1972":]), cm(s.loc["1980-01":"1999-12"]), cm(s.loc["2000-01":])
        hm = G.signal(m_daily_k, *rule)
        pl, p10 = pb.stats(p1_krw(hm)), pb.stats(p1_krw(hm, "2016-11-07", "2016-11-07"))
        rows.append({"rule": f"{rule[0]}{rule[1]} b{rule[2]:.0%} k{rule[3]}", "buys72": int((h.loc["1972":].diff() == 1).sum()),
                     "K 72~": a1[0], "K MDD 72~": a1[1], "K 80~99": a2[0], "K MDD 80~99": a2[1],
                     "K 00~": a3[0], "K MDD 00~": a3[1], "P1K dCAGR": pl["CAGR"] - b_l["CAGR"], "P1K MDD": pl["MDD"],
                     "P1K Sharpe": pl["Sharpe"], "P1K 10y dCAGR": p10["CAGR"] - b_10["CAGR"], "P1K 10y MDD": p10["MDD"]})
    df = pd.DataFrame(rows).set_index("rule")
    df.to_csv(os.path.join(HERE, "gold_filter_grid_krw.csv"), float_format="%.4f")
    fmt = {c: (lambda x: f"{x:.1%}") for c in df.columns if c not in ("buys72", "P1K Sharpe")}
    fmt["P1K Sharpe"] = "{:.2f}".format
    print(df.sort_values("P1K dCAGR", ascending=False).to_string(formatters=fmt))
