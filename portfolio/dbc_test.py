"""Does a broad commodity ETF (DBC; PDBC is the same index without K-1/PTP) help P1? (KRW)

Base = current P1 rules: TQQQ switching 40 / gold held 30 / BTC KRW SMA30 2% 20 / SGOV 10
(BTC weight goes to TQQQ before 2018, as in the other long tests).
DBC: real ETF from 2006-02-06. Before that a proxy from the S&P GSCI spot index (^SPGSCI):
  r = T-bill + a + b * r_GSCI, with a, b fitted on DBC excess returns 2006~2026
  (b ~0.76, a ~-1%/yr, weekly correlation 0.95). The proxy understates the 1999~2000 GSCI total return.
1) DBC alone and its correlation with the P1 sleeves.
2) Weight x funding-source grid with DBC simply held.
3) Trend filters on the DBC sleeve (USD adjusted close; SGOV when out; fee 0.1%).
"""
import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import p1_krw_optimized as O
from bootstrap import trend_pos

pb, K, rs, G, C = O.pb, O.K, O.rs, O.G, O.C
HERE = os.path.dirname(os.path.abspath(__file__))
FEE = 0.001
stock, cal = pb.stock, pb.sleeves.index

raw = yf.download(["DBC", "^SPGSCI", "^IRX"], start="1990-01-01", end=pb.END, auto_adjust=True, progress=False)["Close"]
rf = raw["^IRX"].ffill() / 100 / 252
rr = raw[["DBC", "^SPGSCI"]].pct_change(fill_method=None)
fit = pd.concat([rr, rf.rename("rf")], axis=1).dropna()
X = np.column_stack([np.ones(len(fit)), fit["^SPGSCI"]])
COEF, *_ = np.linalg.lstsq(X, (fit["DBC"] - fit["rf"]).values, rcond=None)
proxy_r = (rf + COEF[0] + COEF[1] * rr["^SPGSCI"]).dropna()
t0 = raw["DBC"].first_valid_index()
dbc_r_full = rr["DBC"].where(rr.index > t0, proxy_r).dropna()          # 1990~, USD
dbc_full = (1 + dbc_r_full).cumprod()                                  # signal series with warm-up
dbc_r = dbc_r_full.reindex(stock).fillna(0.0)


def held_daily(n, band, k):
    h = trend_pos(dbc_full.values, dbc_full.rolling(n).mean().values, band, k, start_pos=1)
    return pd.Series(h, index=dbc_full.index).reindex(stock).ffill()


def held_monthly(kind, n, band, k):
    me = dbc_full.groupby([dbc_full.index.year, dbc_full.index.month]).tail(1)
    return G.signal(me, kind, n, band, k).reindex(stock).ffill().fillna(1)


def sleeve_usd(held=None):
    if held is None:
        return (1 + dbc_r).cumprod()
    prev = held.shift().fillna(1)
    return (1 + dbc_r * prev + (1 - prev) * pb.cash_r - (held != prev) * FEE).cumprod()


def base_sleeves(gold_filter=False):
    sl = K.SLK.copy()
    e = O.btc30.reindex(sl.index).ffill().fillna(1.0)
    sl["BTC"] = e / e.iloc[0]
    if gold_filter:
        h = O.gold_h.loc[K.me_k.index].reindex(stock).ffill().fillna(1)
        prev = h.shift().fillna(1)
        s = (1 + K.gold_krw_r * prev + (1 - prev) * K.cash_krw_d - (h != prev) * G.FEE).cumprod().reindex(sl.index).ffill()
        sl["GLD"] = s / s.iloc[0]
    return sl


def add(sl, name, usd_level):
    s = usd_level.reindex(cal).ffill() * K.fx_cal
    sl[name] = s / s.iloc[0]


PRE, POST = K.P1


def weights(w, src, name="DBC"):
    """Take weight w for DBC from src ('GLD', 'SGOV', 'TQQQ_SW', 'BTC', 'pro' = pro rata, 'G+S' = half gold half SGOV)."""
    out = []
    for base in (PRE, POST):
        x = dict(base)
        if src == "pro":
            x = {k: v * (1 - w) for k, v in x.items()}
        elif src == "G+S":
            x["GLD"] -= w / 2
            x["SGOV"] -= w / 2
        elif src == "BTC" and "BTC" not in x:
            x["TQQQ_SW"] -= w
        else:
            x[src] -= w
        x[name] = w
        assert min(x.values()) >= -1e-9 and abs(sum(x.values()) - 1) < 1e-9
        out.append(x)
    return out


def row(sl, w):
    rs.SL = sl
    lg = rs.fixed(*w)
    ty = rs.fixed(*w, "2016-11-07", "2016-11-07")
    m, p18, m10 = C.metrics(lg), pb.stats(lg.loc["2018-01-01":]), pb.stats(ty)
    x = lg.loc[lg.index.isin(stock)]
    yr = pd.concat([x.iloc[:1], x.resample("YE").last()]).pct_change().dropna()
    yr.index = yr.index.year
    return {"CAGR": m["CAGR"], "MDD": m["MDD"], "Vol": m["Vol"], "Sharpe": m["Sharpe"], "Worst 1y": m["Worst 1y roll"],
            "18~ CAGR": p18["CAGR"], "18~ MDD": p18["MDD"], "18~ Sh": p18["Sharpe"],
            "10y CAGR": m10["CAGR"], "10y MDD": m10["MDD"], "10y Sh": m10["Sharpe"],
            "2008": yr[2008], "2022": yr[2022], "2026": yr[2026]}


def show(rows):
    df = pd.DataFrame(rows).T
    fmt = {c: "{:.1%}".format for c in df.columns if "Sh" not in c}
    fmt.update({c: "{:.2f}".format for c in df.columns if "Sh" in c})
    print(df.to_string(formatters=fmt))
    return df


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(f"proxy fit: alpha {COEF[0] * 252:.2%}/yr, beta {COEF[1]:.3f}; real DBC from {t0.date()}")
    sl = base_sleeves()
    add(sl, "DBC", sleeve_usd())

    if part in ("all", "1"):
        print("\n== 1) DBC held, alone (USD / KRW)")
        k = sl["DBC"]
        for a, b in [("1999-01-04", None), ("1999-01-04", "2008-06-30"), ("2008-07-01", "2020-04-30"),
                     ("2020-05-01", None), ("2018-01-01", None)]:
            u, kk = pb.stats(sleeve_usd().loc[a:b]), pb.stats(k.loc[a:b])
            print(f"  {a[:7]}~{(b or 'now')[:7]}: USD {u['CAGR']:6.1%} / {u['MDD']:6.1%}   KRW {kk['CAGR']:6.1%} / {kk['MDD']:6.1%}  vol {kk['Vol']:.1%}")
        m = sl.loc[sl.index.isin(stock), ["TQQQ_SW", "GLD", "BTC", "SGOV", "DBC"]].resample("ME").last().pct_change().dropna()
        for lab, a in [("1999~", "1999"), ("2018~", "2018")]:
            print(f"  monthly return correlation (KRW) {lab}:")
            print(m.loc[a:].corr().round(2).to_string())
        y = sl.loc[sl.index.isin(stock), ["TQQQ_SW", "GLD", "BTC", "SGOV", "DBC"]]
        yr = pd.concat([y.iloc[:1], y.resample("YE").last()]).pct_change().dropna()
        yr.index = yr.index.year
        rs.SL = sl
        b0 = rs.fixed(PRE, POST)
        b0 = b0.loc[b0.index.isin(stock)]
        yr["P1"] = pd.concat([b0.iloc[:1], b0.resample("YE").last()]).pct_change().dropna().values
        print("  yearly % (KRW):")
        print((yr * 100).round(0).astype(int).T.to_string())
        neg = yr[yr["P1"] < 0.05]
        print(f"  P1 weak years (<5%): DBC mean {neg['DBC'].mean():.1%}, gold mean {neg['GLD'].mean():.1%}, SGOV mean {neg['SGOV'].mean():.1%}")

    if part in ("all", "2"):
        print("\n== 2) DBC held: weight x funding source (long 1999~ | 2018~ | 10y BTC from start)")
        rows = {"base P1": row(sl, (PRE, POST))}
        for src in ("GLD", "SGOV", "G+S", "TQQQ_SW", "BTC", "pro"):
            for w in (0.05, 0.10, 0.15, 0.20):
                if src == "SGOV" and w > 0.10:
                    continue
                rows[f"DBC {w:.0%} from {src}"] = row(sl, weights(w, src))
        show(rows).to_csv(os.path.join(HERE, "dbc_weight_grid.csv"), float_format="%.4f")

    if part in ("all", "3"):
        print("\n== 3) DBC trend filters: alone (USD, 1999~) and inside P1 at 10% from gold / 10% from TQQQ")
        rules = [("hold",)]
        rules += [("d", n, b, k) for n in (100, 150, 200, 250) for b in (0.0, 0.01, 0.02, 0.03) for k in (1, 3)]
        rules += [("sma", n, b, k) for n in (6, 8, 10, 12) for b in (0.0, 0.02, 0.05) for k in (1, 2)]
        rules += [("mom", n, 0.0, 1) for n in (6, 9, 12)]
        base = row(sl, (PRE, POST))
        out = {}
        for r in rules:
            h = None if r[0] == "hold" else held_daily(*r[1:]) if r[0] == "d" else held_monthly(*r)
            s = sleeve_usd(h)
            add(sl, "DF", s)
            st = pb.stats(s)
            p2 = [pb.stats(s.loc[a:b]) for a, b in (("2008-07-01", "2020-04-30"), ("2020-05-01", None))]
            g, t = row(sl, weights(0.10, "GLD", "DF")), row(sl, weights(0.10, "TQQQ_SW", "DF"))
            out[" ".join(map(str, r))] = {
                "trades/yr": 0 if h is None else (h.diff().abs().sum() / 27.8), "in%": 1 if h is None else h.mean(),
                "CAGR": st["CAGR"], "MDD": st["MDD"], "Sharpe": st["Sharpe"],
                "08-20": p2[0]["CAGR"], "08-20 MDD": p2[0]["MDD"], "20~": p2[1]["CAGR"],
                "G dCAGR": g["CAGR"] - base["CAGR"], "G MDD": g["MDD"], "G Sharpe": g["Sharpe"], "G 18~Sharpe": g["18~ Sh"],
                "T dCAGR": t["CAGR"] - base["CAGR"], "T MDD": t["MDD"], "T Sharpe": t["Sharpe"], "T 18~Sharpe": t["18~ Sh"]}
        df = pd.DataFrame(out).T
        df.to_csv(os.path.join(HERE, "dbc_filter_grid.csv"), float_format="%.4f")
        fmt = {c: "{:.1%}".format for c in df.columns if "Sharpe" not in c and c != "trades/yr"}
        fmt.update({c: "{:.2f}".format for c in df.columns if "Sharpe" in c})
        fmt["trades/yr"] = "{:.1f}".format
        print(f"base P1: CAGR {base['CAGR']:.1%} MDD {base['MDD']:.1%} Sharpe {base['Sharpe']:.2f} 18~Sharpe {base['18~ Sh']:.2f}")
        print(df.to_string(formatters=fmt))
