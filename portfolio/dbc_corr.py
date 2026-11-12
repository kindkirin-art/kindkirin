"""DBC correlation with the P1 assets, and DBC behaviour while the P1 trend sleeves are out.

States (at the previous close): TQQQ switching out (200d 1%, 3 up / 2 down), BTC out (BTC/KRW SMA30 2%),
gold out (optional KRW gold monthly SMA12 5% 2-month filter). BTC state exists from 2015; before that
"all out" means TQQQ and gold out. DBC before 2006-02 is the GSCI-based proxy (dbc_test.py).
"""
import numpy as np
import pandas as pd
import dbc_test as D
import btc_krw_grid as BG
from bootstrap import trend_pos

pb, K, O = D.pb, D.K, D.O
stock = pb.stock
_, log = pb.bt.run(pb.bl.px, FEE=pb.FEE, band=pb.BAND, cash_r=pb.cash_r)
ev = pd.Series({pd.Timestamp(x[0]): (1 if x[1] == "BUY" else 0) for x in log if x[1] in ("BUY", "SELL")})
tq_in = ev.reindex(stock).ffill().fillna(0)
gold_in = O.gold_h.loc[K.me_k.index].reindex(stock).ffill().fillna(1)
bh = pd.Series(trend_pos(BG.px.values, BG.px.rolling(30).mean().values, 0.02, 1), index=BG.px.index)
btc_in = bh.reindex(stock, method="ffill")
dbc_in = D.held_daily(250, 0.02, 1)

fx = K.fx_s
lv = pd.DataFrame({"DBC": (1 + D.dbc_r).cumprod(), "QQQ": pb.bl.px["QQQ"].reindex(stock), "TQQQ": pb.bl.px["TQQQ"].reindex(stock),
                   "Gold": (1 + pb.gold_r).cumprod(), "BTC": pb.btc.reindex(stock, method="ffill"),
                   "SGOV": (1 + pb.cash_r).cumprod(), "USDKRW": fx})
lv.loc[:"2014-12-31", "BTC"] = np.nan


def corr(freq, krw, a, b=None):
    x = lv.loc[a:b]
    if krw:
        x = x.drop(columns="USDKRW").mul(fx.loc[a:b], axis=0).assign(USDKRW=fx.loc[a:b])
    r = (x.resample(freq).last() if freq else x).pct_change(fill_method=None)
    return r.corr()["DBC"].drop("DBC")


def cond(out, start, label):
    out = out.loc[start:]
    prev = out.shift().fillna(False).astype(bool)
    r = D.dbc_r.loc[start:]
    a, b = (1 + r[prev]).prod(), (1 + r[~prev]).prod()
    up = dbc_in.shift().loc[start:][prev].mean()
    print(f"  {label:34s} {prev.mean():4.0%} of days | DBC while out {a ** (252 / prev.sum()) - 1:+6.1%}/yr (total {a - 1:+.0%})"
          f" | otherwise {b ** (252 / (~prev).sum()) - 1:+6.1%}/yr | DBC above its 250d filter {up:.0%} of those days")


def episodes(out, start, mind):
    out = out.loc[start:]
    grp = (out != out.shift()).cumsum()
    rows = []
    for _, s in out[out].groupby(grp[out]):
        a, i = s.index[0], stock.get_loc(s.index[-1])
        n = stock[min(i + 1, len(stock) - 1)]
        if (n - a).days < mind:
            continue
        f = lambda c: lv[c][n] / lv[c][a] - 1
        d = lv["DBC"].loc[a:n]
        rows.append({"from": a.date(), "to": n.date(), "days": (n - a).days, "DBC": f("DBC"), "DBC low": d.min() / d.iloc[0] - 1,
                     "DBC filter at start": "in" if dbc_in[a] else "out", "DBC in-share": dbc_in.loc[a:n].mean(),
                     "QQQ": f("QQQ"), "Gold": f("Gold"), "BTC": f("BTC"), "USDKRW": f("USDKRW")})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    print("== Correlation of DBC returns with each asset")
    tab = {}
    for lab, a in [("1999~", "1999-01-04"), ("2006-02~ (real DBC)", "2006-02-06"), ("2018~", "2018-01-01")]:
        tab[f"{lab} monthly USD"] = corr("ME", False, a)
        tab[f"{lab} monthly KRW"] = corr("ME", True, a)
        tab[f"{lab} weekly USD"] = corr("W", False, a)
    print(pd.DataFrame(tab).T.round(2).to_string())
    m = lv.resample("ME").last().pct_change(fill_method=None).loc["2006-02":]
    for lab, sel in [("QQQ down months", m["QQQ"] < 0), ("QQQ worst 10% months", m["QQQ"] < m["QQQ"].quantile(0.1)),
                     ("Gold down months", m["Gold"] < 0), ("BTC down months (2015~)", m["BTC"] < 0)]:
        print(f"  {lab:26s} n={sel.sum():3d}  DBC mean {m.loc[sel, 'DBC'].mean():+.1%}/month, up in {(m.loc[sel, 'DBC'] > 0).mean():.0%}"
              f"  (all months {m['DBC'].mean():+.1%}, up {(m['DBC'] > 0).mean():.0%})")
    roll = lv[["DBC", "QQQ", "Gold"]].resample("ME").last().pct_change().rolling(36).corr(lv["DBC"].resample("ME").last().pct_change())
    print("  36-month rolling corr with DBC, min/median/max:", {c: tuple(roll[c].dropna().quantile([0, .5, 1]).round(2)) for c in ("QQQ", "Gold")})

    t, g, b = tq_in == 0, gold_in == 0, btc_in == 0
    print("\n== DBC (USD) while P1 sleeves are out")
    cond(t, "1999-01-04", "TQQQ out, 1999~")
    cond(g, "1999-01-04", "Gold filter out, 1999~")
    cond(t & g, "1999-01-04", "TQQQ & gold out, 1999~")
    cond(t, "2006-02-06", "TQQQ out, 2006~")
    cond(b, "2015-01-02", "BTC out, 2015~")
    cond(t & b, "2015-01-02", "TQQQ & BTC out, 2015~")
    cond(t & b & g, "2015-01-02", "TQQQ & BTC & gold out, 2015~")
    allout = (t & g & b.where(b.index >= "2015-01-02", True))
    cond(allout, "1999-01-04", "All out (BTC counted from 2015)")
    pct = lambda x: f"{x:+.1%}" if x == x else "-"
    fm = {c: pct for c in ("DBC", "DBC low", "QQQ", "Gold", "BTC", "USDKRW")}
    fm["DBC in-share"] = "{:.0%}".format
    print("\nAll-out episodes (>= 5 days): TQQQ & gold filter out, plus BTC out from 2015")
    print(episodes(allout, "1999-01-04", 5).to_string(index=False, formatters=fm))
    print("\nTQQQ & BTC both out (gold simply held), 2015~, >= 10 days")
    print(episodes(t & b, "2015-01-02", 10).to_string(index=False, formatters=fm))
    print("\nTQQQ out >= 60 days, 1999~")
    print(episodes(t, "1999-01-04", 60).drop(columns="BTC").to_string(index=False, formatters=fm))
    print("\nNow:", {"TQQQ in": int(tq_in.iloc[-1]), "gold in": int(gold_in.iloc[-1]), "BTC in": int(btc_in.iloc[-1]), "DBC in": int(dbc_in.iloc[-1])},
          "all-out since", allout[~allout].index[-1].date() if allout.iloc[-1] else None)
