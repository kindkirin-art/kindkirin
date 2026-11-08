"""Gold returns while the P1 risk sleeves are out of the market.

- Both out: TQQQ switching sleeve in cash (TQQQ+SCHD sold) AND BTC 120MA sleeve in cash.
  BTC data -> from 2015 (BTC is in P1 from 2016-11 in the 10y test, from 2018 in the long test).
- TQQQ out: TQQQ switching sleeve in cash, 1999~ (for reference).
A day's return counts when the state at the previous close was "out".
"""
import pandas as pd
import portfolio_long as pb
bt = pb.bt

stock = pb.stock
_, log = bt.run(pb.bl.px, FEE=pb.FEE, band=pb.BAND, cash_r=pb.cash_r)
ev = pd.Series({pd.Timestamp(x[0]): (1 if x[1] == "BUY" else 0) for x in log if x[1] in ("BUY", "SELL")})
tq_in = ev.reindex(stock).ffill().fillna(0)
btc_in = pb.held.reindex(stock, method="ffill").fillna(0)
gold = (1 + pb.gold_r).cumprod()
qqq = pb.bl.px["QQQ"].reindex(stock)
btc = pb.btc.reindex(stock, method="ffill")


def episodes(out, start):
    out = out.loc[start:]
    grp = (out != out.shift()).cumsum()
    rows = []
    for _, s in out[out].groupby(grp[out]):
        a, b = s.index[0], s.index[-1]          # out after close a ... back in at close of next change
        nxt = stock[stock.get_loc(b) + 1] if stock.get_loc(b) + 1 < len(stock) else b
        rows.append({"from": a.date(), "to": nxt.date(), "days": (nxt - a).days,
                     "gold": gold[nxt] / gold[a] - 1, "QQQ": qqq[nxt] / qqq[a] - 1,
                     "BTC": btc[nxt] / btc[a] - 1 if btc[a] == btc[a] else float("nan")})
    return pd.DataFrame(rows)


def summary(out, start, label):
    out = out.loc[start:]
    prev = out.shift().fillna(False).astype(bool)
    r = pb.gold_r.loc[start:]
    yrs_in = prev.sum() / 252
    g_out = (1 + r[prev]).prod()
    g_rest = (1 + r[~prev]).prod()
    print(f"\n{label} ({start}~): {prev.mean():.0%} of days")
    print(f"  gold while out : total {g_out - 1:+.1%}, annualized {g_out ** (1 / yrs_in) - 1:+.1%}")
    print(f"  gold otherwise : annualized {g_rest ** (252 / (~prev).sum()) - 1:+.1%}")
    print(f"  gold all days  : annualized {(g_out * g_rest) ** (252 / len(r)) - 1:+.1%}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    fmt = lambda x: f"{x:+.1%}" if x == x else "-"
    both = (tq_in == 0) & (btc_in == 0)
    for start in ("2015-05-14", "2016-11-07", "2018-01-01"):
        summary(both, start, "BOTH OUT (TQQQ switching & BTC)")
    e = episodes(both, "2015-05-14")
    e = e[e.days >= 1]
    print("\nBoth-out episodes since 2015 (from close of 'from' to close of 'to'):")
    print(e.to_string(index=False, formatters={c: fmt for c in ("gold", "QQQ", "BTC")}))
    summary(tq_in == 0, "1999-01-04", "TQQQ switching OUT")
    e2 = episodes(tq_in == 0, "1999-01-04")
    e2 = e2[e2.days >= 30]
    print("\nTQQQ-out episodes >= 30 days since 1999:")
    print(e2.drop(columns="BTC").to_string(index=False, formatters={c: fmt for c in ("gold", "QQQ")}))
