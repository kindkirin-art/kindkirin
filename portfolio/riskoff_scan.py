"""Which US-listed ETFs rose while the P1 trend sleeves were out? (USD, total return)

States at the previous close, as in dbc_corr.py:
  TB  = TQQQ switching out and BTC out (2015~)        ALL = TB and gold filter out (BTC counted from 2015)
  T   = TQQQ switching out (each ETF from its own inception)
For each ETF: annualised return on state days vs other days, and episode hit rate (episodes >= 10 days).
"""
import numpy as np
import pandas as pd
import yfinance as yf
import dbc_corr as X

pb, stock = X.pb, X.stock
U = {"cash/short": ["SGOV", "BIL", "SHY"], "treasury": ["IEI", "IEF", "TLT", "EDV", "TIP", "GOVT"],
     "credit/bond": ["AGG", "LQD", "HYG", "EMB", "MUB", "MBB"], "dollar/fx": ["UUP", "FXY", "FXE", "FXF"],
     "managed futures": ["DBMF", "KMLM", "CTA", "WTMF", "FMF"], "hedge": ["TAIL", "BTAL", "VIXY", "SH", "PSQ", "SWAN"],
     "commodity": ["DBC", "PDBC", "USO", "UNG", "DBA", "DBB", "CPER", "SLV", "GLD", "URA"],
     "equity": ["SPY", "QQQ", "XLU", "XLP", "XLV", "XLE", "USMV", "SCHD", "VNQ", "EFA", "EEM", "EWJ", "FXI", "KBWB"]}
tick = [t for v in U.values() for t in v]
px = yf.download(tick, start="1998-01-01", end=pb.END, auto_adjust=True, progress=False)["Close"].reindex(stock)
r = px.pct_change(fill_method=None)
t, g, b = X.tq_in == 0, X.gold_in == 0, X.btc_in == 0
ST = {"TB": (t & b).loc["2015-01-02":], "ALL": (t & g & b.where(b.index >= "2015-01-02", True)), "T": t}


def eps(out, mind=10):
    grp = (out != out.shift()).cumsum()
    res = []
    for _, s in out[out].groupby(grp[out]):
        a, i = s.index[0], stock.get_loc(s.index[-1])
        n = stock[min(i + 1, len(stock) - 1)]
        if (n - a).days >= mind:
            res.append((a, n))
    return res


def scan(out):
    prev = out.shift().fillna(False).astype(bool)
    ep = eps(out)
    rows = {}
    for grp_, ts in U.items():
        for k in ts:
            x = r[k].reindex(out.index).dropna()
            if len(x) < 250:
                continue
            p = prev.reindex(x.index)
            a, o = x[p], x[~p]
            e = [px[k][n] / px[k][s] - 1 for s, n in ep if px[k][s] == px[k][s]]
            rows[k] = {"group": grp_, "from": x.index[0].year, "out days": len(a),
                       "out ann": (1 + a).prod() ** (252 / max(len(a), 1)) - 1, "other ann": (1 + o).prod() ** (252 / len(o)) - 1,
                       "episodes": len(e), "up": sum(v > 0 for v in e), "ep mean": np.mean(e) if e else np.nan,
                       "ep worst": min(e) if e else np.nan, "out daily vol": a.std() * 252 ** 0.5}
    return pd.DataFrame(rows).T.sort_values("out ann", ascending=False)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    f = {c: "{:+.1%}".format for c in ("out ann", "other ann", "ep mean", "ep worst")}
    f["out daily vol"] = "{:.0%}".format
    for k, lab in [("TB", "TQQQ & BTC both out, 2015~"), ("ALL", "TQQQ & gold filter (& BTC from 2015) all out, 1999~"), ("T", "TQQQ out, 1999~")]:
        df = scan(ST[k])
        df.to_csv(f"riskoff_scan_{k}.csv")
        print(f"\n== {lab}: {ST[k].mean():.0%} of days, {len(eps(ST[k]))} episodes >= 10 days")
        print(df.to_string(formatters=f))
    ep = eps(ST["TB"])
    best = ["SGOV", "BIL", "SHY", "IEF", "TLT", "UUP", "FXY", "BTAL", "TAIL", "DBMF", "KMLM", "SH", "XLU", "XLP", "GLD", "DBC", "QQQ"]
    tab = pd.DataFrame({f"{s.date()}~{n.date()}": {k: px[k][n] / px[k][s] - 1 for k in best} for s, n in ep}).T
    print("\nTQQQ & BTC out episodes, % return")
    print((tab * 100).round(1).to_string())
    ep = eps(ST["T"], 60)
    best = ["SHY", "IEF", "TLT", "UUP", "FXY", "BTAL", "SH", "XLU", "XLP", "GLD", "DBC", "QQQ"]
    tab = pd.DataFrame({f"{s.date()}~{n.date()}": {k: px[k][n] / px[k][s] - 1 for k in best} for s, n in ep}).T
    print("\nTQQQ out >= 60 days episodes, % return")
    print((tab * 100).round(1).to_string())
