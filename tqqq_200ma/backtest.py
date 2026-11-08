"""TQQQ 200MA trend-following backtest, with/without the SQQQ 10% hedge add-on.

Rules
- Signal: TQQQ close vs TQQQ 200-day SMA.
- Buy : 3 consecutive closes above the 200MA while flat -> buy at that close.
- Sell: 2 consecutive closes below the 200MA while invested -> sell everything
        (TQQQ, SCHD, SQQQ) to cash at that close.
- Profit taking (once per trade, return = TQQQ close / TQQQ entry price - 1):
    +20%  -> move 25% of current TQQQ holding to SCHD
    +50%  -> move 25% of current TQQQ holding to SCHD
    +100% -> move 50% of current TQQQ holding to SCHD
    +200% -> move 50% of current TQQQ holding to SCHD
- Add-on (variant B): on entry buy 90% TQQQ + 10% SQQQ. 3 trading days later
  sell the SQQQ and put the proceeds into TQQQ. If a sell signal comes first,
  SQQQ is sold together with everything else.
- Variant C: same with TAIL instead of SQQQ, held 2 trading days (3-day run kept as reference).
- Cash earns 0%. Fills at the close. Optional per-trade cost via FEE.
"""
import sys
import pandas as pd
import yfinance as yf

END = "2026-11-07"
START = "2016-11-07"          # last 10 years
LEVELS = [(0.20, 0.25), (0.50, 0.25), (1.00, 0.50), (2.00, 0.50)]


def load():
    px = yf.download(["TQQQ", "SQQQ", "SCHD", "QQQ", "TAIL"], start="2015-01-01", end=END,
                     auto_adjust=True, progress=False)["Close"]
    px = px.dropna(subset=["TQQQ", "SQQQ", "SCHD", "QQQ"])   # TAIL starts 2017-04
    px["MA200"] = px["TQQQ"].rolling(200).mean()
    return px.loc[START:]


def run(px, hedge=None, FEE=0.0, hold=3, buy_n=3, sell_n=2, band=0.0, atr_k=0.0, cash_r=None):
    """cash_r: optional daily return Series earned by idle cash (e.g. SGOV)."""
    """band: fixed % around the MA. atr_k: band = atr_k * px["ATR"] (price units)."""
    """hedge: None, "SQQQ" or "TAIL" (10% of entry for `hold` days; cash if no data)."""
    cash, sh = 1.0, {t: 0.0 for t in ("TQQQ", "SQQQ", "TAIL", "SCHD") if t in px}
    invested, up, dn = False, 0, 0
    entry, hit, hedge_day = None, set(), None
    eq, log = [], []

    def trade(t, amount, p):          # amount in $, + buy / - sell
        nonlocal cash
        sh[t] += amount / p
        cash -= amount + abs(amount) * FEE

    for i, (d, r) in enumerate(px.iterrows()):
        if cash_r is not None and cash > 0:
            cash *= 1 + cash_r.get(d, 0.0)
        w = r.MA200 * band + (atr_k * r.ATR if atr_k else 0.0)
        up = up + 1 if r.TQQQ > r.MA200 + w else 0
        dn = dn + 1 if r.TQQQ < r.MA200 - w else 0

        if not invested and up >= buy_n:
            budget = cash / (1 + FEE)
            if hedge:
                trade("TQQQ", budget * 0.9, r.TQQQ)
                if pd.notna(r[hedge]):
                    trade(hedge, budget * 0.1, r[hedge])
                hedge_day = i + hold
            else:
                trade("TQQQ", budget, r.TQQQ)
            invested, entry, hit = True, r.TQQQ, set()
            log.append((d.date(), "BUY", round(r.TQQQ, 2)))
        elif invested and dn >= sell_n:
            for t in sh:
                if sh[t]:
                    trade(t, -sh[t] * r[t], r[t])
                    sh[t] = 0.0
            invested, hedge_day = False, None
            log.append((d.date(), "SELL", round(r.TQQQ, 2),
                        f"{r.TQQQ / entry - 1:+.1%}"))
        elif invested:
            if hedge_day is not None and i >= hedge_day:
                v = sh[hedge] * r[hedge] if sh.get(hedge) else 0.0
                if v:
                    trade(hedge, -v, r[hedge]); sh[hedge] = 0.0
                else:                     # no hedge data -> idle 10% cash
                    v = cash / (1 + FEE)
                trade("TQQQ", v * (1 - FEE) / (1 + FEE), r.TQQQ)
                hedge_day = None
            ret = r.TQQQ / entry - 1
            for k, (lvl, frac) in enumerate(LEVELS):
                if k not in hit and ret >= lvl:
                    hit.add(k)
                    v = sh["TQQQ"] * r.TQQQ * frac
                    trade("TQQQ", -v, r.TQQQ)
                    trade("SCHD", v * (1 - FEE) / (1 + FEE), r.SCHD)
                    log.append((d.date(), f"TP+{lvl:.0%}", f"{frac:.0%}->SCHD"))
        eq.append(cash + sum(sh[t] * r[t] for t in sh if sh[t]))
    return pd.Series(eq, index=px.index), log


def stats(s):
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    dd = s / s.cummax() - 1
    rets = s.pct_change().dropna()
    return {"Final x": s.iloc[-1] / s.iloc[0], "CAGR": (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1,
            "MDD": dd.min(), "Sharpe": rets.mean() / rets.std() * 252 ** 0.5,
            "Calmar": ((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1) / -dd.min()}


if __name__ == "__main__":
    FEE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0   # e.g. 0.001 = 0.1%
    px = load()
    A, logA = run(px, None, FEE)
    B, logB = run(px, "SQQQ", FEE)
    C, logC = run(px, "TAIL", FEE, hold=2)
    C3, _ = run(px, "TAIL", FEE, hold=3)
    curves = pd.DataFrame({"A_base": A, "B_sqqq_hedge": B, "C_tail_2d": C, "C_tail_3d_ref": C3,
                           "TQQQ_BH": px.TQQQ / px.TQQQ.iloc[0], "QQQ_BH": px.QQQ / px.QQQ.iloc[0]})
    curves.to_csv(__file__.replace("backtest.py", "equity_curves.csv"), float_format="%.6f")

    print(f"Period {px.index[0].date()} ~ {px.index[-1].date()}  fee={FEE}")
    tbl = pd.DataFrame({c: stats(curves[c]) for c in curves}).T
    print(tbl.to_string(formatters={"Final x": "{:.2f}".format, "CAGR": "{:.1%}".format,
                                    "MDD": "{:.1%}".format, "Sharpe": "{:.2f}".format,
                                    "Calmar": "{:.2f}".format}))
    yr = curves.resample("YE").last()
    yr = pd.concat([curves.iloc[:1], yr]).pct_change().dropna()
    yr.index = yr.index.year
    print("\nYearly returns\n", (yr * 100).round(1).to_string())
    print(f"\nTrades: {sum(1 for x in logA if x[1]=='BUY')} entries")
    for x in logA:
        print(*x)
