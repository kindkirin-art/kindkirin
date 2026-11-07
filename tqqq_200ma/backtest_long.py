"""Same strategy as backtest.py, tested from the dot-com bubble (1999-01) to today.

TQQQ/SQQQ (2010-02) and SCHD (2011-10) did not exist then, so they are built:
- Underlying: QQQ total return (from 1999-03), ^NDX price return before that.
- Synthetic TQQQ daily = 3*r - 2*rf - 0.95%/252     (borrowing cost on 2x)
- Synthetic SQQQ daily = -3*r + 4*rf - 0.95%/252    (short + interest on collateral)
  rf = 13-week T-bill (^IRX) / 252. Real TQQQ/SQQQ are spliced in from 2010-02-11.
- SCHD proxy before 2011-10-20: Vanguard Equity Income fund (VEIPX).
"""
import sys
import pandas as pd
import yfinance as yf
from backtest import run, stats, END

START = "1999-01-04"
FEE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
ER = 0.0095 / 252

raw = yf.download(["^NDX", "QQQ", "^IRX", "VEIPX", "SCHD", "TQQQ", "SQQQ"],
                  start="1985-01-01", end=END, auto_adjust=True, progress=False)["Close"]
raw = raw.dropna(subset=["^NDX"])
rf = raw["^IRX"].ffill() / 100 / 252
r = raw["QQQ"].pct_change().fillna(raw["^NDX"].pct_change())
r.iloc[0] = 0.0


def lev(k, cost):
    return (1 + k * r - cost).cumprod()


def splice(syn, real):
    """Synthetic before real start, real after (rescaled to join)."""
    real = real.dropna()
    t0 = real.index[0]
    out = syn.copy()
    out.loc[t0:] = real / real.iloc[0] * syn.loc[t0]
    return out


tqqq_syn = lev(3, 2 * rf + ER)
sqqq_syn = lev(-3, -4 * rf + ER)
px = pd.DataFrame({
    "TQQQ": splice(tqqq_syn, raw["TQQQ"]),
    "SQQQ": splice(sqqq_syn, raw["SQQQ"]),
    "SCHD": splice(raw["VEIPX"].ffill(), raw["SCHD"]),
    "QQQ": (1 + r).cumprod(),
})
px["MA200"] = px["TQQQ"].rolling(200).mean()
px = px.loc[START:].dropna()

if __name__ == "__main__":
    # validation of the synthetic series vs real ETFs (2010-02-11 ~)
    v = pd.DataFrame({"syn_T": tqqq_syn, "real_T": raw.TQQQ,
                      "syn_S": sqqq_syn, "real_S": raw.SQQQ}).dropna()
    v = v / v.iloc[0]
    yr = v.resample("YE").last().pct_change().dropna()
    print("Synthetic vs real, yearly %:\n", (yr * 100).round(1).to_string())
    print("Final multiple:", v.iloc[-1].round(2).to_dict(), "\n")

    A, logA = run(px, None, FEE)
    B, logB = run(px, "SQQQ", FEE)
    curves = pd.DataFrame({"A_base": A, "B_sqqq_hedge": B,
                           "TQQQ_BH": px.TQQQ / px.TQQQ.iloc[0],
                           "QQQ_BH": px.QQQ / px.QQQ.iloc[0]})
    curves.to_csv(__file__.replace("backtest_long.py", "equity_curves_long.csv"),
                  float_format="%.6f")
    fmt = {"Final x": "{:,.1f}".format, "CAGR": "{:.1%}".format, "MDD": "{:.1%}".format,
           "Sharpe": "{:.2f}".format, "Calmar": "{:.2f}".format}
    print(f"Period {px.index[0].date()} ~ {px.index[-1].date()}  fee={FEE}")
    print(pd.DataFrame({c: stats(curves[c]) for c in curves}).T.to_string(formatters=fmt))
    for a, b in [("1999-01-04", "2002-12-31"), ("2003-01-01", "2009-12-31"),
                 ("2010-01-01", "2016-11-06"), ("2016-11-07", END)]:
        print(f"\n{a} ~ {b}")
        print(pd.DataFrame({c: stats(curves[c].loc[a:b]) for c in curves}).T
              .to_string(formatters=fmt))
    yr = curves.resample("YE").last()
    yr = pd.concat([curves.iloc[:1], yr]).pct_change().dropna()
    yr.index = yr.index.year
    print("\nYearly returns\n", (yr * 100).round(1).to_string())
    ratio = B / A
    buys = [x[0] for x in logA if x[1] == "BUY"]
    print(f"\nEntries: {len(buys)}; hedge effect over 3 days (B/A):")
    for d in buys:
        i = px.index.get_loc(pd.Timestamp(d))
        if i + 3 < len(px):
            print(" ", d, f"{(ratio.iloc[i + 3] / ratio.iloc[i] - 1) * 100:+.2f}%")
    print()
    for x in logA:
        print(*x)
