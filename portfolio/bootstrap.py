"""Block-bootstrap comparison of A (original P1, KRW), E (optimised P1, KRW) and P2 (KRW).

Data (daily calendar incl. weekends, 2014-09 ~ 2026-11; base window limited by BTC history):
TQQQ, SCHD, US T-bill, gold (USD), USD/KRW, BTC-USD, BTC/KRW (Upbit spliced), KRW short rate.
Blocks of BLOCK calendar days (multiple of 7 -> weekdays/holidays preserved) are drawn jointly for all
assets and chained into a synthetic path; strategies are re-run on each path:
- TQQQ switching: SMA200 (trading days), 1% band, buy 3 / sell 2, SCHD ladder, fee 0.1%, USD cash.
- BTC A/P2: BTC-USD SMA120 (calendar days), 1% band, fee 0.1%, USD cash.
- BTC E: BTC/KRW SMA30, 2% band, fee 0.05%, KRW cash.
- Gold A/P2: held. Gold E: KRW gold month-end vs SMA12 (months), 5% band, 2-month confirm, KRW cash.
Everything is valued in KRW; annual rebalancing at each synthetic year's last US trading day.
P1 = TQQQ 40 / Gold 30 / BTC 20 / SGOV 10, P2 = TQQQ 50 / BTC 30 / SGOV 20.
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import portfolio_long as pb
import gold_krw as K
import btc_krw_grid as BG

BLOCK = int(sys.argv[1]) if len(sys.argv) > 1 else 182
NPATH = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
YEARS, WARM = 10, 1.5
LEVELS = [(0.20, 0.25), (0.50, 0.25), (1.00, 0.50), (2.00, 0.50)]

# ---------- data on a daily calendar ----------
btc_usd = yf.download("BTC-USD", start="2014-01-01", end=pb.END, auto_adjust=True, progress=False)["Close"].squeeze().dropna()
cal = btc_usd.index[btc_usd.index >= "2014-09-18"]
cal = cal[cal < pb.END]
is_stock = cal.isin(pb.stock)
ff = lambda s: s.reindex(cal.union(s.index)).ffill().reindex(cal)
lv = pd.DataFrame({
    "TQQQ": ff(pb.bl.px["TQQQ"]), "SCHD": ff(pb.bl.px["SCHD"]), "GOLD": ff((1 + pb.gold_r).cumprod()),
    "FX": ff(K.fx_d), "BTCU": btc_usd.reindex(cal), "BTCK": BG.px.reindex(cal).ffill(),
})
R = lv.pct_change().fillna(0.0)
R["CASHU"] = pb.cash_r.reindex(cal).fillna(0.0).values          # 0 on non-trading days
R["CASHK"] = ff(K.rate_m / 100 / 365).fillna(0.0).values
R["STOCK"] = is_stock.astype(float)
COLS = list(R.columns)
RA = R.values
WD = cal.weekday.values


def sma(x, n):
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full(len(x), np.nan)
    out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def trend_pos(price, ma, band, k, start_pos=0):
    pos, up, dn = start_pos, 0, 0
    held = np.empty(len(price))
    for i in range(len(price)):
        m = ma[i]
        if m == m:
            up = up + 1 if price[i] > m * (1 + band) else 0
            dn = dn + 1 if price[i] < m * (1 - band) else 0
            if pos == 0 and up >= k:
                pos = 1
            elif pos == 1 and dn >= k:
                pos = 0
        held[i] = pos
    return held


def sleeve_trend(ret, price, ma, band, k, cash, fee):
    held = trend_pos(price, ma, band, k)
    prev = np.concatenate([[0.0], held[:-1]])
    r = ret * prev + (1 - prev) * cash - (held != prev) * fee
    return np.cumprod(1 + r)


def sleeve_tqqq(t, s, cashu, stock):
    idx = np.where(stock > 0)[0]
    ts, ss = t[idx], s[idx]
    ma = sma(ts, 200)
    cash, st, sh = 1.0, 0.0, 0.0
    inv, up, dn, entry, hit = False, 0, 0, 0.0, 0
    eq = np.empty(len(t))
    j = -1
    for i in range(len(t)):
        if cash > 0:
            cash *= 1 + cashu[i]
        if stock[i] > 0:
            j += 1
            p, m = ts[j], ma[j]
            if m == m:
                up = up + 1 if p > m * 1.01 else 0
                dn = dn + 1 if p < m * 0.99 else 0
                if not inv and up >= 3:
                    st, cash = cash / 1.001 / p, 0.0
                    inv, entry, hit = True, p, 0
                elif inv and dn >= 2:
                    cash += (st * p + sh * ss[j]) * 0.999
                    st = sh = 0.0
                    inv = False
                elif inv:
                    while hit < 4 and p / entry - 1 >= LEVELS[hit][0]:
                        v = st * p * LEVELS[hit][1]
                        st -= v / p
                        sh += v * 0.999 / 1.001 / ss[j]
                        hit += 1
        eq[i] = cash + st * t[i] + sh * s[i]
    return eq


def gold_filter(goldk, gret, cashk, stock, months):
    """KRW gold, month-end SMA12 5% band 2-month confirm; months: array of month ids per day."""
    idx = np.where(stock > 0)[0]
    last = {}
    for i in idx:
        last[months[i]] = i
    me = np.array(sorted(last.values()))
    mp = goldk[me]
    held_m = trend_pos(mp, sma(mp, 12), 0.05, 2, start_pos=1)
    held = np.ones(len(goldk))
    for a, h in zip(me, held_m):
        held[a:] = h
    prev = np.concatenate([[1.0], held[:-1]])
    r = gret * prev + (1 - prev) * cashk - (held != prev) * 0.001
    return np.cumprod(1 + r)


def portfolios(A, start="2000-01-01"):
    """A: return matrix (days x COLS) of a path. Returns dict of KRW equity arrays."""
    g = {c: A[:, i] for i, c in enumerate(COLS)}
    lvl = lambda r: np.cumprod(1 + r)
    t, s, gold, fx = lvl(g["TQQQ"]), lvl(g["SCHD"]), lvl(g["GOLD"]), lvl(g["FX"])
    bu, bk = lvl(g["BTCU"]), lvl(g["BTCK"])
    goldk = gold * fx
    tq = sleeve_tqqq(t, s, g["CASHU"], g["STOCK"]) * fx
    sgov = lvl(g["CASHU"]) * fx
    btcA = sleeve_trend(g["BTCU"], bu, sma(bu, 120), 0.01, 1, g["CASHU"], 0.001) * fx
    btcE = sleeve_trend(g["BTCK"], bk, sma(bk, 30), 0.02, 1, g["CASHK"], 0.0005)
    days = pd.date_range(start, periods=len(t), freq="D")
    months = (days.year * 12 + days.month).values
    goldE = gold_filter(goldk, np.concatenate([[0.0], goldk[1:] / goldk[:-1] - 1]), g["CASHK"], g["STOCK"], months)
    years = days.year.values
    stock = g["STOCK"] > 0
    reb = np.zeros(len(t), bool)
    for y in np.unique(years):
        ii = np.where((years == y) & stock)[0]
        if len(ii):
            reb[ii[-1]] = True
    w0 = int(WARM * 365)

    def port(sleeves, w):
        sl = np.column_stack(sleeves)[w0:]
        sl = sl / sl[0]
        units = np.array(w) / sl[0]
        out = np.empty(len(sl))
        rb = reb[w0:]
        for i in range(len(sl)):
            v = units @ sl[i]
            out[i] = v
            if rb[i]:
                units = v * np.array(w) / sl[i]
        return out, stock[w0:]

    return {"A": port([tq, goldk, btcA, sgov], [.4, .3, .2, .1]),
            "E": port([tq, goldE, btcE, sgov], [.4, .3, .2, .1]),
            "P2": port([tq, btcA, sgov], [.5, .3, .2])}


def metrics(eq, stock):
    x = eq[stock]
    yrs = YEARS
    cagr = x[-1] ** (1 / yrs) - 1
    mdd = (x / np.maximum.accumulate(x) - 1).min()
    r = x[1:] / x[:-1] - 1
    return cagr, mdd, r.mean() / r.std() * np.sqrt(252)


def sample_path(rng, n):                                 # weekday-aligned block chaining
    idx = []
    first = rng.integers(0, len(RA) - BLOCK)
    idx.extend(range(first, first + BLOCK))
    while len(idx) < n:
        want = (WD[idx[-1]] + 1) % 7
        cands = np.where(WD[: len(RA) - BLOCK] == want)[0]
        a = rng.choice(cands)
        idx.extend(range(a, a + BLOCK))
    return RA[np.array(idx[:n])]


if __name__ == "__main__":
    n = int((YEARS + WARM) * 365)
    # sanity: actual history, last 10 years
    act = portfolios(RA[-n:], start=cal[-n])
    print("Actual last 10y:", {k: tuple(round(v, 3) for v in metrics(*act[k])) for k in act})
    rng = np.random.default_rng(42)
    res = {k: [] for k in ("A", "E", "P2")}
    for p in range(NPATH):
        out = portfolios(sample_path(rng, n))
        for k in res:
            res[k].append(metrics(*out[k]))
    res = {k: np.array(v) for k, v in res.items()}
    print(f"\nBlock {BLOCK} days, {NPATH} paths of {YEARS} years (KRW)")
    q = [5, 25, 50, 75, 95]
    for name, j in [("CAGR", 0), ("MDD", 1), ("Sharpe", 2)]:
        print(f"\n{name} percentiles {q}")
        for k in res:
            v = np.percentile(res[k][:, j], q)
            print(f"  {k:3s} " + "  ".join(f"{x:7.1%}" if j < 2 else f"{x:6.2f}" for x in v)
                  + f"   mean {res[k][:, j].mean():.1%}" if j < 2 else f"  {k:3s} " + "  ".join(f"{x:6.2f}" for x in v))
    A, E, P = res["A"], res["E"], res["P2"]
    print("\nHead-to-head (share of paths):")
    print(f"  E CAGR > A: {(E[:,0] > A[:,0]).mean():.0%}   E MDD better than A: {(E[:,1] > A[:,1]).mean():.0%}   E Sharpe > A: {(E[:,2] > A[:,2]).mean():.0%}")
    print(f"  P2 CAGR > A: {(P[:,0] > A[:,0]).mean():.0%}  P2 MDD better than A: {(P[:,1] > A[:,1]).mean():.0%}  P2 Sharpe > A: {(P[:,2] > A[:,2]).mean():.0%}")
    print(f"  P2 CAGR > E: {(P[:,0] > E[:,0]).mean():.0%}  P2 MDD better than E: {(P[:,1] > E[:,1]).mean():.0%}  P2 Sharpe > E: {(P[:,2] > E[:,2]).mean():.0%}")
    print("\nTail risk:")
    for k in res:
        v = res[k]
        print(f"  {k:3s} P(CAGR<0) {(v[:,0] < 0).mean():.1%}  P(CAGR<10%) {(v[:,0] < .10).mean():.1%}  "
              f"P(MDD<-30%) {(v[:,1] < -.30).mean():.1%}  P(MDD<-40%) {(v[:,1] < -.40).mean():.1%}")
    d = E[:, 0] - A[:, 0]
    print(f"\nE-A CAGR diff: median {np.median(d):+.1%}, 5%~95% [{np.percentile(d,5):+.1%}, {np.percentile(d,95):+.1%}]")
