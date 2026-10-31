"""
닷컴버블 포함 장기 백테스트 (세전)

포트폴리오: 레버리지 30 / 금 30 / 비트코인 30 / SGOV 10, 연말 리밸런싱. 매매 규칙은 backtest.py와 동일.
  - TQQQ 전략 30%: TQQQ 200일선 3일/2일 + SCHD 익절(+20/50/100/200% → 잔량 25/25/50/50%)
  - QLD 추세추종 30%: ^IXIC 200일선 3일/2일
  - QLD 장기보유 30%

상장 전 구간은 대체 데이터로 이어 붙임 (상장 후는 실제 수정종가):
  - TQQQ(2010-02 전), QLD(2006-06 전): 나스닥100(^NDX) 일간수익률 × 3(2) − 차입비용(T-bill × 2(1)) − 보수 0.95%
    ^NDX는 가격지수(배당 제외)라 실제보다 연 2%p 안팎 낮게 나옴 (2010~2026 검증: 합성 39.5% vs 실제 41.7%)
  - SCHD(2011-10 전): SPY
  - 금: GLD(2004-11~) → 금 선물 GC=F(2000-08~) → 월별 금 가격(data/gold_monthly.csv, 일별 보간)
  - SGOV(2020-06~) → BIL(2007-05~) → 13주 T-bill 금리(^IRX)
  - 비트코인: 2018-01-01부터 편입 (120일선, 대기 없음). 그 전에는 비트코인 30% 자리를 금으로 채움 (금 60%)

사용법: python longrun.py [시작일] [종료일]   (기본 1996-01-01 ~ 오늘)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from backtest import LADDER, ma_signal

START = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp("1996-01-01")
END = pd.Timestamp(sys.argv[2]) if len(sys.argv) > 2 else pd.Timestamp.today().normalize()
W = {"LEV": 0.30, "GLD": 0.30, "BTC": 0.30, "CASH": 0.10}
FEE = 0.0095
BTC_FROM = pd.Timestamp("2018-01-01")


def load():
    tickers = ["^NDX", "^IXIC", "^IRX", "TQQQ", "QLD", "SCHD", "SPY", "GLD", "GC=F",
               "BTC-USD", "SGOV", "BIL"]
    px = yf.download(tickers, start="1990-01-01", end=END + pd.Timedelta(days=1),
                     auto_adjust=True, progress=False)["Close"]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    return px


def splice(*series):
    """수익률 기준으로 앞 시리즈를 뒤 시리즈의 시작점까지 채움. series는 오래된 것부터."""
    out = series[0].pct_change()
    for s in series[1:]:
        r = s.pct_change()
        first = s.first_valid_index()
        out = out.where(out.index <= first, r) if first is not None else out
    return out


def main():
    px = load()
    days = px["^NDX"].dropna().index
    days = days[days <= END]
    px = px.reindex(days)
    gap = pd.Series(days, index=days).diff().dt.days.fillna(1)
    irx = px["^IRX"].ffill() / 100

    # 현금: T-bill → BIL → SGOV
    r_tbill = irx.shift(1) * gap / 365
    r_cash = r_tbill.copy()
    for t in ["BIL", "SGOV"]:
        r = px[t].pct_change()
        ok = px[t].shift(1).notna() & r.notna()
        r_cash = r_cash.where(~ok, r)
    r_cash = r_cash.fillna(0.0)

    # 레버리지 ETF: 합성 → 실제
    r_ndx = px["^NDX"].pct_change()
    lev = {}
    for t, L in [("TQQQ", 3), ("QLD", 2)]:
        syn = L * r_ndx - ((L - 1) * irx.shift(1) + FEE) * gap / 365
        real = px[t].pct_change()
        ok = px[t].shift(1).notna() & real.notna()
        lev[t] = syn.where(~ok, real).fillna(0.0)
    price = {t: (1 + lev[t]).cumprod() for t in lev}

    # 금: 월별 → GC=F → GLD
    gm = pd.read_csv(Path(__file__).parent / "data" / "gold_monthly.csv")
    gm.index = pd.to_datetime(gm["Date"]) + pd.offsets.MonthEnd(0)
    g_month = np.log(gm["Price"]).reindex(gm.index.union(days)).interpolate("time").reindex(days)
    r_gold = np.exp(g_month).pct_change()
    for t in ["GC=F", "GLD"]:
        r = px[t].ffill().pct_change()
        ok = px[t].shift(1).notna()
        r_gold = r_gold.where(~ok, r)
    r_gold = r_gold.fillna(0.0)

    # SCHD: SPY → SCHD
    r_schd = px["SPY"].ffill().pct_change()
    ok = px["SCHD"].shift(1).notna()
    r_schd = r_schd.where(~ok, px["SCHD"].pct_change()).fillna(0.0)

    # 비트코인: 주말 포함 달력에서 120일선, 주식 거래일 기준으로 묶어서 반영
    btc = yf.download("BTC-USD", start="2014-01-01", end=END + pd.Timedelta(days=1),
                               auto_adjust=True, progress=False)["Close"].squeeze()
    btc.index = pd.to_datetime(btc.index).tz_localize(None)
    sma = btc.rolling(120).mean()
    held = (btc > sma).astype(int).where(sma.notna(), 0).shift(1).fillna(0)
    r_btc_cal = pd.Series(np.where(held == 1, btc.pct_change().fillna(0.0), np.nan), index=btc.index)
    # 주식 거래일 사이의 BTC 보유일 수익률을 누적, 미보유일은 현금
    grp = pd.Series(days, index=days).reindex(btc.index, method="bfill")
    btc_part = (1 + r_btc_cal.fillna(0.0)).groupby(grp).prod() - 1
    held_frac = held.groupby(grp).mean()
    r_btc = (btc_part.reindex(days).fillna(0.0)
             + (1 - held_frac.reindex(days).fillna(0.0)) * r_cash)
    r_btc = r_btc.where(days >= BTC_FROM, r_gold)  # 편입 전에는 금

    sim = days[(days >= START)]
    start_day = sim[0]
    yr = pd.Series(sim, index=sim)
    rebal = {d for d in yr.groupby(yr.index.year).max() if d.month == 12 and d.day >= 24}

    def simulate(asset, sig, ladder):
        sig = sig.reindex(days).fillna(0)
        p = price[asset]
        a = {"lev": 0.0, "schd": 0.0, "cash": W["LEV"], "GLD": W["GLD"], "BTC": W["BTC"], "CASH": W["CASH"]}
        state, avg, level, eq, n = 0, np.nan, 0, [], {"매수": 0, "매도": 0, "익절": 0}
        for d in sim:
            if d != start_day:
                a["lev"] *= 1 + lev[asset][d]
                a["schd"] *= 1 + r_schd[d]
                a["cash"] *= 1 + r_cash[d]
                a["GLD"] *= 1 + r_gold[d]
                a["BTC"] *= 1 + r_btc[d]
                a["CASH"] *= 1 + r_cash[d]
            if state == 0 and sig[d] == 1:
                a["lev"] += a["cash"] + a["schd"]
                a["cash"] = a["schd"] = 0.0
                state, avg, level = 1, p[d], 0
                n["매수"] += 1
            elif state == 1 and sig[d] == 0:
                a["cash"] += a["lev"] + a["schd"]
                a["lev"] = a["schd"] = 0.0
                state = 0
                n["매도"] += 1
            if state == 1:
                while level < len(ladder) and p[d] / avg - 1 >= ladder[level][0]:
                    amt = a["lev"] * ladder[level][1]
                    a["lev"] -= amt
                    a["schd"] += amt
                    level += 1
                    n["익절"] += 1
            total = sum(a.values())
            if d in rebal:
                k = W["LEV"] * total / (a["lev"] + a["schd"] + a["cash"])
                if k > 1 and a["lev"] > 0:
                    avg = (avg + (k - 1) * p[d]) / k
                for x in ("lev", "schd", "cash"):
                    a[x] *= k
                for x in ("GLD", "BTC", "CASH"):
                    a[x] = W[x] * total
            eq.append(total)
        return pd.Series(eq, index=sim), n

    hold = pd.Series(1, index=days)
    tq_sig = ma_signal(price["TQQQ"])
    ix_sig = ma_signal(px["^IXIC"].ffill())
    runs = {
        "TQQQ 전략 30%": simulate("TQQQ", tq_sig, LADDER),
        "TQQQ 30% 익절 없음": simulate("TQQQ", tq_sig, []),
        "QLD 추세추종 30%": simulate("QLD", ix_sig, []),
        "QLD 장기보유 30%": simulate("QLD", hold, []),
        "TQQQ 장기보유 30%": simulate("TQQQ", hold, []),
    }
    ndx = px["^NDX"].loc[sim]
    curves = {k: v[0] for k, v in runs.items()}
    curves["(참고) 나스닥100 지수"] = ndx / ndx.iloc[0]

    def stats(e):
        daily = e.pct_change().dropna()
        years = (e.index[-1] - e.index[0]).days / 365.25
        cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1
        dd = e / e.cummax() - 1
        t = dd.idxmin()
        peak = e.loc[:t].idxmax()
        rec = e.loc[t:][e.loc[t:] >= e.loc[peak]]
        vol = daily.std() * np.sqrt(252)
        return {"최종 배수": f"{e.iloc[-1] / e.iloc[0]:,.1f}배", "CAGR": f"{cagr:.1%}", "MDD": f"{dd.min():.1%}",
                "MDD 구간": f"{peak.date()}~{t.date()}",
                "회복": rec.index[0].date().isoformat() if len(rec) else "미회복",
                "변동성": f"{vol:.1%}", "샤프": f"{daily.mean() * 252 / vol:.2f}",
                "CAGR/MDD": f"{cagr / abs(dd.min()):.2f}"}

    def yearly(e):
        out, prev = {}, e.iloc[0]
        for y, g in e.groupby(e.index.year):
            path = pd.concat([pd.Series([prev]), g.reset_index(drop=True)])
            out[y] = f"{g.iloc[-1] / prev - 1:.1%} / {(path / path.cummax() - 1).min():.1%}"
            prev = g.iloc[-1]
        return out

    pd.set_option("display.width", 250)
    pd.set_option("display.unicode.east_asian_width", True)
    print(f"기간: {sim[0].date()} ~ {sim[-1].date()}")
    for k, (_, n) in runs.items():
        print(f"  {k} 매매: {n}")
    print("\n[성과 요약]")
    print(pd.DataFrame({k: stats(v) for k, v in curves.items()}).T.to_string())
    print("\n[구간별 CAGR / MDD]")
    seg = {"닷컴 상승 1996~1999": ("1996", "1999"), "닷컴 붕괴 2000~2002": ("2000", "2002"),
           "회복 2003~2007": ("2003", "2007"), "금융위기 2008": ("2008", "2008"),
           "2009~2016": ("2009", "2016"), "2017~": ("2017", "2026")}
    rows = {}
    for sname, (a, b) in seg.items():
        for k, e in curves.items():
            prev = e.loc[:f"{int(a) - 1}-12-31"]
            base = prev.iloc[-1] if len(prev) else e.iloc[0]
            part = pd.concat([pd.Series([base]), e.loc[a:b].reset_index(drop=True)])
            if len(part) < 2:
                continue
            yrs = len(part) / 252
            rows.setdefault(k, {})[sname] = (f"{(part.iloc[-1] / base) ** (1 / yrs) - 1:.0%} / "
                                             f"{(part / part.cummax() - 1).min():.0%}")
    print(pd.DataFrame(rows).to_string())
    print("\n[연도별 수익률 / 연중 MDD]")
    print(pd.DataFrame({k: yearly(v) for k, v in curves.items()}).to_string())
    pd.DataFrame(curves).to_csv("equity_curves_longrun.csv", float_format="%.6f")


if __name__ == "__main__":
    main()
