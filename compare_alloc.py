"""
장기보유 자산배분 포트폴리오 비교 (세전 / 세후)

  포트폴리오1: QLD 25 / 금(GLD) 30 / 비트코인 30 / SGOV 15
  포트폴리오2: QLD 40 / SCHD 30 / 금(GLD) 10 / 비트코인 10 / SGOV 10
  - QLD, SCHD, 금: 장기보유
  - 비트코인: 120일선, 대기 없음 (미보유 시 SGOV)
  - SGOV: 2020-06 상장 전은 BIL
  - 연말(마지막 미국 거래일 종가) 리밸런싱

세금 가정은 after_tax.py와 동일 (해외 ETF 양도차익 22% 연간 합산·다음 해 5월 SGOV에서 납부,
배당·SGOV 수익 15.4%, 비트코인 현물 비과세, 기본공제·환율 미반영).

사용법: python compare_alloc.py [종료일] [시작일]
"""
import sys

import numpy as np
import pandas as pd
import yfinance as yf

END = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp.today().normalize()
START = pd.Timestamp(sys.argv[2]) if len(sys.argv) > 2 else END - pd.DateOffset(years=10)
WARMUP = START - pd.DateOffset(years=1)
CGT, DIV_TAX = 0.22, 0.154
ETFS = ["QLD", "SCHD", "GLD"]
PORTFOLIOS = {
    "포트폴리오1": {"QLD": 0.25, "GLD": 0.30, "BTC": 0.30, "CASH": 0.15},
    "포트폴리오2": {"QLD": 0.40, "SCHD": 0.30, "GLD": 0.10, "BTC": 0.10, "CASH": 0.10},
}


def main():
    tickers = ETFS + ["BTC-USD", "SGOV", "BIL"]
    raw = yf.download(tickers, start=WARMUP, end=END + pd.Timedelta(days=1),
                      auto_adjust=False, actions=True, progress=False)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    adj, close, divs = raw["Adj Close"], raw["Close"], raw["Dividends"].fillna(0.0)

    stock_days = adj[ETFS].dropna().index
    cal = adj["BTC-USD"].dropna().index.union(stock_days)
    s = adj.loc[stock_days]
    r_cash = (s["SGOV"].pct_change().where(s["SGOV"].shift(1).notna(), s["BIL"].pct_change())
              .reindex(cal).fillna(0.0))
    r = {t: adj[t].reindex(cal).ffill().pct_change().fillna(0.0) for t in ETFS + ["BTC-USD"]}
    dy = {t: (divs[t] / close[t].ffill().shift(1)).reindex(cal).fillna(0.0) for t in ETFS}

    btc = adj["BTC-USD"].dropna()
    sma = btc.rolling(120).mean()
    btc_held = ((btc > sma).astype(int).where(sma.notna(), 0)
                .reindex(cal).ffill().fillna(0).shift(1).fillna(0))

    sim_stock = stock_days[(stock_days >= START) & (stock_days <= END)]
    start_day = sim_stock[0]
    sim_days = cal[(cal >= start_day) & (cal <= END)]
    yr = pd.Series(sim_stock, index=sim_stock)
    rebal = {d for d in yr.groupby(yr.index.year).max() if d.month == 12 and d.day >= 24}
    may = yr[yr.index.month == 5]
    pay_day = {d: d.year - 1 for d in may.groupby(may.index.year).max()}

    def simulate(w, taxed):
        cgt, dtx = (CGT, DIV_TAX) if taxed else (0.0, 0.0)
        rc = r_cash.where(r_cash <= 0, r_cash * (1 - dtx))
        r_btc = pd.Series(np.where(btc_held == 1, r["BTC-USD"], rc), index=cal)
        v = dict(w)
        b = {x: w[x] for x in w if x in ETFS}
        realized, paid, eq = {}, {}, []
        for d in sim_days:
            if d != start_day:
                for x in b:
                    dv = v[x] * dy[x][d]
                    v[x] *= 1 + r[x][d]
                    v[x] -= dv * dtx
                    b[x] += dv * (1 - dtx)
                v["BTC"] *= 1 + r_btc[d]
                v["CASH"] *= 1 + rc[d]
            if d in pay_day:
                tax = cgt * max(realized.get(pay_day[d], 0.0), 0.0)
                v["CASH"] -= tax
                paid[pay_day[d]] = tax
            total = sum(v.values())
            if d in rebal:
                for x in w:
                    diff = w[x] * total - v[x]
                    if x in b:
                        if diff < 0:
                            f = -diff / v[x]
                            realized[d.year] = realized.get(d.year, 0.0) + f * (v[x] - b[x])
                            b[x] *= 1 - f
                        else:
                            b[x] += diff
                    v[x] += diff
            eq.append(total)
        eq = pd.Series(eq, index=sim_days)
        last = sim_days[-1].year
        owed = cgt * max(realized.get(last, 0.0), 0.0)
        liq = cgt * max(realized.get(last, 0.0) + sum(v[x] - b[x] for x in b), 0.0)
        paid[last] = owed
        return eq, owed, liq, paid

    years = (sim_days[-1] - sim_days[0]).days / 365.25

    def stats(e, final):
        daily = e.pct_change().dropna()
        dd = e / e.cummax() - 1
        t = dd.idxmin()
        cagr = final ** (1 / years) - 1
        vol = daily.std() * np.sqrt(365)
        return {"최종 배수": f"{final:.1f}배", "CAGR": f"{cagr:.1%}", "MDD": f"{dd.min():.1%}",
                "MDD 구간": f"{e.loc[:t].idxmax().date()}~{t.date()}", "변동성": f"{vol:.1%}",
                "샤프": f"{daily.mean() * 365 / vol:.2f}", "CAGR/MDD": f"{cagr / abs(dd.min()):.2f}"}

    def yearly(e):
        out, prev = {}, e.iloc[0]
        for y, g in e.groupby(e.index.year):
            path = pd.concat([pd.Series([prev]), g.reset_index(drop=True)])
            out[y] = (g.iloc[-1] / prev - 1, (path / path.cummax() - 1).min())
            prev = g.iloc[-1]
        return out

    pd.set_option("display.width", 200)
    pd.set_option("display.unicode.east_asian_width", True)
    print(f"기간: {sim_days[0].date()} ~ {sim_days[-1].date()}")
    rows, ytab = {}, {}
    for name, w in PORTFOLIOS.items():
        pre = simulate(w, False)[0]
        eq, owed, liq, paid = simulate(w, True)
        rows[f"{name} 세전"] = stats(pre, pre.iloc[-1])
        rows[f"{name} 세후 (청산 안 함)"] = stats(eq, eq.iloc[-1] - owed)
        rows[f"{name} 세후 (기말 전량 청산)"] = stats(eq, eq.iloc[-1] - liq)
        post = eq.copy()
        post.iloc[-1] -= owed
        for y, (ret, mdd) in yearly(pre).items():
            ytab.setdefault(y, {})[f"{name} 세전"] = f"{ret:.1%}"
            ytab[y][f"{name} MDD"] = f"{mdd:.1%}"
        for y, (ret, _) in yearly(post).items():
            ytab[y][f"{name} 세후"] = f"{ret:.1%}"
        print(f"  {name}: 납부 양도세 {sum(paid.values()) - owed:.2f} + 올해 미납 {owed:.2f}, "
              f"기말 청산 시 {liq:.2f} (초기 투자금 = 1)")
    print("\n[성과 요약]")
    print(pd.DataFrame(rows).T.to_string())
    print("\n[연도별 수익률 / 연중 MDD]  (첫해·마지막해는 부분 기간, 세후는 청산 안 함 기준)")
    print(pd.DataFrame(ytab).T.to_string())


if __name__ == "__main__":
    main()
