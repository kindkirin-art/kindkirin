"""
TQQQ 전략 vs QLD 전략 세후 백테스트 (한국 거주자 기준)

매매 규칙은 backtest.py와 동일. 세금 가정:
  - 해외 상장 ETF(TQQQ, QLD, SCHD, GLD) 매매차익: 연간 실현손익 합산(손실 상계) 후 22%
    (이동평균 단가, 미실현 이익은 과세 이연, 손실 이월 없음, 기본공제 250만원 미반영)
  - 배당(TQQQ, QLD, SCHD 분배금): 15.4% 원천징수 후 재투자
  - SGOV/BIL: 수익 전체를 분배금으로 보고 15.4%
  - 비트코인(현물 직접 보유): 가상자산 과세 2027년 시행 예정이므로 비과세
  - 양도세는 다음 해 5월 말에 SGOV(15% 칸)에서 납부
  - 환율 변동, 금융소득종합과세, 거래비용은 미반영
결과는 '청산 안 함'(마지막 해 미납 세금만 차감)과 '기말 전량 청산'(미실현 이익까지 과세) 두 가지.
"""
import sys

import numpy as np
import pandas as pd
import yfinance as yf

from backtest import LADDER, WEIGHTS, ma_signal

END = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp.today().normalize()
START = END - pd.DateOffset(years=10)
WARMUP = START - pd.DateOffset(years=2)
CGT, DIV_TAX = 0.22, 0.154


def load():
    tickers = ["TQQQ", "QLD", "SCHD", "^IXIC", "GLD", "BTC-USD", "SGOV", "BIL"]
    raw = yf.download(tickers, start=WARMUP, end=END + pd.Timedelta(days=1),
                      auto_adjust=False, actions=True, progress=False)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    return raw["Adj Close"], raw["Close"], raw["Dividends"].fillna(0.0)


def main():
    adj, close, divs = load()
    stock_days = adj[["TQQQ", "QLD", "^IXIC", "GLD"]].dropna().index
    cal = adj["BTC-USD"].dropna().index.union(stock_days)

    s = adj.loc[stock_days]
    r_cash = (s["SGOV"].pct_change().where(s["SGOV"].shift(1).notna(), s["BIL"].pct_change())
              .reindex(cal).fillna(0.0))
    r = {t: adj[t].reindex(cal).ffill().pct_change().fillna(0.0)
         for t in ["TQQQ", "QLD", "SCHD", "GLD", "BTC-USD"]}
    # 배당락일의 배당수익률 (전일 원가격 기준)
    dy = {t: (divs[t] / close[t].ffill().shift(1)).reindex(cal).fillna(0.0)
          for t in ["TQQQ", "QLD", "SCHD", "GLD"]}
    price = {t: adj[t].reindex(cal).ffill() for t in ["TQQQ", "QLD"]}

    btc = adj["BTC-USD"].dropna()
    btc_sma = btc.rolling(120).mean()
    btc_held = ((btc > btc_sma).astype(int).where(btc_sma.notna(), 0)
                .reindex(cal).ffill().fillna(0).shift(1).fillna(0))

    sim_stock = stock_days[(stock_days >= START) & (stock_days <= END)]
    start_day = sim_stock[0]
    sim_days = cal[(cal >= start_day) & (cal <= END)]
    stock_set = set(sim_stock)
    yr = pd.Series(sim_stock, index=sim_stock)
    rebal = {d for d in yr.groupby(yr.index.year).max() if d.month == 12 and d.day >= 24}
    pay_day = {d: d.year for d in yr[yr.index.month == 5].groupby(yr[yr.index.month == 5].index.year).max()}

    def simulate(asset, sig, ladder, taxed=True):
        cgt = CGT if taxed else 0.0
        dtx = DIV_TAX if taxed else 0.0
        rc = r_cash.where(r_cash <= 0, r_cash * (1 - dtx))
        r_btc = pd.Series(np.where(btc_held == 1, r["BTC-USD"], rc), index=cal)
        sig = sig.reindex(cal).ffill().fillna(0)
        p_lev = price[asset]
        tkr = {"lev": asset, "schd": "SCHD", "GLD": "GLD"}
        v = {"lev": 0.0, "schd": 0.0, "GLD": WEIGHTS["GLD"],
             "cash": WEIGHTS["LEV"], "BTC": WEIGHTS["BTC"], "CASH": WEIGHTS["CASH"]}
        b = {"lev": 0.0, "schd": 0.0, "GLD": WEIGHTS["GLD"]}  # 취득가액
        realized, paid = {}, {}
        state, avg, level = 0, np.nan, 0
        eq = []

        def sell(x, amt, d):
            if v[x] <= 0 or amt <= 0:
                return
            f = min(amt / v[x], 1.0)
            realized[d.year] = realized.get(d.year, 0.0) + f * (v[x] - b[x])
            b[x] *= 1 - f
            v[x] -= amt

        def buy(x, amt):
            v[x] += amt
            b[x] += amt

        for d in sim_days:
            if d != start_day:
                for x, t in tkr.items():
                    dv = v[x] * dy[t][d]
                    v[x] *= 1 + r[t][d]
                    v[x] -= dv * dtx
                    b[x] += dv * (1 - dtx)
                v["cash"] *= 1 + rc[d]
                v["CASH"] *= 1 + rc[d]
                v["BTC"] *= 1 + r_btc[d]
            if d in pay_day:
                tax = cgt * max(realized.get(pay_day[d] - 1, 0.0), 0.0)
                v["CASH"] -= tax
                paid[pay_day[d] - 1] = tax
            if d in stock_set:
                p = p_lev[d]
                if state == 0 and sig[d] == 1:
                    amt = v["cash"] + v["schd"]
                    sell("schd", v["schd"], d)
                    v["cash"] = 0.0
                    buy("lev", amt)
                    state, avg, level = 1, p, 0
                elif state == 1 and sig[d] == 0:
                    amt = v["lev"] + v["schd"]
                    sell("lev", v["lev"], d)
                    sell("schd", v["schd"], d)
                    v["cash"] += amt
                    state = 0
                if state == 1:
                    while level < len(ladder) and p / avg - 1 >= ladder[level][0]:
                        amt = v["lev"] * ladder[level][1]
                        sell("lev", amt, d)
                        buy("schd", amt)
                        level += 1
            total = sum(v.values())
            if d in rebal:
                sleeve = v["lev"] + v["schd"] + v["cash"]
                k = WEIGHTS["LEV"] * total / sleeve
                if k > 1 and v["lev"] > 0:
                    avg = (avg + (k - 1) * p_lev[d]) / k
                for x in ("lev", "schd"):
                    (sell(x, v[x] * (1 - k), d) if k < 1 else buy(x, v[x] * (k - 1)))
                v["cash"] *= k
                g = WEIGHTS["GLD"] * total - v["GLD"]
                (buy("GLD", g) if g > 0 else sell("GLD", -g, d))
                v["BTC"] = WEIGHTS["BTC"] * total
                v["CASH"] = WEIGHTS["CASH"] * total
            eq.append(total)

        eq = pd.Series(eq, index=sim_days)
        last = sim_days[-1].year
        owed = cgt * max(realized.get(last, 0.0), 0.0)
        unreal = sum(v[x] - b[x] for x in tkr)
        liq = cgt * max(realized.get(last, 0.0) + unreal, 0.0)
        paid[last] = owed
        return eq, owed, liq, realized, paid

    tqqq_sig = ma_signal(adj["TQQQ"].loc[stock_days])
    ixic_sig = ma_signal(adj["^IXIC"].loc[stock_days])
    specs = {"TQQQ 전략": ("TQQQ", tqqq_sig, LADDER), "QLD 전략": ("QLD", ixic_sig, [])}

    years = (sim_days[-1] - sim_days[0]).days / 365.25

    def summary(e, final):
        dd = (e / e.cummax() - 1).min()
        cagr = final ** (1 / years) - 1
        return cagr, dd, final

    res, rows = {}, {}
    for name, (asset, sig, lad) in specs.items():
        pre = simulate(asset, sig, lad, taxed=False)[0]
        eq, owed, liq, realized, paid = simulate(asset, sig, lad)
        res[name] = (pre, eq, owed, liq, realized, paid)
        for label, e, fin in [("세전", pre, pre.iloc[-1]),
                              ("세후 (청산 안 함)", eq, eq.iloc[-1] - owed),
                              ("세후 (기말 전량 청산)", eq, eq.iloc[-1] - liq)]:
            c, dd, f = summary(e, fin)
            rows[f"{name} {label}"] = {"최종 배수": f"{f:.1f}배", "CAGR": f"{c:.1%}", "MDD": f"{dd:.1%}",
                                      "CAGR/MDD": f"{c / abs(dd):.2f}"}

    pd.set_option("display.width", 200)
    pd.set_option("display.unicode.east_asian_width", True)
    print(f"기간: {sim_days[0].date()} ~ {sim_days[-1].date()}")
    print("\n[성과 요약]")
    print(pd.DataFrame(rows).T.to_string())

    print("\n[세금 합계 (초기 투자금 = 1 기준)]")
    for name, (pre, eq, owed, liq, realized, paid) in res.items():
        print(f"  {name}: 납부 양도세 {sum(paid.values()) - owed:.2f} + 올해 미납 {owed:.2f}, "
              f"기말 청산 시 양도세 {liq:.2f}, 세전-세후 최종자산 차이 {pre.iloc[-1] - eq.iloc[-1] + owed:.2f}"
              f" (청산 시 {pre.iloc[-1] - eq.iloc[-1] + liq:.2f})")

    print("\n[연도별 세후 수익률 / 연중 MDD / 그해 실현이익에 대한 양도세 (그해 초 자산 대비, 다음 해 5월 납부)]  (2026은 청산 안 함 기준)")
    tbl = {}
    for name, (pre, eq, owed, liq, realized, paid) in res.items():
        e = eq.copy()
        e.iloc[-1] -= owed
        prev = e.iloc[0]
        for y, g in e.groupby(e.index.year):
            path = pd.concat([pd.Series([prev]), g.reset_index(drop=True)])
            tbl.setdefault(y, {})[f"{name} 수익률"] = f"{g.iloc[-1] / prev - 1:.1%}"
            tbl[y][f"{name} MDD"] = f"{(path / path.cummax() - 1).min():.1%}"
            tbl[y][f"{name} 양도세"] = f"{paid.get(y, 0.0) / prev:.1%}"
            prev = g.iloc[-1]
    print(pd.DataFrame(tbl).T.to_string())

    print("\n[연도별 실현손익 (초기 투자금 = 1 기준)]")
    print(pd.DataFrame({n: {y: f"{x:+.2f}" for y, x in v[4].items()} for n, v in res.items()}).to_string())


if __name__ == "__main__":
    main()
