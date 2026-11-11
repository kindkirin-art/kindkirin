"""
닷컴버블 포함 장기 백테스트 (1991 ~ 현재)
규칙은 backtest.py와 동일:
  TQQQ 200일선, 3일 연속 위 → 매수 / 2일 연속 아래 → TQQQ·SCHD 전량 매도
  매수가 대비 +20/50/100/200% 도달 시 남은 TQQQ의 25/25/50/50%를 SCHD로
  전량 매도 후 대기자산: SGOV vs GLD

상장 전 구간 대체 데이터
  TQQQ : 합성 = 3 × QQQ 일간 총수익(보수 0.20% 되돌림) − 보수 0.95% − 2 × (T-bill + 스프레드 0.85%)
         QQQ 상장(1999-03-10) 전은 ^NDX 가격지수. 실제 TQQQ(2010-02-11~)와 이어붙임.
         (2010~2026 실제 TQQQ와 비교해 스프레드 보정, 상관계수 0.998)
  SCHD : 2011-10-20 이전 = Vanguard Value Index(VIVAX, 배당재투자)
  SGOV : 2020-06 이전 BIL, 2007-05 이전은 13주 T-bill 금리(^IRX) 일할
  GLD  : 2004-11 이전 금 선물(GC=F), 2000-08 이전은 월평균 금 가격(datasets/gold-prices)을 일간 로그보간
거래비용/세금/슬리피지 미반영.
"""
import os
import sys

import numpy as np
import pandas as pd

from backtest import LADDER, signal, stats

END = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp.today().normalize()
SPREAD = 0.0085
CACHE = os.environ.get("PX_CACHE", "prices_long_cache.csv")
WINDOWS = [
    ("전체 1991~", "1991-01-02"),
    ("1996~ (버블 형성기부터)", "1996-01-02"),
    ("2000-03-10~ (나스닥 정점 진입)", "2000-03-10"),
]
TICKERS = ["^NDX", "QQQ", "TQQQ", "SCHD", "VIVAX", "SGOV", "BIL", "^IRX", "GLD", "GC=F"]


def download():
    if os.path.exists(CACHE):
        return pd.read_csv(CACHE, index_col=0, parse_dates=True)
    import yfinance as yf
    px = yf.download(TICKERS, start="1989-01-01", end=END + pd.Timedelta(days=1),
                     auto_adjust=True, progress=False)["Close"]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    px.to_csv(CACHE)
    return px


def splice(base, *later):
    """base 수익률로 시작해, 각 later 시리즈가 존재하는 날부터 그 수익률로 교체."""
    r = base.pct_change()
    for s in later:
        rs = s.pct_change()
        r = rs.where(s.shift(1).notna() & s.notna(), r)
    return (1 + r.fillna(0)).cumprod()


def load():
    px = download()
    px = px[px["^NDX"].notna()].loc[:END]
    irx = px["^IRX"].ffill()
    rf = irx.shift(1) / 100 / 252

    # 합성 TQQQ
    rq = px["QQQ"].pct_change() + 0.002 / 252
    rq = rq.where(px["QQQ"].shift(1).notna(), px["^NDX"].pct_change())
    syn = 3 * rq - 0.0095 / 252 - 2 * (rf + SPREAD / 252)
    tq = (1 + syn.fillna(0)).cumprod()
    out = pd.DataFrame(index=px.index)
    out["TQQQ"] = splice(tq, px["TQQQ"])
    out["NDX"] = splice(px["^NDX"], px["QQQ"])

    out["SCHD"] = splice(px["VIVAX"].ffill(), px["SCHD"])

    tbill = (1 + rf.fillna(0)).cumprod()
    out["CASH"] = splice(tbill, px["BIL"], px["SGOV"])

    g = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "data", "gold_monthly_1985_2001.csv"))
    g.index = pd.to_datetime(g["Date"]) + pd.offsets.MonthBegin(0) + pd.Timedelta(days=14)
    gd = np.exp(np.log(g["Price"]).reindex(g.index.union(px.index)).interpolate("time")).reindex(px.index)
    out["GLD"] = splice(gd, px["GC=F"].where(px.index < "2004-11-18"), px["GLD"])
    return out


def run(p_all, sig_all, start, park, ladder=True):
    p = p_all.loc[start:]
    sig = sig_all.loc[start:]
    sh = {"TQQQ": 0.0, "SCHD": 0.0, park: 1.0 / p[park].iloc[0]}
    entry, stage, eq, log = None, 0, [], []
    for i, d in enumerate(p.index):
        c = p.loc[d]
        s = sig.iloc[i]
        if s == 1 and sh["TQQQ"] == 0 and sh["SCHD"] == 0:
            v = sh[park] * c[park]
            sh[park], sh["TQQQ"], entry, stage = 0.0, v / c["TQQQ"], c["TQQQ"], 0
            log.append((d, "BUY"))
        elif s == 0 and (sh["TQQQ"] > 0 or sh["SCHD"] > 0):
            v = sh["TQQQ"] * c["TQQQ"] + sh["SCHD"] * c["SCHD"]
            sh["TQQQ"] = sh["SCHD"] = 0.0
            sh[park] = v / c[park]
            log.append((d, "SELL"))
        if ladder and sh["TQQQ"] > 0:
            while stage < len(LADDER) and c["TQQQ"] / entry - 1 >= LADDER[stage][0]:
                q = sh["TQQQ"] * LADDER[stage][1]
                sh["TQQQ"] -= q
                sh["SCHD"] += q * c["TQQQ"] / c["SCHD"]
                log.append((d, f"TP{int(LADDER[stage][0]*100)}"))
                stage += 1
        eq.append(sum(sh[k] * c[k] for k in sh))
    return pd.Series(eq, index=p.index), log


def table(res):
    t = pd.DataFrame({k: stats(v) for k, v in res.items()}).T
    f = t.copy()
    for col in ["CAGR", "MDD", "연변동성"]:
        f[col] = (t[col] * 100).map("{:.1f}%".format)
    f["최종(1달러→)"] = t["최종(1달러→)"].map("{:,.1f}".format)
    f["샤프(rf=0)"] = t["샤프(rf=0)"].map("{:.2f}".format)
    return f


def main():
    p = load()
    sig = signal(p["TQQQ"])
    all_curves = {}
    for name, start in WINDOWS:
        res = {}
        res["전략 + SGOV 대기"], log = run(p, sig, start, "CASH")
        res["전략 + GLD 대기"], _ = run(p, sig, start, "GLD")
        res["(참고) 익절없음 + SGOV"], _ = run(p, sig, start, "CASH", ladder=False)
        res["(참고) 익절없음 + GLD"], _ = run(p, sig, start, "GLD", ladder=False)
        q = p.loc[start:]
        res["TQQQ(합성) 보유"] = q["TQQQ"] / q["TQQQ"].iloc[0]
        res["나스닥100 보유"] = q["NDX"] / q["NDX"].iloc[0]
        res["금 보유"] = q["GLD"] / q["GLD"].iloc[0]
        idx = res["TQQQ(합성) 보유"].index
        print(f"\n=== {name}: {idx[0].date()} ~ {idx[-1].date()} ===")
        print(table(res).to_string())
        if start == WINDOWS[0][1]:
            all_curves = res
            n = {k: sum(x[1].startswith(k) for x in log) for k in ["BUY", "SELL", "TP"]}
            print(f"매매 횟수: 매수 {n['BUY']}, 매도 {n['SELL']}, 익절전환 {n['TP']}")

    curves = pd.DataFrame(all_curves)
    yearly = pd.concat([curves.iloc[[0]], curves.resample("YE").last()]).pct_change().dropna()
    yearly.index = yearly.index.year
    print("\n연도별 수익률(%) — 전체 구간")
    print((yearly[["전략 + SGOV 대기", "전략 + GLD 대기", "TQQQ(합성) 보유", "나스닥100 보유"]] * 100)
          .round(1).to_string())

    # 닷컴 붕괴 구간 상세
    a, b = "2000-03-10", "2002-10-09"
    print(f"\n닷컴 붕괴 구간 ({a} 나스닥 정점 ~ {b} 바닥) 수익률")
    for k in ["전략 + SGOV 대기", "전략 + GLD 대기", "TQQQ(합성) 보유", "나스닥100 보유"]:
        c = curves[k]
        print(f"  {k:18s} {(c.loc[:b].iloc[-1] / c.loc[:a].iloc[-1] - 1) * 100:7.1f}%")
    curves.round(4).to_csv("equity_curves_long.csv")


if __name__ == "__main__":
    main()
