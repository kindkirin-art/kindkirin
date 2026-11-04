"""
TQQQ 200일선 추세추종 + 단계별 SCHD 익절 전환 백테스트 (최근 10년)

규칙
  - 신호: TQQQ 종가 vs TQQQ 200일 이동평균
      3일 연속 200일선 위  → 3일째 종가에 대기자산 전액으로 TQQQ 매수
      2일 연속 200일선 아래 → 2일째 종가에 TQQQ·SCHD 전량 매도 → 대기자산
  - 익절 전환(보유 사이클마다 1회씩, 매수가 대비 TQQQ 가격 수익률 기준, 종가 체결)
      +20%  → 보유 TQQQ의 25%를 SCHD로
      +50%  → 남은 TQQQ의 25%를 SCHD로
      +100% → 남은 TQQQ의 50%를 SCHD로
      +200% → 남은 TQQQ의 50%를 SCHD로
  - 대기자산 비교: SGOV(상장 전 2020-06 이전은 BIL로 대체) vs GLD
  - 시작일에 이미 매수 상태면 시작일 종가에 신규 진입한 것으로 처리
  - 거래비용/세금/슬리피지 미반영, 배당 재투자(수정주가)
"""
import sys

import numpy as np
import pandas as pd
import yfinance as yf

END = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp.today().normalize()
START = END - pd.DateOffset(years=10)
WARMUP = START - pd.DateOffset(years=2)
LADDER = [(0.20, 0.25), (0.50, 0.25), (1.00, 0.50), (2.00, 0.50)]


def load():
    px = yf.download(["TQQQ", "SCHD", "SGOV", "BIL", "GLD", "QQQ"], start=WARMUP,
                     end=END + pd.Timedelta(days=1), auto_adjust=True, progress=False)["Close"]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    px = px.dropna(subset=["TQQQ"])
    # SGOV 상장 전은 BIL 수익률로 이어붙임
    r = px["SGOV"].pct_change().where(px["SGOV"].shift(1).notna(), px["BIL"].pct_change())
    px["CASH"] = (1 + r.fillna(0)).cumprod()
    return px


def signal(tqqq):
    sma = tqqq.rolling(200).mean()
    pos = pd.Series(0, index=tqqq.index)
    state = up = down = 0
    for i in range(len(tqqq)):
        if np.isnan(sma.iloc[i]):
            continue
        if tqqq.iloc[i] > sma.iloc[i]:
            up, down = up + 1, 0
        else:
            up, down = 0, down + 1
        if state == 0 and up >= 3:
            state = 1
        elif state == 1 and down >= 2:
            state = 0
        pos.iloc[i] = state
    return pos


def run(px, park, ladder=True):
    p = px.loc[START:]
    sig = signal(px["TQQQ"]).loc[START:]
    sh = {"TQQQ": 0.0, "SCHD": 0.0, park: 1.0 / p[park].iloc[0]}
    entry, stage = None, 0
    eq, log = [], []
    for i, d in enumerate(p.index):
        c = p.loc[d]
        s = sig.iloc[i]
        if s == 1 and sh["TQQQ"] == 0 and sh["SCHD"] == 0:      # 매수
            v = sh[park] * c[park]
            sh[park], sh["TQQQ"], entry, stage = 0.0, v / c["TQQQ"], c["TQQQ"], 0
            log.append((d, "BUY", c["TQQQ"]))
        elif s == 0 and (sh["TQQQ"] > 0 or sh["SCHD"] > 0):    # 전량 매도
            v = sh["TQQQ"] * c["TQQQ"] + sh["SCHD"] * c["SCHD"]
            sh["TQQQ"] = sh["SCHD"] = 0.0
            sh[park] = v / c[park]
            log.append((d, "SELL", c["TQQQ"]))
        if ladder and sh["TQQQ"] > 0:
            while stage < len(LADDER) and c["TQQQ"] / entry - 1 >= LADDER[stage][0]:
                q = sh["TQQQ"] * LADDER[stage][1]
                sh["TQQQ"] -= q
                sh["SCHD"] += q * c["TQQQ"] / c["SCHD"]
                log.append((d, f"TP+{int(LADDER[stage][0]*100)}%", c["TQQQ"]))
                stage += 1
        eq.append(sum(sh[k] * c[k] for k in sh))
    return pd.Series(eq, index=p.index), log


def stats(e):
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    r = e.pct_change().dropna()
    dd = e / e.cummax() - 1
    return {
        "최종(1달러→)": e.iloc[-1] / e.iloc[0],
        "CAGR": (e.iloc[-1] / e.iloc[0]) ** (1 / yrs) - 1,
        "MDD": dd.min(),
        "연변동성": r.std() * np.sqrt(252),
        "샤프(rf=0)": r.mean() / r.std() * np.sqrt(252),
    }


def main():
    px = load()
    res, logs = {}, {}
    res["전략 + SGOV 대기"], logs["SGOV"] = run(px, "CASH")
    res["전략 + GLD 대기"], logs["GLD"] = run(px, "GLD")
    res["(참고) 익절없음 + SGOV"], _ = run(px, "CASH", ladder=False)
    res["(참고) 익절없음 + GLD"], _ = run(px, "GLD", ladder=False)
    p = px.loc[START:]
    for t in ["TQQQ", "QQQ", "GLD"]:
        res[f"{t} 보유"] = p[t] / p[t].iloc[0]
    curves = pd.DataFrame(res)
    print(f"기간: {curves.index[0].date()} ~ {curves.index[-1].date()}\n")
    t = pd.DataFrame({k: stats(v) for k, v in res.items()}).T
    fmt = t.copy()
    for col in ["CAGR", "MDD", "연변동성"]:
        fmt[col] = (t[col] * 100).map("{:.1f}%".format)
    fmt["최종(1달러→)"] = t["최종(1달러→)"].map("{:.2f}".format)
    fmt["샤프(rf=0)"] = t["샤프(rf=0)"].map("{:.2f}".format)
    print(fmt.to_string())

    yearly = curves.resample("YE").last()
    yearly = pd.concat([curves.iloc[[0]], yearly]).pct_change().dropna()
    yearly.index = yearly.index.year
    print("\n연도별 수익률(%)")
    print((yearly[["전략 + SGOV 대기", "전략 + GLD 대기", "TQQQ 보유"]] * 100).round(1).to_string())

    lg = logs["SGOV"]
    print(f"\n매매 횟수: 매수 {sum(x[1]=='BUY' for x in lg)}, 매도 {sum(x[1]=='SELL' for x in lg)}, "
          f"익절전환 {sum(x[1].startswith('TP') for x in lg)}")
    for d, a, pr in lg:
        print(f"  {d.date()} {a:8s} TQQQ {pr:.2f}")
    curves.round(4).to_csv("equity_curves.csv")


if __name__ == "__main__":
    main()
