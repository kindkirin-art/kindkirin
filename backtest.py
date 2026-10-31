"""
비트코인·금 포트폴리오 백테스트 (최근 10년)

공통 비트코인 신호: BTC 120일선, 대기 없음
  종가 > 120일선 → 당일 종가 매수 / 종가 < 120일선 → 당일 종가 매도
  (보유하지 않을 때는 현금성 자산 SGOV로 대기, 2020-06 상장 전은 BIL)

1) 동시 보유 포트폴리오
  - 비트코인(120일선) 50% + 금(GLD) 50%
  - 연말(그 해 마지막 미국 증시 거래일 종가) 리밸런싱

2) 스위치 포트폴리오
  - 매수신호 시 전 재산을 비트코인으로 매수
  - 매수가 대비 수익률이 25% / 50% / 100% / 200%에 도달한 날 종가에
    그 시점 비트코인 잔량의 25% / 25% / 50% / 50%를 금으로 전환
  - 매도신호 시 비트코인과 금을 모두 매도하고 SGOV로 대기
  - 참고용 변형: 매도신호 때 비트코인만 팔고 금은 계속 보유

거래비용/세금/슬리피지는 반영하지 않음.
"""
import sys

import numpy as np
import pandas as pd
import yfinance as yf

END = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp.today().normalize()
START = END - pd.DateOffset(years=10)
WARMUP = START - pd.DateOffset(years=1)
LEVELS = [(0.25, 0.25), (0.50, 0.25), (1.00, 0.50), (2.00, 0.50)]  # (수익률, 금 전환 비율)


def load():
    tickers = ["GLD", "BTC-USD", "SGOV", "BIL"]
    px = yf.download(tickers, start=WARMUP, end=END + pd.Timedelta(days=1),
                     auto_adjust=True, progress=False)["Close"]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    return px


def cash_returns(px, stock_days):
    s = px.loc[stock_days]
    r_sgov = s["SGOV"].pct_change()
    r_bil = s["BIL"].pct_change()
    return r_sgov.where(s["SGOV"].shift(1).notna(), r_bil).fillna(0.0)


def btc_signal(btc):
    sma = btc.rolling(120).mean()
    return (btc > sma).astype(int).where(sma.notna(), 0)


def stats(equity):
    equity = equity.dropna()
    daily = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    mdd = (equity / equity.cummax() - 1).min()
    vol = daily.std() * np.sqrt(365)
    sharpe = daily.mean() * 365 / vol if vol else np.nan
    return {
        "최종 배수": equity.iloc[-1] / equity.iloc[0],
        "CAGR": cagr,
        "MDD": mdd,
        "변동성(연)": vol,
        "샤프(무위험0)": sharpe,
        "CAGR/MDD": cagr / abs(mdd),
    }


def main():
    px = load()
    stock_days = px["GLD"].dropna().index
    btc_days = px["BTC-USD"].dropna().index
    cal = btc_days.union(stock_days)  # 주말 포함 달력 (금·현금은 주말 수익률 0)

    r_cash = cash_returns(px, stock_days).reindex(cal).fillna(0.0)
    r_gld = px["GLD"].reindex(cal).ffill().pct_change().fillna(0.0)
    btc = px["BTC-USD"].reindex(cal).ffill()
    r_btc = btc.pct_change().fillna(0.0)
    bpos = btc_signal(px["BTC-USD"].dropna()).reindex(cal).ffill().fillna(0)

    sim_stock = stock_days[(stock_days >= START) & (stock_days <= END)]
    start_day = sim_stock[0]
    sim_days = cal[(cal >= start_day) & (cal <= END)]
    s = pd.Series(sim_stock, index=sim_stock)
    rebal_days = {d for d in s.groupby(s.index.year).max() if d.month == 12 and d.day >= 24}

    # ---- 1) 동시 보유: 첫날 종가에 BTC 신호대로 50/50 편입 ----
    held_b = bpos.shift(1).fillna(0)
    sleeve_btc = pd.Series(np.where(held_b == 1, r_btc, r_cash), index=cal)
    vals = np.array([0.5, 0.5])
    eq = []
    for d in sim_days:
        if d != start_day:
            vals = vals * (1 + np.array([sleeve_btc.loc[d], r_gld.loc[d]]))
        total = vals.sum()
        if d in rebal_days:
            vals = np.array([0.5, 0.5]) * total
        eq.append(total)
    combo = pd.Series(eq, index=sim_days)

    # ---- 2) 스위치: 당일 종가 기준으로 수익 실현/매도 체결 ----
    def switch(keep_gold):
        b, g, c = 0.0, 0.0, 1.0  # 비트코인, 금, 현금
        entry, hit = None, 0
        eq, log = [], []
        for d in sim_days:
            if d != start_day:
                b *= 1 + r_btc.loc[d]
                g *= 1 + r_gld.loc[d]
                c *= 1 + r_cash.loc[d]
            sig = bpos.loc[d]
            if sig == 1 and entry is None:  # 매수: 현금(+금) 전부 비트코인으로
                b += c + (0.0 if keep_gold else g)
                c = 0.0
                if not keep_gold:
                    g = 0.0
                entry, hit = btc.loc[d], 0
                log.append((d, "매수", btc.loc[d], None))
            elif sig == 1:
                ret = btc.loc[d] / entry - 1
                while hit < len(LEVELS) and ret >= LEVELS[hit][0]:
                    amt = b * LEVELS[hit][1]
                    b, g = b - amt, g + amt
                    log.append((d, f"+{LEVELS[hit][0]:.0%} 금전환", btc.loc[d], ret))
                    hit += 1
            elif sig == 0 and entry is not None:  # 매도
                ret = btc.loc[d] / entry - 1
                c += b + (0.0 if keep_gold else g)
                b = 0.0
                if not keep_gold:
                    g = 0.0
                entry = None
                log.append((d, "매도", btc.loc[d], ret))
            eq.append(b + g + c)
        return pd.Series(eq, index=sim_days), log

    sw, sw_log = switch(keep_gold=False)
    sw_keep, _ = switch(keep_gold=True)
    btc_only = (1 + sleeve_btc.loc[sim_days].where(sim_days != start_day, 0)).cumprod()
    btc_bh = btc.loc[sim_days] / btc.loc[start_day]

    curves = {
        "동시 보유 (BTC120 50 + 금 50)": combo,
        "스위치 (매도시 전량 현금)": sw,
        "[참고] 스위치 (매도시 금 유지)": sw_keep,
        "[참고] BTC 120일선 단독": btc_only,
        "[참고] BTC 장기보유": btc_bh,
    }
    st = pd.DataFrame({k: stats(v) for k, v in curves.items()}).T

    yearly = pd.DataFrame({k: v.groupby(v.index.year).last() for k, v in curves.items()})
    first = pd.DataFrame({k: [v.iloc[0]] for k, v in curves.items()}, index=[yearly.index[0] - 1])
    yearly = pd.concat([first, yearly]).pct_change().dropna()

    trades_b = int((bpos.loc[sim_days].diff().abs() > 0).sum())
    pd.set_option("display.width", 250)
    print(f"기간: {sim_days[0].date()} ~ {sim_days[-1].date()}")
    print(f"BTC 신호 전환 {trades_b}회 (보유비중 {bpos.loc[sim_days].mean():.0%})")
    print("\n[성과 요약]")
    fmt = st.copy()
    for c in ["CAGR", "MDD", "변동성(연)"]:
        fmt[c] = fmt[c].map("{:.1%}".format)
    for c in ["최종 배수", "샤프(무위험0)", "CAGR/MDD"]:
        fmt[c] = fmt[c].map("{:.2f}".format)
    print(fmt.to_string())
    print("\n[연도별 수익률] (첫해·마지막해는 부분 기간)")
    print(yearly.map("{:.1%}".format).to_string())

    print("\n[최대낙폭 구간]")
    for k, v in curves.items():
        dd = v / v.cummax() - 1
        trough = dd.idxmin()
        peak = v.loc[:trough].idxmax()
        print(f"  {k}: {peak.date()} → {trough.date()} ({dd.min():.1%})")

    # 스위치 매매 사이클 요약
    print("\n[스위치 매매 사이클] (매수가 대비 매도 시점 BTC 수익률, 도달한 금전환 단계)")
    cycles, cur = [], None
    for d, act, p, ret in sw_log:
        if act == "매수":
            cur = {"매수일": d.date(), "단계": 0}
        elif act == "매도" and cur:
            cur.update({"매도일": d.date(), "BTC 수익률": ret})
            cycles.append(cur)
            cur = None
        elif cur:
            cur["단계"] += 1
    if cur:
        cur.update({"매도일": "보유중", "BTC 수익률": btc.loc[sim_days[-1]] / sw_log[-1][2] - 1
                    if sw_log[-1][1] == "매수" else np.nan})
        cycles.append(cur)
    cyc = pd.DataFrame(cycles)
    print(f"  사이클 {len(cyc)}회, 금전환 단계 도달 분포: "
          + ", ".join(f"{k}단계 {v}회" for k, v in cyc["단계"].value_counts().sort_index().items()))
    big = cyc[cyc["단계"] >= 2]
    print(big.assign(**{"BTC 수익률": big["BTC 수익률"].map(
        lambda x: "" if pd.isna(x) else f"{x:.0%}")}).to_string(index=False))

    pd.DataFrame(curves).to_csv("equity_curves.csv", float_format="%.6f")


if __name__ == "__main__":
    main()
