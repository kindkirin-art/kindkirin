"""
포트폴리오 백테스트 (최근 10년)

전략 포트폴리오
  - QLD 25%  : 나스닥종합(^IXIC) 200일선 기준
               3일 연속 200일선 위 → 3일째 종가 매수
               2일 연속 200일선 아래 → 2일째 종가 매도
               (보유하지 않을 때는 현금성 자산(SGOV)으로 대기)
  - 금(GLD) 30%
  - 비트코인 30% : BTC 120일선 기준, 대기 없음
               종가 > 120일선 → 당일 종가 매수 / 종가 < 120일선 → 당일 종가 매도
               (보유하지 않을 때는 SGOV로 대기)
  - SGOV 15%  (상장 전 2020-06 이전 구간은 BIL로 대체)
  - 연말(그 해 마지막 미국 증시 거래일 종가) 리밸런싱

비교 대상
  - QLD 100% 장기보유
  - QLD 200일선 전략 단독 (참고용)

거래비용/세금/슬리피지는 반영하지 않음.
"""
import sys

import numpy as np
import pandas as pd
import yfinance as yf

END = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp.today().normalize()
START = END - pd.DateOffset(years=10)
WARMUP = START - pd.DateOffset(years=2)
WEIGHTS = {"QLD": 0.25, "GLD": 0.30, "BTC": 0.30, "CASH": 0.15}


def load():
    tickers = ["QLD", "^IXIC", "GLD", "BTC-USD", "SGOV", "BIL"]
    px = yf.download(tickers, start=WARMUP, end=END + pd.Timedelta(days=1),
                     auto_adjust=True, progress=False)["Close"]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    return px


def cash_returns(px, stock_days):
    s = px.loc[stock_days]
    r_sgov = s["SGOV"].pct_change()
    r_bil = s["BIL"].pct_change()
    return r_sgov.where(s["SGOV"].shift(1).notna(), r_bil).fillna(0.0)


def qld_signal(ixic):
    """종가 기준 체결 후의 보유 여부(1/0)를 반환. 3일 대기 매수, 2일 대기 매도."""
    sma = ixic.rolling(200).mean()
    above = ixic > sma
    pos = pd.Series(0, index=ixic.index)
    state, up, down = 0, 0, 0
    for i, d in enumerate(ixic.index):
        if np.isnan(sma.iloc[i]):
            pos.iloc[i] = 0
            continue
        if above.iloc[i]:
            up, down = up + 1, 0
        else:
            up, down = 0, down + 1
        if state == 0 and up >= 3:
            state = 1
        elif state == 1 and down >= 2:
            state = 0
        pos.iloc[i] = state
    return pos


def btc_signal(btc):
    sma = btc.rolling(120).mean()
    return (btc > sma).astype(int).where(sma.notna(), 0)


def stats(equity):
    equity = equity.dropna()
    daily = equity.resample("D").last().ffill().pct_change().dropna()
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
    stock_days = px["^IXIC"].dropna().index
    btc_days = px["BTC-USD"].dropna().index
    cal = btc_days.union(stock_days)  # 주말 포함 달력 (주식은 주말 수익률 0)

    # ---- 일간 수익률 (모두 '전일 종가→당일 종가') ----
    r_cash_s = cash_returns(px, stock_days)
    r_cash = r_cash_s.reindex(cal).fillna(0.0)
    r_qld = px["QLD"].reindex(cal).ffill().pct_change().fillna(0.0)
    r_gld = px["GLD"].reindex(cal).ffill().pct_change().fillna(0.0)
    r_btc = px["BTC-USD"].reindex(cal).ffill().pct_change().fillna(0.0)

    # ---- 신호 (당일 종가에 체결 → 다음 날 수익률부터 반영) ----
    qpos = qld_signal(px["^IXIC"].dropna()).reindex(cal).ffill().fillna(0)
    bpos = btc_signal(px["BTC-USD"].dropna()).reindex(cal).ffill().fillna(0)
    held_q = qpos.shift(1).fillna(0)
    held_b = bpos.shift(1).fillna(0)

    sleeve_r = pd.DataFrame({
        "QLD": np.where(held_q == 1, r_qld, r_cash),
        "GLD": r_gld,
        "BTC": np.where(held_b == 1, r_btc, r_cash),
        "CASH": r_cash,
    }, index=cal)

    sim_days = cal[(cal >= START) & (cal <= END)]
    sim_stock = stock_days[(stock_days >= START) & (stock_days <= END)]
    start_day = sim_stock[0]
    sim_days = sim_days[sim_days >= start_day]
    # 연말 리밸런싱일: 각 연도의 마지막 미국 증시 거래일 (마지막 해의 미완료 연도는 제외)
    s = pd.Series(sim_stock, index=sim_stock)
    rebal_days = set(s.groupby(s.index.year).max())
    rebal_days = {d for d in rebal_days if d.month == 12 and d.day >= 24}

    vals = pd.Series(WEIGHTS) * 1.0
    eq = []
    for d in sim_days:
        if d != start_day:
            vals = vals * (1 + sleeve_r.loc[d])
        total = vals.sum()
        if d in rebal_days:
            vals = pd.Series(WEIGHTS) * total
        eq.append(total)
    port = pd.Series(eq, index=sim_days)

    # 비교: QLD 장기보유, QLD 200일선 단독
    bh = (1 + r_qld.loc[sim_days].where(sim_days != start_day, 0)).cumprod()
    q_only = (1 + sleeve_r["QLD"].loc[sim_days].where(sim_days != start_day, 0)).cumprod()

    curves = {"전략 포트폴리오": port, "QLD 장기보유": bh, "QLD 200일선 단독(참고)": q_only}
    # 주식 거래일 기준으로 비교 (주말 BTC 변동 포함하되 샘플링은 거래일)
    st = pd.DataFrame({k: stats(v) for k, v in curves.items()}).T

    yearly = pd.DataFrame({k: v.groupby(v.index.year).last() for k, v in curves.items()})
    first = pd.DataFrame({k: [v.iloc[0]] for k, v in curves.items()}, index=[yearly.index[0] - 1])
    yearly = pd.concat([first, yearly]).pct_change().dropna()

    trades_q = int((qpos.loc[sim_days].diff().abs() > 0).sum())
    trades_b = int((bpos.loc[sim_days].diff().abs() > 0).sum())
    exposure_q = qpos.loc[sim_stock].mean()
    exposure_b = bpos.loc[sim_days].mean()

    pd.set_option("display.width", 200)
    print(f"기간: {sim_days[0].date()} ~ {sim_days[-1].date()}")
    print(f"QLD 신호 전환 {trades_q}회 (보유비중 {exposure_q:.0%}), "
          f"BTC 신호 전환 {trades_b}회 (보유비중 {exposure_b:.0%})")
    print("\n[성과 요약]")
    fmt = st.copy()
    for c in ["CAGR", "MDD", "변동성(연)"]:
        fmt[c] = fmt[c].map("{:.1%}".format)
    for c in ["최종 배수", "샤프(무위험0)", "CAGR/MDD"]:
        fmt[c] = fmt[c].map("{:.2f}".format)
    print(fmt.to_string())
    print("\n[연도별 수익률] (첫해·마지막해는 부분 기간)")
    print(yearly.map("{:.1%}".format).to_string())

    # MDD 발생 구간
    print("\n[최대낙폭 구간]")
    for k, v in curves.items():
        dd = v / v.cummax() - 1
        trough = dd.idxmin()
        peak = v.loc[:trough].idxmax()
        print(f"  {k}: {peak.date()} → {trough.date()} ({dd.min():.1%})")

    out = pd.DataFrame(curves)
    out.to_csv("equity_curves.csv", float_format="%.6f")


if __name__ == "__main__":
    main()
