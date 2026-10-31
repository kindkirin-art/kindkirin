"""
25% 슬리브 비교: QLD 추세추종 vs TQQQ 전략들 (최근 10년)

나머지 75%는 backtest.py와 동일 (금 30%, 비트코인 120일선 30%, SGOV 15%, 연말 리밸런싱).
슬리브 후보
  - QLD 추세추종      : ^IXIC 200일선, 3일 연속 위 매수 / 2일 연속 아래 매도
  - QLD 장기보유
  - TQQQ 추세추종     : QLD와 같은 신호(^IXIC 200일선 3일/2일)로 TQQQ 매매
  - TQQQ 200일선     : QQQ 종가 > 200일선이면 보유, 아래면 당일 매도 (대기 없음)
  - TQQQ 200일선+5%  : QQQ 종가 > 200일선×1.05 매수, 종가 < 200일선 매도 (엔벨로프)
  - TQQQ 장기보유
미보유 시 SGOV(상장 전 BIL). 거래비용/세금/슬리피지 미반영.
"""
import sys

import numpy as np
import pandas as pd
import yfinance as yf

import backtest as bt

END = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp.today().normalize()
START = END - pd.DateOffset(years=10)
WARMUP = START - pd.DateOffset(years=2)


def envelope_signal(px, n=200, band=0.05):
    sma = px.rolling(n).mean()
    pos = pd.Series(0, index=px.index)
    state = 0
    for i in range(len(px)):
        if np.isnan(sma.iloc[i]):
            continue
        if state == 0 and px.iloc[i] > sma.iloc[i] * (1 + band):
            state = 1
        elif state == 1 and px.iloc[i] < sma.iloc[i]:
            state = 0
        pos.iloc[i] = state
    return pos


def main():
    tickers = ["QLD", "TQQQ", "QQQ", "^IXIC", "GLD", "BTC-USD", "SGOV", "BIL"]
    px = yf.download(tickers, start=WARMUP, end=END + pd.Timedelta(days=1),
                     auto_adjust=True, progress=False)["Close"]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    stock_days = px["^IXIC"].dropna().index
    stock_days = stock_days[stock_days.isin(px["TQQQ"].dropna().index)]
    btc_days = px["BTC-USD"].dropna().index
    cal = btc_days.union(stock_days)

    r_cash = bt.cash_returns(px, stock_days).reindex(cal).fillna(0.0)
    ret = {t: px[t].reindex(cal).ffill().pct_change().fillna(0.0)
           for t in ["QLD", "TQQQ", "GLD", "BTC-USD"]}

    def held(sig):
        return sig.reindex(cal).ffill().fillna(0).shift(1).fillna(0)

    ixic = px["^IXIC"].loc[stock_days]
    qqq = px["QQQ"].loc[stock_days]
    sig_ixic = bt.qld_signal(ixic)
    sig_qqq = (qqq > qqq.rolling(200).mean()).astype(int)
    sig_env = envelope_signal(qqq)
    hb = held(bt.btc_signal(px["BTC-USD"].dropna()))
    r_btc = pd.Series(np.where(hb == 1, ret["BTC-USD"], r_cash), index=cal)

    sleeves = {
        "QLD 추세추종": (ret["QLD"], sig_ixic),
        "QLD 장기보유": (ret["QLD"], None),
        "TQQQ 추세추종(동일신호)": (ret["TQQQ"], sig_ixic),
        "TQQQ 200일선(QQQ)": (ret["TQQQ"], sig_qqq),
        "TQQQ 200일선+5%": (ret["TQQQ"], sig_env),
        "TQQQ 장기보유": (ret["TQQQ"], None),
    }

    sim_stock = stock_days[(stock_days >= START) & (stock_days <= END)]
    start_day = sim_stock[0]
    sim_days = cal[(cal >= start_day) & (cal <= END)]
    s = pd.Series(sim_stock, index=sim_stock)
    rebal = {d for d in s.groupby(s.index.year).max() if d.month == 12 and d.day >= 24}

    def simulate(sr):
        w = pd.Series(bt.WEIGHTS)
        vals, eq = w.copy(), []
        for d in sim_days:
            if d != start_day:
                vals = vals * (1 + sr.loc[d])
            total = vals.sum()
            if d in rebal:
                vals = w * total
            eq.append(total)
        return pd.Series(eq, index=sim_days)

    def sleeve_only(r):
        return (1 + r.loc[sim_days].iloc[1:]).cumprod()

    port, alone, info = {}, {}, {}
    for name, (r, sig) in sleeves.items():
        r_s = r if sig is None else pd.Series(np.where(held(sig) == 1, r, r_cash), index=cal)
        sr = pd.DataFrame({"QLD": r_s, "GLD": ret["GLD"], "BTC": r_btc, "CASH": r_cash}, index=cal)
        port[name] = simulate(sr)
        alone[name] = sleeve_only(r_s)
        if sig is not None:
            sg = sig.reindex(sim_stock)
            info[name] = (int((sg.diff().abs() > 0).sum()), sg.mean())

    pd.set_option("display.width", 220)
    print(f"기간: {sim_days[0].date()} ~ {sim_days[-1].date()}")
    for k, (n, e) in info.items():
        print(f"  {k}: 신호 전환 {n}회, 보유비중 {e:.0%}")

    def table(curves):
        st = pd.DataFrame({k: bt.stats(v) for k, v in curves.items()}).T
        for c in ["CAGR", "MDD", "변동성(연)"]:
            st[c] = st[c].map("{:.1%}".format)
        for c in ["최종 배수", "샤프(무위험0)", "CAGR/MDD"]:
            st[c] = st[c].map("{:.2f}".format)
        return st.to_string()

    print("\n[포트폴리오 전체 (25% 슬리브만 교체)]")
    print(table(port))
    print("\n[25% 슬리브 단독]")
    print(table(alone))

    yearly = pd.DataFrame({k: v.groupby(v.index.year).last() for k, v in port.items()})
    first = pd.DataFrame({k: [v.iloc[0]] for k, v in port.items()}, index=[yearly.index[0] - 1])
    yearly = pd.concat([first, yearly]).pct_change().dropna()
    print("\n[포트폴리오 연도별 수익률]")
    print(yearly.map("{:.1%}".format).to_string())

    ys = pd.DataFrame({k: v.groupby(v.index.year).last() for k, v in alone.items()})
    ys = pd.concat([pd.DataFrame({k: [1.0] for k in alone}, index=[ys.index[0] - 1]), ys]).pct_change().dropna()
    print("\n[슬리브 단독 연도별 수익률]")
    print(ys.map("{:.1%}".format).to_string())

    print("\n[슬리브 단독 최대낙폭 구간]")
    for k, v in alone.items():
        dd = v / v.cummax() - 1
        t = dd.idxmin()
        print(f"  {k}: {v.loc[:t].idxmax().date()} → {t.date()} ({dd.min():.1%})")

    print("\n[2018-01-01 이후 재계산 (포트폴리오)]")
    print(table({k: v.loc["2018-01-01":] for k, v in port.items()}))

    pd.DataFrame(port).to_csv("equity_curves_tqqq.csv", float_format="%.6f")


if __name__ == "__main__":
    main()
