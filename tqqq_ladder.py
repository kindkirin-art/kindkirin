"""
TQQQ 수익률 기준 SCHD 익절 전략 (25% 슬리브) 백테스트

  - 매수/매도 신호: 200일선 (^IXIC 3일/2일 또는 QQQ 당일)
  - 보유 중 매수단가 대비 +25/+50/+100/+200% 도달 시
    남은 TQQQ의 25/25/50/50%를 SCHD로 전환 (종가 체결)
  - 매도 신호: 남은 TQQQ → SGOV
  - 복귀 A: 다음 매수 신호 때 SCHD+SGOV 전부 TQQQ로 (매수단가·단계 초기화)
    복귀 B: SCHD는 계속 보유, SGOV만 TQQQ로
  - 비교: 익절 자금을 SCHD 대신 SGOV로 (SGOV익절 A)
  - 연말 리밸런싱: 슬리브를 25%로 맞출 때 내부 구성비 유지 (추가 매수분은 평단에 반영)
나머지 75%는 backtest.py와 동일. 거래비용/세금/슬리피지 미반영.
"""
import sys

import numpy as np
import pandas as pd
import yfinance as yf

import backtest as bt

END = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp.today().normalize()
START = END - pd.DateOffset(years=10)
WARMUP = START - pd.DateOffset(years=2)
LADDER = [(0.25, 0.25), (0.50, 0.25), (1.00, 0.50), (2.00, 0.50)]


def main():
    tickers = ["QLD", "TQQQ", "QQQ", "SCHD", "^IXIC", "GLD", "BTC-USD", "SGOV", "BIL"]
    px = yf.download(tickers, start=WARMUP, end=END + pd.Timedelta(days=1),
                     auto_adjust=True, progress=False)["Close"]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    stock_days = px["^IXIC"].dropna().index
    stock_days = stock_days[stock_days.isin(px["TQQQ"].dropna().index)]
    cal = px["BTC-USD"].dropna().index.union(stock_days)

    r_cash = bt.cash_returns(px, stock_days).reindex(cal).fillna(0.0)
    ret = {t: px[t].reindex(cal).ffill().pct_change().fillna(0.0)
           for t in ["QLD", "TQQQ", "SCHD", "GLD", "BTC-USD"]}
    hb = bt.btc_signal(px["BTC-USD"].dropna()).reindex(cal).ffill().fillna(0).shift(1).fillna(0)
    r_btc = pd.Series(np.where(hb == 1, ret["BTC-USD"], r_cash), index=cal)
    tq_px = px["TQQQ"].reindex(cal).ffill()

    ixic, qqq = px["^IXIC"].loc[stock_days], px["QQQ"].loc[stock_days]
    signals = {"IXIC 3일/2일": bt.qld_signal(ixic),
               "QQQ 200일선": (qqq > qqq.rolling(200).mean()).astype(int)}

    sim_stock = stock_days[(stock_days >= START) & (stock_days <= END)]
    start_day = sim_stock[0]
    sim_days = cal[(cal >= start_day) & (cal <= END)]
    s = pd.Series(sim_stock, index=sim_stock)
    rebal = {d for d in s.groupby(s.index.year).max() if d.month == 12 and d.day >= 24}
    stock_set = set(sim_stock)
    W = bt.WEIGHTS

    def run(sig, ladder, mode, asset="TQQQ", dest="SCHD"):
        r_lev = ret[asset]
        r_dest = ret["SCHD"] if dest == "SCHD" else r_cash
        p_lev = px[asset].reindex(cal).ffill()
        sig = sig.reindex(cal).ffill().fillna(0)
        a = dict(lev=0.0, schd=0.0, cash=W["QLD"], GLD=W["GLD"], BTC=W["BTC"], CASH=W["CASH"])
        state, avg, level, eq, sl, n_ladder = 0, np.nan, 0, [], [], 0
        sleeve_prev = W["QLD"]
        sl_ret = []
        for d in sim_days:
            if d != start_day:
                a["lev"] *= 1 + r_lev[d]; a["schd"] *= 1 + r_dest[d]
                a["cash"] *= 1 + r_cash[d]; a["GLD"] *= 1 + ret["GLD"][d]
                a["BTC"] *= 1 + r_btc[d]; a["CASH"] *= 1 + r_cash[d]
            sleeve = a["lev"] + a["schd"] + a["cash"]
            sl_ret.append(sleeve / sleeve_prev - 1 if d != start_day else 0.0)
            if d in stock_set:
                p = p_lev[d]
                if state == 0 and sig[d] == 1:
                    move = a["cash"] + (a["schd"] if mode == "A" else 0.0)
                    a["cash"] -= a["cash"]
                    if mode == "A":
                        a["schd"] = 0.0
                    a["lev"] += move
                    state, avg, level = 1, p, 0
                elif state == 1 and sig[d] == 0:
                    a["cash"] += a["lev"]; a["lev"] = 0.0
                    state = 0
                if state == 1:
                    while level < len(ladder) and p / avg - 1 >= ladder[level][0]:
                        amt = a["lev"] * ladder[level][1]
                        a["lev"] -= amt; a["schd"] += amt
                        level += 1; n_ladder += 1
            total = sum(a.values())
            if d in rebal:
                k = W["QLD"] * total / (a["lev"] + a["schd"] + a["cash"])
                if k > 1 and a["lev"] > 0:
                    avg = (avg + (k - 1) * p_lev[d]) / k
                for x in ("lev", "schd", "cash"):
                    a[x] *= k
                for x in ("GLD", "BTC", "CASH"):
                    a[x] = W[x] * total
            sleeve_prev = a["lev"] + a["schd"] + a["cash"]
            eq.append(total)
        port = pd.Series(eq, index=sim_days)
        alone = (1 + pd.Series(sl_ret, index=sim_days)).cumprod()
        return port, alone, n_ladder

    rows = {}
    for sname, sig in signals.items():
        rows[f"TQQQ {sname} (익절 없음)"] = run(sig, [], "A")
        rows[f"TQQQ {sname} + SCHD익절 A"] = run(sig, LADDER, "A")
        rows[f"TQQQ {sname} + SCHD익절 B"] = run(sig, LADDER, "B")
        rows[f"TQQQ {sname} + SGOV익절 A"] = run(sig, LADDER, "A", dest="SGOV")
    rows["QLD IXIC 3일/2일 (기존)"] = run(signals["IXIC 3일/2일"], [], "A", "QLD")
    rows["QLD 장기보유"] = run(pd.Series(1, index=stock_days), [], "A", "QLD")
    rows["TQQQ 장기보유"] = run(pd.Series(1, index=stock_days), [], "A")

    def table(curves):
        st = pd.DataFrame({k: bt.stats(v) for k, v in curves.items()}).T
        for c in ["CAGR", "MDD", "변동성(연)"]:
            st[c] = st[c].map("{:.1%}".format)
        for c in ["최종 배수", "샤프(무위험0)", "CAGR/MDD"]:
            st[c] = st[c].map("{:.2f}".format)
        return st.to_string()

    pd.set_option("display.width", 220)
    print(f"기간: {sim_days[0].date()} ~ {sim_days[-1].date()}")
    print("익절 실행 횟수:", {k: v[2] for k, v in rows.items() if "익절 A" in k or "익절 B" in k})
    print("\n[포트폴리오 전체]")
    print(table({k: v[0] for k, v in rows.items()}))
    print("\n[25% 슬리브 단독]")
    print(table({k: v[1] for k, v in rows.items()}))
    print("\n[2018-01-01 이후 포트폴리오]")
    print(table({k: v[0].loc["2018-01-01":] for k, v in rows.items()}))
    ys = pd.DataFrame({k: v[1].groupby(v[1].index.year).last() for k, v in rows.items()})
    ys = pd.concat([pd.DataFrame({k: [1.0] for k in rows}, index=[ys.index[0] - 1]), ys]).pct_change().dropna()
    print("\n[슬리브 단독 연도별]")
    print(ys.map("{:.0%}".format).to_string())


if __name__ == "__main__":
    main()
