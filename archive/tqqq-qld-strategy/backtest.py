"""
TQQQ 전략 포트폴리오 vs QLD 전략 포트폴리오 (최근 10년 백테스트)

공통: 금(GLD) 30%, 비트코인 30%, SGOV 15%, 연말 리밸런싱
  - 비트코인: BTC 종가 > 120일선이면 보유, 아래면 당일 종가 매도 (대기 없음). 미보유 시 SGOV
  - SGOV: 2020-06 상장 전은 BIL로 대체
  - 연말 리밸런싱: 그해 마지막 미국 증시 거래일 종가

TQQQ 전략 25%
  - TQQQ 200일선: 3일 연속 위 → 3일째 종가 매수 / 2일 연속 아래 → 2일째 종가 매도
  - 보유 중 매수단가 대비 +20/+50/+100/+200% 도달 시
    TQQQ 잔량의 25/25/50/50%를 SCHD로 전환 (해당일 종가)
  - 매도 신호 시 TQQQ와 SCHD 모두 매도 → SGOV
  - 다음 매수 신호 때 전액 TQQQ 매수, 매수단가/익절 단계 초기화
  - 연말 리밸런싱으로 슬리브를 25%로 맞출 때 TQQQ/SCHD/SGOV 구성비는 유지,
    TQQQ 추가 매수분은 매수단가(평단)에 반영

QLD 전략 25%
  - 나스닥종합(^IXIC) 200일선: 3일 연속 위 → 3일째 종가 매수 / 2일 연속 아래 → 2일째 종가 매도
  - 미보유 시 SGOV

신호는 당일 종가에 체결하고 다음 날 수익률부터 반영. 거래비용/세금/슬리피지 미반영.
데이터: Yahoo Finance 수정종가 (배당 재투자 반영).
"""
import sys

import numpy as np
import pandas as pd
import yfinance as yf

END = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp.today().normalize()
START = pd.Timestamp(sys.argv[2]) if len(sys.argv) > 2 else END - pd.DateOffset(years=10)
WARMUP = START - pd.DateOffset(years=2)
WEIGHTS = {"LEV": 0.25, "GLD": 0.30, "BTC": 0.30, "CASH": 0.15}
LADDER = [(0.20, 0.25), (0.50, 0.25), (1.00, 0.50), (2.00, 0.50)]  # (수익률, 잔량 중 전환 비율)


def load():
    tickers = ["TQQQ", "QLD", "SCHD", "^IXIC", "GLD", "BTC-USD", "SGOV", "BIL"]
    px = yf.download(tickers, start=WARMUP, end=END + pd.Timedelta(days=1),
                     auto_adjust=True, progress=False)["Close"]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    return px


def ma_signal(price, n=200, up_days=3, down_days=2):
    """n일선 기준 보유 여부(1/0). up_days 연속 위면 매수, down_days 연속 아래면 매도."""
    sma = price.rolling(n).mean()
    pos = pd.Series(0, index=price.index)
    state, up, down = 0, 0, 0
    for i in range(len(price)):
        if np.isnan(sma.iloc[i]):
            continue
        if price.iloc[i] > sma.iloc[i]:
            up, down = up + 1, 0
        else:
            up, down = 0, down + 1
        if state == 0 and up >= up_days:
            state = 1
        elif state == 1 and down >= down_days:
            state = 0
        pos.iloc[i] = state
    return pos


def main():
    px = load()
    stock_days = px[["TQQQ", "QLD", "^IXIC", "GLD"]].dropna().index
    btc_days = px["BTC-USD"].dropna().index
    cal = btc_days.union(stock_days)  # 주말 포함 (주식은 주말 수익률 0)

    def ret(t):
        return px[t].reindex(cal).ffill().pct_change().fillna(0.0)

    s = px.loc[stock_days]
    r_cash = (s["SGOV"].pct_change().where(s["SGOV"].shift(1).notna(), s["BIL"].pct_change())
              .reindex(cal).fillna(0.0))
    r = {t: ret(t) for t in ["TQQQ", "QLD", "SCHD", "GLD", "BTC-USD"]}
    price = {t: px[t].reindex(cal).ffill() for t in ["TQQQ", "QLD"]}

    btc = px["BTC-USD"].dropna()
    btc_sma = btc.rolling(120).mean()
    btc_pos = (btc > btc_sma).astype(int).where(btc_sma.notna(), 0)
    btc_held = btc_pos.reindex(cal).ffill().fillna(0).shift(1).fillna(0)
    r_btc = pd.Series(np.where(btc_held == 1, r["BTC-USD"], r_cash), index=cal)

    sim_stock = stock_days[(stock_days >= START) & (stock_days <= END)]
    start_day = sim_stock[0]
    sim_days = cal[(cal >= start_day) & (cal <= END)]
    stock_set = set(sim_stock)
    yr = pd.Series(sim_stock, index=sim_stock)
    rebal = {d for d in yr.groupby(yr.index.year).max() if d.month == 12 and d.day >= 24}

    def simulate(asset, sig, ladder):
        sig = sig.reindex(cal).ffill().fillna(0)
        p_lev, r_lev = price[asset], r[asset]
        a = {"lev": 0.0, "schd": 0.0, "cash": WEIGHTS["LEV"],
             "GLD": WEIGHTS["GLD"], "BTC": WEIGHTS["BTC"], "CASH": WEIGHTS["CASH"]}
        state, avg, level = 0, np.nan, 0
        eq, sleeve_r, log = [], [], []
        prev_sleeve = WEIGHTS["LEV"]
        for d in sim_days:
            if d != start_day:
                a["lev"] *= 1 + r_lev[d]
                a["schd"] *= 1 + r["SCHD"][d]
                a["cash"] *= 1 + r_cash[d]
                a["GLD"] *= 1 + r["GLD"][d]
                a["BTC"] *= 1 + r_btc[d]
                a["CASH"] *= 1 + r_cash[d]
            if d in stock_set:
                p = p_lev[d]
                if state == 0 and sig[d] == 1:
                    a["lev"] += a["cash"] + a["schd"]
                    a["cash"] = a["schd"] = 0.0
                    state, avg, level = 1, p, 0
                    log.append((d, "매수"))
                elif state == 1 and sig[d] == 0:
                    a["cash"] += a["lev"] + a["schd"]
                    a["lev"] = a["schd"] = 0.0
                    state = 0
                    log.append((d, "매도"))
                if state == 1:
                    while level < len(ladder) and p / avg - 1 >= ladder[level][0]:
                        amt = a["lev"] * ladder[level][1]
                        a["lev"] -= amt
                        a["schd"] += amt
                        log.append((d, f"+{ladder[level][0]:.0%} 익절"))
                        level += 1
            cur = a["lev"] + a["schd"] + a["cash"]
            sleeve_r.append(cur / prev_sleeve - 1)
            total = sum(a.values())
            if d in rebal:
                k = WEIGHTS["LEV"] * total / (a["lev"] + a["schd"] + a["cash"])
                if k > 1 and a["lev"] > 0:
                    avg = (avg + (k - 1) * p_lev[d]) / k
                for x in ("lev", "schd", "cash"):
                    a[x] *= k
                for x in ("GLD", "BTC", "CASH"):
                    a[x] = WEIGHTS[x] * total
            eq.append(total)
            prev_sleeve = a["lev"] + a["schd"] + a["cash"]
        sleeves[asset if ladder else f"{asset} (익절 없음)"] = (1 + pd.Series(sleeve_r, index=sim_days)).cumprod()
        return pd.Series(eq, index=sim_days), log

    sleeves = {}
    tqqq_sig = ma_signal(px["TQQQ"].loc[stock_days])
    ixic_sig = ma_signal(px["^IXIC"].loc[stock_days])
    ports = {}
    ports["TQQQ 전략"], tqqq_log = simulate("TQQQ", tqqq_sig, LADDER)
    ports["QLD 전략"], qld_log = simulate("QLD", ixic_sig, [])
    # 참고용 변형
    ref = {"TQQQ 전략 (익절 없음)": simulate("TQQQ", tqqq_sig, [])[0]}
    sl = dict(sleeves)
    ref["TQQQ 전략 (신호를 ^IXIC로)"] = simulate("TQQQ", ixic_sig, LADDER)[0]
    ref["QLD 장기보유"] = simulate("QLD", pd.Series(1, index=stock_days), [])[0]
    sleeves = {"TQQQ 25% 칸 (익절 포함)": sl["TQQQ"], "QLD 25% 칸": sl["QLD (익절 없음)"],
               "TQQQ 25% 칸 (익절 없음)": sl["TQQQ (익절 없음)"],
               "QLD 25% 칸 (장기보유)": sleeves["QLD (익절 없음)"]}

    def stats(e):
        daily = e.pct_change().dropna()
        years = (e.index[-1] - e.index[0]).days / 365.25
        cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1
        dd = e / e.cummax() - 1
        vol = daily.std() * np.sqrt(365)
        trough = dd.idxmin()
        return {"최종 배수": f"{e.iloc[-1] / e.iloc[0]:.1f}배", "CAGR": f"{cagr:.1%}",
                "MDD": f"{dd.min():.1%}", "MDD 구간": f"{e.loc[:trough].idxmax().date()}~{trough.date()}",
                "변동성": f"{vol:.1%}", "샤프": f"{daily.mean() * 365 / vol:.2f}",
                "CAGR/MDD": f"{cagr / abs(dd.min()):.2f}"}

    def yearly(e):
        rows = {}
        prev = e.iloc[0]
        for y, g in e.groupby(e.index.year):
            path = pd.concat([pd.Series([prev]), g.reset_index(drop=True)])
            rows[y] = (g.iloc[-1] / prev - 1, (path / path.cummax() - 1).min())
            prev = g.iloc[-1]
        return rows

    pd.set_option("display.width", 200)
    pd.set_option("display.unicode.east_asian_width", True)
    print(f"기간: {sim_days[0].date()} ~ {sim_days[-1].date()}")
    for name, log in [("TQQQ 전략", tqqq_log), ("QLD 전략", qld_log)]:
        kinds = pd.Series([k for _, k in log]).value_counts().to_dict()
        print(f"  {name} 매매: {kinds}")

    print("\n[성과 요약]")
    print(pd.DataFrame({k: stats(v) for k, v in {**ports, **ref}.items()}).T.to_string())

    print("\n[25% 칸 단독 성과]")
    print(pd.DataFrame({k: stats(v) for k, v in sleeves.items()}).T.to_string())
    ys = {k: yearly(v) for k, v in sleeves.items()}
    print(pd.DataFrame({f"{k} 수익률/MDD": {y: f"{a:.0%} / {b:.0%}" for y, (a, b) in v.items()}
                        for k, v in ys.items()}).to_string())

    print("\n[연도별 수익률 / 연중 MDD]  (첫해·마지막해는 부분 기간)")
    yt, yq, yh = yearly(ports["TQQQ 전략"]), yearly(ports["QLD 전략"]), yearly(ref["QLD 장기보유"])
    tbl = pd.DataFrame({
        "TQQQ 수익률": {y: f"{v[0]:.1%}" for y, v in yt.items()},
        "TQQQ MDD": {y: f"{v[1]:.1%}" for y, v in yt.items()},
        "QLD 수익률": {y: f"{v[0]:.1%}" for y, v in yq.items()},
        "QLD MDD": {y: f"{v[1]:.1%}" for y, v in yq.items()},
        "QLD 장기보유 수익률": {y: f"{v[0]:.1%}" for y, v in yh.items()},
        "QLD 장기보유 MDD": {y: f"{v[1]:.1%}" for y, v in yh.items()},
    })
    print(tbl.to_string())

    print("\n[2018-01-01 이후 재계산 (2017 비트코인 폭등 제외)]")
    print(pd.DataFrame({k: stats(v.loc["2018-01-01":]) for k, v in {**ports, "QLD 장기보유": ref["QLD 장기보유"]}.items()}).T.to_string())

    print("\n[TQQQ 전략 매매 내역]")
    for d, k in tqqq_log:
        if d >= start_day:
            print(f"  {d.date()} {k}")

    pd.DataFrame({**ports, **ref}).to_csv("equity_curves.csv", float_format="%.6f")


if __name__ == "__main__":
    main()
