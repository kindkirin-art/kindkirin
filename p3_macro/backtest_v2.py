"""P3 개선안 비교: 매매를 늘리는 경로와 손절을 더한 규칙.

backtest.py의 기준 규칙 위에 아래 옵션을 켜고 끕니다.
- trend   : 추세 매수 경로. 첫 반등 이후 이중바닥이 오지 않아도
            QQQ 종가 > 50일선 이고 HY <= HYmax - trend_drop 이면 1차 매수 (V자 회복용).
            trend_ma=200 이면 QQQ 종가 > QQQ 200일선 도 요구.
- credit  : 연준 조치가 없어도 HY <= HYmax - credit_drop 이면 2단계 통과로 인정.
- dd      : 감시 시작 하락 기준 (기본 0.20).
- stop    : 손절 폭. stop_ref='L'  -> QQQ 종가 < (매수 시점 L) x (1 - stop)
                     stop_ref='entry' -> QQQ 종가 < 1차 매수 시점 QQQ 종가 x (1 - stop)
                     stop_ref='asset' -> 매매 대상 종가 < 1차 매수가 x (1 - stop)
- asset   : 매매 대상 열 (기본 'qqq'; 'qld', 'tqqq'). 신호 판단은 언제나 QQQ.
            손절하면 전량 매도하고 2단계 통과 상태(첫 반등 대기)로 돌아가 L을 다시 갱신.
            같은 감시 안에서 다시 살 수 있음.
2차 매수, 매도(P1 재진입) 규칙은 기준과 같습니다.
P3 자금 곡선: 1차·2차에 반씩, 쉬는 돈은 T-bill 금리(^IRX)를 받음.
"""
import os
import numpy as np
import pandas as pd
from backtest import load, fed_actions, p1_reentry, month_edge_spike, yahoo, BASE

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = dict(BASE, trend=False, trend_drop=1.0, trend_ma=None, credit=False, credit_drop=1.0,
          stop=None, stop_ref="L")


def alarms(df, P):
    s1 = month_edge_spike(df.sofr_ior.fillna(-1), P["sofr_alarm"])
    s2 = month_edge_spike(df.srf.fillna(-1), P["srf_alarm"])
    a = {"HY": df.hy >= P["hy_alarm"], "MOVE": df.move >= P["move_alarm"],
         "SOFR-IORB": (df.sofr_ior >= P["sofr_alarm"]) & ~s1,
         "SRF": (df.srf >= P["srf_alarm"]) & ~s2}
    return a, a["HY"] | a["MOVE"] | a["SOFR-IORB"] | a["SRF"]


def run(df, P, fed, reentry, rf):
    al, alarm = alarms(df, P)
    q, hy, hi52, sma, sofr = (df[c].values for c in ("qqq", "hy", "hi52", "sma50", "sofr_ior"))
    x = df[P.get("asset", "qqq")].values   # 매매 대상 가격 (신호는 언제나 QQQ)
    tma = df.qqq.rolling(P["trend_ma"]).mean().values if P["trend_ma"] else np.full(len(df), -np.inf)
    idx = df.index
    trades, eps = [], []
    state, ep, tr = "IDLE", None, None
    cash, units, eq = 1.0, 0.0, np.empty(len(df))   # P3 자금 곡선
    half = None

    def close_trade(i, how):
        nonlocal cash, units, tr
        cash += units * x[i]; units = 0.0
        tr.update(exit=idx[i], exit_px=x[i], how=how)
        trades.append(tr); tr = None

    for i, d in enumerate(idx):
        cash *= 1 + rf.iloc[i]
        c, h = q[i], hy[i]
        if state == "IDLE":
            if c <= hi52[i] * (1 - P["dd"]) and alarm.iloc[i]:
                ep = dict(watch=d, alarm="/".join(k for k, s in al.items() if s.iloc[i]),
                          L=c, hymax=h, fed=None, gate=None)
                eps.append(ep)
                state = "WATCH"
            else:
                eq[i] = cash; continue
        ep["hymax"] = max(ep["hymax"], h)
        if reentry.iloc[i] and d > ep["watch"]:
            if tr is not None:
                close_trade(i, "매도(P1)")
            ep["end"] = d
            state, ep = "IDLE", None
            eq[i] = cash; continue
        if state in ("WATCH", "FED") and c < ep["L"]:
            ep["L"] = c
        if state == "WATCH":
            hits = fed[(fed.index >= ep["watch"]) & (fed.index <= d)]
            if len(hits):
                ep["fed"], ep["gate"] = hits.index[0], "연준"
                state = "FED"
            elif P["credit"] and h <= ep["hymax"] - P["credit_drop"]:
                ep["fed"], ep["gate"] = d, "신용"
                state = "FED"
        if state == "FED":
            if c >= ep["L"] * (1 + P["rebound"]):
                state = "REB"
            eq[i] = cash + units * x[i]; continue
        if state == "REB":
            path = None
            if c < ep["L"] * (1 - P["band_lo"]):
                ep["L"] = c; state = "FED"
            elif c <= ep["L"] * (1 + P["band_hi"]) and h <= ep["hymax"] - P["hy_drop1"]:
                path = "이중바닥"
            elif P["trend"] and c > sma[i] and c > tma[i] and h <= ep["hymax"] - P["trend_drop"]:
                path = "추세"
            if path:
                half = cash / 2
                units += half / x[i]; cash -= half
                tr = dict(watch=ep["watch"], gate=ep["gate"], path=path, b1=d, b1_px=x[i], b1_q=c,
                          L=ep["L"], b2=None, b2_px=None)
                state = "HOLD1"
            eq[i] = cash + units * x[i]; continue
        if state in ("HOLD1", "HOLD2"):
            if P["stop"] is not None:
                if P["stop_ref"] == "asset":   # 매매 대상 가격 기준
                    hit = x[i] < tr["b1_px"] * (1 - P["stop"])
                else:                          # QQQ 기준 (L 또는 1차 매수 시점 QQQ 종가)
                    ref = tr["L"] if P["stop_ref"] == "L" else tr["b1_q"]
                    hit = c < ref * (1 - P["stop"])
                if hit:
                    close_trade(i, "손절")
                    ep["L"] = c
                    state = "FED"
                    eq[i] = cash; continue
            if state == "HOLD1":
                so = sofr[i]
                if h <= ep["hymax"] - P["hy_drop2"] and c > sma[i] and (np.isnan(so) or so < P["sofr_ok"]):
                    units += cash / x[i]; cash = 0.0
                    tr.update(b2=d, b2_px=x[i])
                    state = "HOLD2"
        eq[i] = cash + units * x[i]
    if tr is not None:
        tr.update(exit=None, exit_px=x[-1], how="보유 중")
        trades.append(tr)
    return trades, eps, pd.Series(eq, index=idx), state


def trade_table(trades):
    rows = []
    for t in trades:
        r1 = t["exit_px"] / t["b1_px"] - 1
        r2 = t["exit_px"] / t["b2_px"] - 1 if t["b2"] is not None else 0.0
        rows.append({"감시": t["watch"].date(), "관문": t["gate"], "경로": t["path"],
                     "1차 매수": t["b1"].date(),
                     "2차 매수": t["b2"].date() if t["b2"] is not None else "-",
                     "종료": t["exit"].date() if t["exit"] is not None else "보유 중",
                     "방식": t["how"], "1차%": r1 * 100,
                     "2차%": r2 * 100 if t["b2"] is not None else np.nan,
                     "P3 자금%": (r1 + r2) / 2 * 100})
    return pd.DataFrame(rows)


def stats(trades, eq):
    t = trade_table(trades)
    done = t[t["종료"] != "보유 중"] if len(t) else t
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    mdd = (eq / eq.cummax() - 1).min()
    return {"매매": len(done),
            "승": int((done["P3 자금%"] > 0).sum()) if len(done) else 0,
            "최악%": round(done["P3 자금%"].min(), 1) if len(done) else np.nan,
            "최고%": round(done["P3 자금%"].max(), 1) if len(done) else np.nan,
            "누적 배수": round(np.prod(1 + done["P3 자금%"] / 100), 2) if len(done) else 1.0,
            "자금곡선 CAGR%": round((eq.iloc[-1] ** (1 / yrs) - 1) * 100, 2),
            "자금곡선 MDD%": round(mdd * 100, 1)}


def setup():
    df, _ = load()
    y = yahoo()
    rf = (y["^IRX"].ffill() / 100 / 252).reindex(df.index).ffill().fillna(0.0)
    return df, fed_actions(df.index), p1_reentry(df), rf


# 채택한 개선안: 하락 -15%, 1차 매수의 HY 0.3%p 조건 삭제, 1차 매수가 -10% 손절
FINAL = dict(dd=0.15, hy_drop1=0.0, stop=0.10, stop_ref="entry")

VARIANTS = {
    "기준 규칙": {},
    "개선안 (채택)": FINAL,
    "손절 L-10%": dict(stop=0.10),
    "손절 L-15%": dict(stop=0.15),
    "손절 매수가-10%": dict(stop=0.10, stop_ref="entry"),
    "손절 매수가-15%": dict(stop=0.15, stop_ref="entry"),
    "추세 경로": dict(trend=True),
    "신용 관문": dict(credit=True),
    "하락 -15%": dict(dd=0.15),
    "추세+신용": dict(trend=True, credit=True),
    "추세+신용+손절 L-10%": dict(trend=True, credit=True, stop=0.10),
    "추세+신용+손절 매수가-10%": dict(trend=True, credit=True, stop=0.10, stop_ref="entry"),
    "추세+신용+손절 매수가-15%": dict(trend=True, credit=True, stop=0.15, stop_ref="entry"),
    "추세+신용+하락-15%+손절 매수가-10%": dict(trend=True, credit=True, dd=0.15, stop=0.10, stop_ref="entry"),
    "추세200+손절 매수가-10%": dict(trend=True, trend_ma=200, stop=0.10, stop_ref="entry"),
    "HY 조건 삭제+손절 매수가-10%": dict(hy_drop1=0.0, stop=0.10, stop_ref="entry"),
    "개선안, 손절 없음": dict(dd=0.15, hy_drop1=0.0),
}

if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 30)
    df, fed, reentry, rf = setup()
    qbh = df.qqq / df.qqq.iloc[0]
    yrs = (df.index[-1] - df.index[0]).days / 365.25
    cash_cagr = ((1 + rf).prod() ** (1 / yrs) - 1) * 100
    print(f"비교: 현금(T-bill) CAGR {cash_cagr:.2f}%, QQQ 보유 CAGR {(qbh.iloc[-1] ** (1 / yrs) - 1) * 100:.2f}% "
          f"MDD {(qbh / qbh.cummax() - 1).min() * 100:.1f}%  ({df.index[0].date()}~{df.index[-1].date()})\n")
    rows = []
    for name, opt in VARIANTS.items():
        trades, eps, eq, state = run(df, {**V2, **opt}, fed, reentry, rf)
        rows.append({"변형": name, "감시": len(eps), **stats(trades, eq), "현재 단계": state})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(os.path.join(HERE, "v2_variants.csv"), index=False)

    trades, eps, eq, state = run(df, {**V2, **FINAL}, fed, reentry, rf)
    tt = trade_table(trades)
    print("\n개선안 매매 내역 (현재 단계:", state + ")")
    print(tt.round(1).to_string())
    tt.round(2).to_csv(os.path.join(HERE, "v2_trades.csv"), index=False)
    held = sum(((eq.index >= t["b1"]) & (eq.index < (t["exit"] or eq.index[-1]))).sum()
               for t in trades) / len(eq)
    print(f"투자 중인 날 비율 약 {held * 100:.1f}%")
