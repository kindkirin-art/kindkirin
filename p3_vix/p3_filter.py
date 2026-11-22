"""P3 v2(채택안)에 VIX 필터를 붙인 백테스트, 1986-09 ~ 현재.

p3_macro/backtest_v2.py 의 run()을 그대로 옮기고 VIX 조건만 더했습니다.
VIX(^VIX)는 1990-01부터 있어서, 그 전(1986~1989)은 VIX 조건을 충족한 것으로 봅니다.

필터 (P 의 키)
- b1     : 1차 매수에 붙이는 조건
    ("peak", x)  VIX 종가 <= 감시 시작 이후 VIX 최고 종가 x (1 - x)   ... 'VIX 낮은 고점'
    ("lday", x)  VIX 종가 <= L(최저 종가)을 찍은 날의 VIX x (1 - x)   ... 바닥 재시험 때 VIX 다이버전스
    ("fall", n)  VIX 종가 < VIX n일 단순이동평균                      ... VIX 하락 중
    ("sma", n, k) VIX 종가 <= VIX n일 단순이동평균 x k                ... VIX 재상승 중이 아님
- b2     : 2차 매수에 붙이는 조건 ("peak", x) 또는 ("lvl", v): VIX 종가 <= v
- size   : 1차 매수 비중 (기본 0.5). b1을 통과하지 못하면 매수하지 않음
- asset  : 매매 대상 'qqq' / 'qld' / 'tqqq' (신호는 언제나 QQQ)
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "p3_macro"))
from backtest_v2 import V2, FINAL, alarms, setup, trade_table, stats  # noqa: E402

P0 = {**V2, **FINAL, "b1": None, "b2": None, "size": 0.5, "asset": "qqq"}


def load():
    df, fed, reentry, rf = setup()
    v = pd.read_csv(os.path.join(HERE, "data", "vix.csv"), index_col=0, parse_dates=True)["^VIX"]
    df = df.copy()
    df["vix"] = v.reindex(df.index).ffill(limit=3)
    from leverage import add_qld
    df = add_qld(df)
    return df, fed, reentry, rf


def vok(cond, vix, i, ep, vsma):
    if cond is None or np.isnan(vix[i]):
        return True
    kind, a = cond[0], cond[1]
    if kind == "sma":                      # ("sma", n, k): VIX <= VIX n일선 x k
        return vix[i] <= vsma[a][i] * cond[2]
    if kind == "peak":
        return vix[i] <= ep["vmax"] * (1 - a)
    if kind == "lday":
        return np.isnan(ep["vL"]) or vix[i] <= ep["vL"] * (1 - a)
    if kind == "fall":
        return vix[i] < vsma[a][i]
    if kind == "lvl":
        return vix[i] <= a
    raise ValueError(kind)


def run(df, P, fed, reentry, rf):
    al, alarm = alarms(df, P)
    q, hy, hi52, sma, sofr = (df[c].values for c in ("qqq", "hy", "hi52", "sma50", "sofr_ior"))
    vix = df.vix.values
    vsma = {n: df.vix.rolling(n).mean().values for n in (5, 10, 20, 50)}
    x = df[P["asset"]].values
    idx = df.index
    trades, eps = [], []
    state, ep, tr = "IDLE", None, None
    cash, units, eq = 1.0, 0.0, np.empty(len(df))
    blocked = 0

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
                ep = dict(watch=d, L=c, hymax=h, fed=None, gate="연준", vmax=vix[i], vL=vix[i], blocked=0)
                eps.append(ep)
                state = "WATCH"
            else:
                eq[i] = cash; continue
        ep["hymax"] = max(ep["hymax"], h)
        if not np.isnan(vix[i]):
            ep["vmax"] = vix[i] if np.isnan(ep["vmax"]) else max(ep["vmax"], vix[i])
        if reentry.iloc[i] and d > ep["watch"]:
            if tr is not None:
                close_trade(i, "매도(P1)")
            ep["end"] = d
            state, ep = "IDLE", None
            eq[i] = cash; continue
        if state in ("WATCH", "FED") and c < ep["L"]:
            ep["L"], ep["vL"] = c, vix[i]
        if state == "WATCH":
            hits = fed[(fed.index >= ep["watch"]) & (fed.index <= d)]
            if len(hits):
                ep["fed"] = hits.index[0]
                state = "FED"
        if state == "FED":
            if c >= ep["L"] * (1 + P["rebound"]):
                state = "REB"
            eq[i] = cash + units * x[i]; continue
        if state == "REB":
            if c < ep["L"] * (1 - P["band_lo"]):
                ep["L"], ep["vL"] = c, vix[i]; state = "FED"
            elif c <= ep["L"] * (1 + P["band_hi"]) and h <= ep["hymax"] - P["hy_drop1"]:
                if vok(P["b1"], vix, i, ep, vsma):
                    amt = cash * P["size"]
                    units += amt / x[i]; cash -= amt
                    tr = dict(watch=ep["watch"], gate="연준", path="이중바닥", b1=d, b1_px=x[i], b1_q=c,
                              L=ep["L"], b2=None, b2_px=None, b1_vix=vix[i], vmax=ep["vmax"], vL=ep["vL"])
                    state = "HOLD1"
                else:
                    blocked += 1
            eq[i] = cash + units * x[i]; continue
        if state in ("HOLD1", "HOLD2"):
            if c < tr["b1_q"] * (1 - P["stop"]):
                close_trade(i, "손절")
                ep["L"], ep["vL"] = c, vix[i]
                state = "FED"
                eq[i] = cash; continue
            if state == "HOLD1":
                so = sofr[i]
                if (h <= ep["hymax"] - P["hy_drop2"] and c > sma[i] and (np.isnan(so) or so < P["sofr_ok"])
                        and vok(P["b2"], vix, i, ep, vsma)):
                    units += cash / x[i]; cash = 0.0
                    tr.update(b2=d, b2_px=x[i])
                    state = "HOLD2"
        eq[i] = cash + units * x[i]
    if tr is not None:
        tr.update(exit=None, exit_px=x[-1], how="보유 중")
        trades.append(tr)
    return trades, eps, pd.Series(eq, index=idx), state
