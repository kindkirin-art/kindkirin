"""P3 v2(채택안) 상태 머신을 돌려 날짜별 단계와 사건을 뽑습니다.

p3_macro/backtest_v2.py 의 run()과 같은 규칙이며, 매매 대신 단계만 기록합니다.
단계: IDLE(대기) WATCH(감시) FED(연준 항복, 첫 반등 대기) REB(첫 반등 후, 이중바닥 대기)
      HOLD1(1차 매수 후) HOLD2(2차 매수 후)
사건: watch, fed, rebound, buy1, buy2, stop, sell, vend(매수 없이 감시 종료)
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "p3_macro"))
from backtest_v2 import V2, FINAL, alarms, setup  # noqa: E402

P = {**V2, **FINAL}


def p3_states(df=None, fed=None, reentry=None, P=P):
    if df is None:
        df, fed, reentry, _ = setup()
    al, alarm = alarms(df, P)
    q, hy, hi52, sma, sofr = (df[c].values for c in ("qqq", "hy", "hi52", "sma50", "sofr_ior"))
    idx = df.index
    st = np.empty(len(df), dtype=object)
    ev = np.full(len(df), "", dtype=object)
    state, ep = "IDLE", None
    for i, d in enumerate(idx):
        c, h = q[i], hy[i]
        if state == "IDLE":
            if c <= hi52[i] * (1 - P["dd"]) and alarm.iloc[i]:
                ep = dict(watch=d, L=c, hymax=h, b1q=None)
                state, ev[i] = "WATCH", "watch"
            else:
                st[i] = state; continue
        ep["hymax"] = max(ep["hymax"], h)
        if reentry.iloc[i] and d > ep["watch"]:
            ev[i] = "sell" if state in ("HOLD1", "HOLD2") else "vend"
            state, ep = "IDLE", None
            st[i] = "IDLE"; continue
        if state in ("WATCH", "FED") and c < ep["L"]:
            ep["L"] = c
        if state == "WATCH":
            hits = fed[(fed.index >= ep["watch"]) & (fed.index <= d)]
            if len(hits):
                state = "FED"; ev[i] = (ev[i] + " fed").strip()
        if state == "FED":
            if c >= ep["L"] * (1 + P["rebound"]):
                state = "REB"; ev[i] = (ev[i] + " rebound").strip()
            st[i] = state; continue
        if state == "REB":
            if c < ep["L"] * (1 - P["band_lo"]):
                ep["L"] = c; state = "FED"
            elif c <= ep["L"] * (1 + P["band_hi"]) and h <= ep["hymax"] - P["hy_drop1"]:
                ep["b1q"] = c; state = "HOLD1"; ev[i] = "buy1"
            st[i] = state; continue
        if state in ("HOLD1", "HOLD2"):
            if c < ep["b1q"] * (1 - P["stop"]):
                ep["L"] = c; state = "FED"; ev[i] = "stop"
                st[i] = state; continue
            if state == "HOLD1":
                so = sofr[i]
                if h <= ep["hymax"] - P["hy_drop2"] and c > sma[i] and (np.isnan(so) or so < P["sofr_ok"]):
                    state = "HOLD2"; ev[i] = "buy2"
        st[i] = state
    out = pd.DataFrame({"state": st, "event": ev, "alarm": alarm.values}, index=idx)
    return out, df


if __name__ == "__main__":
    s, _ = p3_states()
    e = s[s.event != ""]
    print(e.to_string())
    print(s.state.value_counts())
