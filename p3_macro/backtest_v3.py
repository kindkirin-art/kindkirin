"""P3 v3: v2 채택안에 1차 매수 VIX 다이버전스 조건을 더한 백테스트, 1986-09 ~ 현재.

backtest_v2.py 의 run()을 그대로 옮기고 VIX 조건만 더했습니다.
채택 규칙(V3): 1차 매수일 VIX 종가 <= L일 VIX x 0.90
  L일 VIX = 최저 종가 L을 기록한 날의 VIX 종가. L이 바뀔 때(감시·연준 단계의 저점 갱신,
  재시험 실패, 손절) 함께 바꿉니다. 가격 조건만 맞고 VIX 조건이 안 맞으면 사지 않고 기다립니다.
VIX(data/vix.csv, Yahoo ^VIX)는 1990-01부터 있어서, 그 전(1986~1989)은 VIX 조건을 충족한 것으로 봅니다.

비교용 필터 (P 의 키)
- b1     : 1차 매수에 붙이는 조건
    ("lday", x)   VIX 종가 <= L일 VIX x (1 - x)                 ... 채택 (x = 0.10)
    ("peak", x)   VIX 종가 <= 감시 시작 이후 VIX 최고 종가 x (1 - x)
    ("sma", n, k) VIX 종가 <= VIX n일 단순이동평균 x k
- b2     : 2차 매수에 붙이는 조건 ("peak", x) 또는 ("lvl", v): VIX 종가 <= v
- asset  : 매매 대상 'qqq' / 'qld' / 'tqqq' (신호는 언제나 QQQ)

python backtest_v3.py  # → v3_filters.csv, v3_leverage.csv, v3_trades.csv
"""
import os
import numpy as np
import pandas as pd

from backtest_v2 import V2, FINAL, alarms, setup, trade_table, stats

HERE = os.path.dirname(os.path.abspath(__file__))

P0 = {**V2, **FINAL, "b1": None, "b2": None, "size": 0.5, "asset": "qqq"}
V3 = {**P0, "b1": ("lday", 0.10)}   # 채택: 1차 매수일 VIX <= L일 VIX x 0.90


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
    if kind == "lvl":
        return vix[i] <= a
    raise ValueError(kind)


def run(df, P, fed, reentry, rf):
    al, alarm = alarms(df, P)
    q, hy, hi52, sma, sofr = (df[c].values for c in ("qqq", "hy", "hi52", "sma50", "sofr_ior"))
    vix = df.vix.values
    vsma = {n: df.vix.rolling(n).mean().values for n in (5, 10, 20)}
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


def filter_variants():
    out = {"v2 (VIX 필터 없음)": {}}
    for x in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        out[f"1차: VIX <= L일 VIX x{1 - x:.2f}"] = dict(b1=("lday", x))
    for x in (0.1, 0.2, 0.3):
        out[f"1차: VIX <= 감시 후 최고 x{1 - x:.1f}"] = dict(b1=("peak", x))
    for n in (5, 10, 20):
        for k in (1.0, 1.05, 1.1):
            out[f"1차: VIX <= {n}일선 x{k:.2f}"] = dict(b1=("sma", n, k))
    for v in (30, 40):
        out[f"2차: VIX <= {v}"] = dict(b2=("lvl", v))
    out["2차: VIX <= 감시 후 최고 x0.6"] = dict(b2=("peak", 0.4))
    return out


def split_mult(trades, a=None, b=None):
    tt = [x for x in trades if (a is None or x["b1"] >= pd.Timestamp(a)) and (b is None or x["b1"] <= pd.Timestamp(b))]
    r = [((x["exit_px"] / x["b1_px"] - 1) + ((x["exit_px"] / x["b2_px"] - 1) if x["b2"] is not None else 0)) / 2
         for x in tt]
    return round(float(np.prod([1 + v for v in r])), 2) if r else 1.0


if __name__ == "__main__":
    pd.set_option("display.width", 250, "display.max_columns", 30)
    df, fed, reentry, rf = load()

    rows = []
    for name, opt in filter_variants().items():
        t, e, eq, st = run(df, {**P0, **opt}, fed, reentry, rf)
        rows.append({"변형": name, **stats(t, eq), "~2004 배수": split_mult(t, b="2004-12-31"),
                     "2005~ 배수": split_mult(t, a="2005-01-01")})
    grid = pd.DataFrame(rows)
    print(grid.to_string(index=False))
    grid.to_csv(os.path.join(HERE, "v3_filters.csv"), index=False)

    rows = []
    for asset in ("qqq", "qld", "tqqq"):
        for name, P in (("v2", P0), ("v3", V3)):
            t, e, eq, st = run(df, {**P, "asset": asset}, fed, reentry, rf)
            rows.append({"자산": asset.upper(), "규칙": name, **stats(t, eq)})
    lev = pd.DataFrame(rows)
    print("\n" + lev.to_string(index=False))
    lev.to_csv(os.path.join(HERE, "v3_leverage.csv"), index=False)

    t, e, eq, st = run(df, V3, fed, reentry, rf)
    tt = trade_table(t)
    tt["VIX"] = [round(x["b1_vix"], 1) for x in t]
    tt["L일 VIX"] = [round(x["vL"], 1) for x in t]
    print(f"\nv3 매매 내역 (감시 {len(e)}번, 현재 단계 {st})")
    print(tt.round(1).to_string())
    tt.round(2).to_csv(os.path.join(HERE, "v3_trades.csv"), index=False)
    held = sum(((eq.index >= x["b1"]) & (eq.index < (x["exit"] or eq.index[-1]))).sum() for x in t) / len(eq)
    print(f"투자 중인 날 비율 약 {held * 100:.1f}%")
    for s in (0.06, 0.08, 0.10, 0.12, 0.15, 0.20):
        t2, _, q2, _ = run(df, {**V3, "stop": s}, fed, reentry, rf)
        r = stats(t2, q2)
        print(f"손절 {s:.0%}: 누적 {r['누적 배수']} MDD {r['자금곡선 MDD%']}%")
